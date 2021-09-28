/*
 * (c) Copyright 2019-2021 Xilinx, Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */
/**
 * @file snappy_decompress_mm.cpp
 * @brief Source for snappy decompression kernel.
 *
 * This file is part of Vitis Data Compression Library.
 */
#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include "hls_stream.h"
#include <ap_int.h>

#include "snappy_decompress_mm.hpp"

typedef ap_uint<8> uintV_t;
const int c_gmemBurstSize = (2 * GMEM_BURST_SIZE);

// namespace  hw_decompress {

void snappyCoreDec(hls::stream<xf::compression::uintMemWidth_t>& inStreamMemWidth,
                   hls::stream<xf::compression::uintMemWidth_t>& outStreamMemWidth,
                   const uint32_t _input_size,
                   const uint32_t _output_size) {
    uint32_t input_size = _input_size;
    uint32_t output_size = _output_size;
    uint32_t input_size1 = input_size;
    uint32_t output_size1 = output_size;
    hls::stream<uintV_t> instreamV("instreamV");
    hls::stream<xf::compression::compressd_dt> decompressd_stream("decompressd_stream");
    hls::stream<uintV_t> decompressed_stream("decompressed_stream");
#pragma HLS STREAM variable = instreamV depth = 8
#pragma HLS STREAM variable = decompressd_stream depth = 8
#pragma HLS STREAM variable = decompressed_stream depth = 8
#pragma HLS BIND_STORAGE variable = instreamV type = FIFO impl = SRL
#pragma HLS BIND_STORAGE variable = decompressd_stream type = FIFO impl = SRL
#pragma HLS BIND_STORAGE variable = decompressed_stream type = FIFO impl = SRL

#pragma HLS dataflow
    xf::compression::details::streamDownsizer<uint32_t, GMEM_DWIDTH, 8>(inStreamMemWidth, instreamV, input_size);
    xf::compression::snappyDecompress(instreamV, decompressd_stream, input_size1);
    xf::compression::lzDecompress<HISTORY_SIZE>(decompressd_stream, decompressed_stream, output_size);
    xf::compression::details::streamUpsizer<uint32_t, 8, GMEM_DWIDTH>(decompressed_stream, outStreamMemWidth,
                                                                      output_size1);
}

void snappyDec(const xf::compression::uintMemWidth_t* in,
               xf::compression::uintMemWidth_t* out,
               const uint32_t input_idx[PARALLEL_BLOCK],
               const uint32_t input_size[PARALLEL_BLOCK],
               const uint32_t output_size[PARALLEL_BLOCK],
               const uint32_t input_size1[PARALLEL_BLOCK],
               const uint32_t output_size1[PARALLEL_BLOCK]) {
    hls::stream<xf::compression::uintMemWidth_t> inStreamMemWidth[PARALLEL_BLOCK];
    hls::stream<xf::compression::uintMemWidth_t> outStreamMemWidth[PARALLEL_BLOCK];

#pragma HLS STREAM variable = inStreamMemWidth depth = c_gmemBurstSize
#pragma HLS STREAM variable = outStreamMemWidth depth = c_gmemBurstSize
#pragma HLS BIND_STORAGE variable = inStreamMemWidth type = FIFO impl = SRL
#pragma HLS BIND_STORAGE variable = outStreamMemWidth type = FIFO impl = SRL

#pragma HLS dataflow
    // Transfer data from global memory to kernel
    xf::compression::details::mm2sNb<GMEM_DWIDTH, GMEM_BURST_SIZE, PARALLEL_BLOCK>(in, input_idx, inStreamMemWidth,
                                                                                   input_size);
    for (uint8_t i = 0; i < PARALLEL_BLOCK; i++) {
#pragma HLS UNROLL
        snappyCoreDec(inStreamMemWidth[i], outStreamMemWidth[i], input_size1[i], output_size1[i]);
    }
    // Transfer data from kernel to global memory
    xf::compression::details::s2mmNb<uint32_t, GMEM_BURST_SIZE, GMEM_DWIDTH, PARALLEL_BLOCK>(
        out, input_idx, outStreamMemWidth, output_size);
}

//}//end of namepsace

extern "C" {

void xilSnappyDecompress(const xf::compression::uintMemWidth_t* in,
                         xf::compression::uintMemWidth_t* out,
                         uint32_t* in_block_size,
                         uint32_t* in_compress_size,
                         uint32_t block_size_in_kb,
                         uint32_t no_blocks) {
#pragma HLS INTERFACE m_axi port = in offset = slave bundle = gmem0
#pragma HLS INTERFACE m_axi port = out offset = slave bundle = gmem0
#pragma HLS INTERFACE m_axi port = in_block_size offset = slave bundle = gmem1
#pragma HLS INTERFACE m_axi port = in_compress_size offset = slave bundle = gmem1
#pragma HLS INTERFACE s_axilite port = in bundle = control
#pragma HLS INTERFACE s_axilite port = out bundle = control
#pragma HLS INTERFACE s_axilite port = in_block_size bundle = control
#pragma HLS INTERFACE s_axilite port = in_compress_size bundle = control
#pragma HLS INTERFACE s_axilite port = block_size_in_kb bundle = control
#pragma HLS INTERFACE s_axilite port = no_blocks bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

    uint32_t max_block_size = block_size_in_kb * 1024;
    uint32_t compress_size[PARALLEL_BLOCK];
    uint32_t compress_size1[PARALLEL_BLOCK];
    uint32_t block_size[PARALLEL_BLOCK];
    uint32_t block_size1[PARALLEL_BLOCK];
    uint32_t input_idx[PARALLEL_BLOCK];
#pragma HLS ARRAY_PARTITION variable = input_idx dim = 0 complete
#pragma HLS ARRAY_PARTITION variable = compress_size dim = 0 complete
#pragma HLS ARRAY_PARTITION variable = compress_size1 dim = 0 complete
#pragma HLS ARRAY_PARTITION variable = block_size dim = 0 complete
#pragma HLS ARRAY_PARTITION variable = block_size1 dim = 0 complete

    for (int i = 0; i < no_blocks; i += PARALLEL_BLOCK) {
        int nblocks = PARALLEL_BLOCK;
        if ((i + PARALLEL_BLOCK) > no_blocks) {
            nblocks = no_blocks - i;
        }

        for (int j = 0; j < PARALLEL_BLOCK; j++) {
            if (j < nblocks) {
                uint32_t iSize = in_compress_size[i + j];
                uint32_t oSize = in_block_size[i + j];
                compress_size[j] = iSize;
                block_size[j] = oSize;
                compress_size1[j] = iSize;
                block_size1[j] = oSize;
                input_idx[j] = (i + j) * max_block_size;
            } else {
                compress_size[j] = 0;
                block_size[j] = 0;
                compress_size1[j] = 0;
                block_size1[j] = 0;
                input_idx[j] = 0;
            }
        }
        snappyDec(in, out, input_idx, compress_size, block_size, compress_size1, block_size1);
    }
}
}
