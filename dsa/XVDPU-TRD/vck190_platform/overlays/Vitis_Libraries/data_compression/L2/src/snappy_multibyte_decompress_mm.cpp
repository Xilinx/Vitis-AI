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

#include "snappy_multibyte_decompress_mm.hpp"

const int c_parallelBit = PARALLEL_BYTE * 8;
const int historySize = MAX_OFFSET;
const int c_gmemBurstSize = (2 * GMEM_BURST_SIZE);

typedef ap_uint<c_parallelBit> uintpV_t;
// namespace  hw_decompress {

void snappyCoreDec(hls::stream<ap_uint<PARALLEL_BYTE * 8> >& inStream,
                   hls::stream<ap_uint<(PARALLEL_BYTE * 8) + 8> >& outStream,
                   const uint32_t _input_size) {
    uint32_t input_size = _input_size;
    hls::stream<uint32_t> decStreamSize;
    hls::stream<uint32_t> blockCompSize;

    // send each block compressed size and 0 to indicate end of data
    blockCompSize << input_size;
    blockCompSize << 0;

#pragma HLS dataflow
    xf::compression::snappyDecompressCoreEngine<PARALLEL_BYTE, historySize>(inStream, outStream, decStreamSize,
                                                                            blockCompSize);
    {
        uint32_t outsize = decStreamSize.read(); // Dummy module to empty SizeStream
    }
}

void snappyDec(const ap_uint<PARALLEL_BYTE * 8>* in,
               ap_uint<PARALLEL_BYTE * 8>* out,
               const uint32_t input_idx,
               const uint32_t input_size,
               const uint32_t input_size1) {
    const int c_byteSize = 8;
    const int c_wordSize = (PARALLEL_BYTE * 8) / c_byteSize;

    uint32_t rIdx = input_idx / c_wordSize;

    hls::stream<ap_uint<PARALLEL_BYTE * 8> > inStream;
    hls::stream<ap_uint<(PARALLEL_BYTE * 8) + 8> > outStream;
#pragma HLS STREAM variable = inStream depth = c_gmemBurstSize
#pragma HLS STREAM variable = outStream depth = c_gmemBurstSize
#pragma HLS BIND_STORAGE variable = inStream type = FIFO impl = SRL
#pragma HLS BIND_STORAGE variable = outStream type = FIFO impl = SRL

#pragma HLS dataflow
    // Transfer data from global memory to kernel
    xf::compression::details::mm2sSimple<PARALLEL_BYTE * 8, GMEM_BURST_SIZE>(&(in[rIdx]), inStream, input_size);

    // Snappy Single Instance
    snappyCoreDec(inStream, outStream, input_size1);

    // Transfer data from kernel to global memory
    xf::compression::details::s2mmEosStreamSimple<PARALLEL_BYTE * 8, GMEM_BURST_SIZE>(&(out[rIdx]), outStream);
}

//}//end of namepsace

extern "C" {

void xilSnappyDecompress(const ap_uint<PARALLEL_BYTE * 8>* in,
                         ap_uint<PARALLEL_BYTE * 8>* out,
                         uint32_t* in_block_size,
                         uint32_t* in_compress_size,
                         uint32_t block_size_in_kb,
                         uint32_t no_blocks) {
#pragma HLS INTERFACE m_axi port = in offset = slave bundle = gmem max_read_burst_length = 128 num_read_outstanding = \
    4 max_write_burst_length = 128 num_write_outstanding = 4
#pragma HLS INTERFACE m_axi port = out offset = slave bundle = gmem max_read_burst_length = 128 num_read_outstanding = \
    4 max_write_burst_length = 128 num_write_outstanding = 4
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
    uint32_t compress_size;
    uint32_t compress_size1;
    uint32_t input_idx;

    for (uint32_t i = 0; i < no_blocks; i++) {
        uint32_t iSize = in_compress_size[i];
        compress_size = iSize;
        compress_size1 = iSize;
        input_idx = i * max_block_size;

        // Single Engine LZ4 Decompression
        snappyDec(in, out, input_idx, compress_size, compress_size1);
    }
}
}
