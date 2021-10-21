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
 * @file lz4_p2p_decompress_kernel.cpp
 * @brief Source for LZ4 decompression kernel.
 *
 * This file is part of Vitis Data Compression Library.
 */

#include "lz4_p2p_decompress_kernel.hpp"

#define MAX_OFFSET 65536
#define HISTORY_SIZE MAX_OFFSET
const int c_gmemBurstSize = (2 * GMEM_BURST_SIZE);

#define READ_STATE 0
#define MATCH_STATE 1
#define LOW_OFFSET_STATE 2

typedef ap_uint<8> uintV_t;

void lz4CoreDec(hls::stream<xf::compression::uintMemWidth_t>& inStreamMemWidth,
                hls::stream<xf::compression::uintMemWidth_t>& outStreamMemWidth,
                const uint32_t _input_size,
                const uint32_t _output_size,
                const uint32_t _input_start_idx) {
    uint32_t input_size = _input_size;
    uint32_t output_size = _output_size;
    uint32_t input_size1 = input_size;
    uint32_t output_size1 = output_size;
    uint32_t input_start_idx = _input_start_idx;
    hls::stream<uintV_t> instreamV("instreamV");
    hls::stream<ap_uint<32> > decompressd_stream("decompressd_stream");
    hls::stream<uintV_t> decompressed_stream("decompressed_stream");
#pragma HLS STREAM variable = instreamV depth = 8
#pragma HLS STREAM variable = decompressd_stream depth = 8
#pragma HLS STREAM variable = decompressed_stream depth = 8
#pragma HLS BIND_STORAGE variable = instreamV type = FIFO impl = SRL
#pragma HLS BIND_STORAGE variable = decompressd_stream type = FIFO impl = SRL
#pragma HLS BIND_STORAGE variable = decompressed_stream type = FIFO impl = SRL
    bool uncomp_flag = 0;
    if (input_size == output_size) uncomp_flag = 1;

#pragma HLS dataflow
    xf::compression::details::streamDownsizerP2P<uint32_t, GMEM_DWIDTH, 8>(inStreamMemWidth, instreamV, input_size,
                                                                           input_start_idx);
    xf::compression::lz4DecompressSimple(instreamV, decompressd_stream, input_size1, uncomp_flag);
    xf::compression::lzDecompress<HISTORY_SIZE>(decompressd_stream, decompressed_stream, output_size);
    xf::compression::details::streamUpsizer<uint32_t, 8, GMEM_DWIDTH>(decompressed_stream, outStreamMemWidth,
                                                                      output_size1);
}

void lz4Dec(const xf::compression::uintMemWidth_t* in,
            xf::compression::uintMemWidth_t* out,
            const uint32_t input_idx[PARALLEL_BLOCK],
            const uint32_t input_idx1[PARALLEL_BLOCK],
            const uint32_t input_size[PARALLEL_BLOCK],
            const uint32_t output_size[PARALLEL_BLOCK],
            const uint32_t input_size1[PARALLEL_BLOCK],
            const uint32_t output_size1[PARALLEL_BLOCK],
            const uint32_t output_idx[PARALLEL_BLOCK]) {
    hls::stream<xf::compression::uintMemWidth_t> inStreamMemWidth[PARALLEL_BLOCK];
    hls::stream<xf::compression::uintMemWidth_t> outStreamMemWidth[PARALLEL_BLOCK];
#pragma HLS STREAM variable = inStreamMemWidth depth = c_gmemBurstSize
#pragma HLS STREAM variable = outStreamMemWidth depth = c_gmemBurstSize
#pragma HLS BIND_STORAGE variable = inStreamMemWidth type = FIFO impl = SRL
#pragma HLS BIND_STORAGE variable = outStreamMemWidth type = FIFO impl = SRL

#pragma HLS dataflow
    // Transfer data from global memory to kernel
    xf::compression::details::mm2sNbRoundOff<GMEM_DWIDTH, GMEM_BURST_SIZE, PARALLEL_BLOCK>(
        in, input_idx, inStreamMemWidth, input_size);
    for (uint8_t i = 0; i < PARALLEL_BLOCK; i++) {
#pragma HLS UNROLL
        // lz4CoreDec is instantiated based on the PARALLEL_BLOCK
        lz4CoreDec(inStreamMemWidth[i], outStreamMemWidth[i], input_size1[i], output_size1[i], input_idx1[i]);
    }

    // Transfer data from kernel to global memory
    xf::compression::details::s2mmNb<uint32_t, GMEM_BURST_SIZE, GMEM_DWIDTH, PARALLEL_BLOCK>(
        out, output_idx, outStreamMemWidth, output_size);
}
//} // namespace end

extern "C" {

void xilLz4P2PDecompress(const xf::compression::uintMemWidth_t* in,
                         xf::compression::uintMemWidth_t* out,
                         dt_blockInfo* decompress_block_info,
                         dt_chunkInfo* decompress_chunk_info,
                         uint32_t block_size_in_kb,
                         uint32_t compute_unit,
                         uint8_t total_no_cu,
                         uint32_t num_blocks) {
#pragma HLS INTERFACE m_axi port = in offset = slave bundle = gmem
#pragma HLS INTERFACE m_axi port = out offset = slave bundle = gmem
#pragma HLS INTERFACE m_axi port = decompress_block_info offset = slave bundle = gmem
#pragma HLS INTERFACE m_axi port = decompress_chunk_info offset = slave bundle = gmem
#pragma HLS INTERFACE s_axilite port = in bundle = control
#pragma HLS INTERFACE s_axilite port = out bundle = control
#pragma HLS INTERFACE s_axilite port = decompress_block_info bundle = control
#pragma HLS INTERFACE s_axilite port = decompress_chunk_info bundle = control
#pragma HLS INTERFACE s_axilite port = block_size_in_kb bundle = control
#pragma HLS INTERFACE s_axilite port = compute_unit bundle = control
#pragma HLS INTERFACE s_axilite port = total_no_cu bundle = control
#pragma HLS INTERFACE s_axilite port = num_blocks bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

    uint32_t max_block_size = block_size_in_kb * 1024;
    uint32_t compress_size[PARALLEL_BLOCK];
    uint32_t compress_size1[PARALLEL_BLOCK];
    uint32_t block_size[PARALLEL_BLOCK];
    uint32_t block_size1[PARALLEL_BLOCK];
    uint32_t input_idx[PARALLEL_BLOCK];
    uint32_t input_idx1[PARALLEL_BLOCK];
    uint32_t output_idx[PARALLEL_BLOCK];
#pragma HLS ARRAY_PARTITION variable = input_idx dim = 0 complete
#pragma HLS ARRAY_PARTITION variable = input_idx1 dim = 0 complete
#pragma HLS ARRAY_PARTITION variable = output_idx dim = 0 complete
#pragma HLS ARRAY_PARTITION variable = compress_size dim = 0 complete
#pragma HLS ARRAY_PARTITION variable = compress_size1 dim = 0 complete
#pragma HLS ARRAY_PARTITION variable = block_size dim = 0 complete
#pragma HLS ARRAY_PARTITION variable = block_size1 dim = 0 complete

    uint32_t curr_no_blocks = decompress_chunk_info->numBlocksPerCU[compute_unit];
    int offset = num_blocks * compute_unit;
    // printf ("In decode compute unit %d no_blocks %d\n", D_COMPUTE_UNIT, curr_no_blocks);

    for (uint32_t i = 0; i < curr_no_blocks; i += PARALLEL_BLOCK) {
        uint32_t nblocks = PARALLEL_BLOCK;
        if ((i + PARALLEL_BLOCK) > curr_no_blocks) {
            nblocks = curr_no_blocks - i;
        }

        for (uint32_t j = 0; j < PARALLEL_BLOCK; j++) {
            dt_blockInfo bInfo = decompress_block_info[i + j + offset];
            if (j < nblocks) {
                uint32_t iSize = bInfo.compressedSize;
                uint32_t oSize = bInfo.blockSize;
                uint32_t startIdx = bInfo.blockStartIdx;
                compress_size[j] = iSize;
                block_size[j] = oSize;
                compress_size1[j] = iSize;
                block_size1[j] = oSize;
                input_idx[j] = startIdx;
                input_idx1[j] = startIdx;
                // printf("iSize:%d\toSize:%d\tblockIdx:%d\n", iSize, oSize, input_idx[j]);
                output_idx[j] = (i + j) * max_block_size;
            } else {
                compress_size[j] = 0;
                block_size[j] = 0;
                compress_size1[j] = 0;
                block_size1[j] = 0;
                input_idx[j] = 0;
                input_idx1[j] = 0;
                output_idx[j] = 0;
            }
        }

        lz4Dec(in, out, input_idx, input_idx1, compress_size, block_size, compress_size1, block_size1, output_idx);
    }
}
}
