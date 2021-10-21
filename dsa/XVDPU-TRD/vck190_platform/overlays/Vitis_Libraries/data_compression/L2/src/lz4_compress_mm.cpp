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
 * @file lz4_compress_mm.cpp
 * @brief Source for LZ4 compression kernel.
 *
 * This file is part of Vitis Data Compression Library.
 */

#include "lz4_compress_mm.hpp"

const int c_sizeStreamDepth = 8;
const int c_lz4MaxLiteralCount = MAX_LIT_COUNT;

// namespace hw_compress {

void lz4Core(hls::stream<uintInV_t>& inStream,
             hls::stream<uintOutV_t>& outStream,
             hls::stream<bool>& outStreamEos,
             hls::stream<uint32_t>& compressedSize,
             uint32_t max_lit_limit[PARALLEL_BLOCK],
             uint32_t input_size,
             uint32_t core_idx) {
    hls::stream<xf::compression::compressd_dt> compressdStream("compressdStream");
    hls::stream<xf::compression::compressd_dt> bestMatchStream("bestMatchStream");
    hls::stream<xf::compression::compressd_dt> boosterStream("boosterStream");
#pragma HLS STREAM variable = compressdStream depth = 8
#pragma HLS STREAM variable = bestMatchStream depth = 8
#pragma HLS STREAM variable = boosterStream depth = 8

#pragma HLS BIND_STORAGE variable = compressdStream type = FIFO impl = SRL
#pragma HLS BIND_STORAGE variable = boosterStream type = FIFO impl = SRL

#pragma HLS dataflow
    xf::compression::lzCompress<MATCH_LEN, MIN_MATCH, LZ_MAX_OFFSET_LIMIT>(inStream, compressdStream, input_size);
    xf::compression::lzBestMatchFilter<MATCH_LEN, OFFSET_WINDOW>(compressdStream, bestMatchStream, input_size);
    xf::compression::lzBooster<MAX_MATCH_LEN>(bestMatchStream, boosterStream, input_size);
    xf::compression::lz4Compress<MAX_LIT_COUNT, PARALLEL_BLOCK>(boosterStream, outStream, max_lit_limit, input_size,
                                                                outStreamEos, compressedSize, core_idx);
}

/**
 * @brief LZ4 compression kernel top.
 *
 * @param in input stream width
 * @param out output stream width
 * @param input_idx output size
 * @param output_idx input size
 * @param input_size input size
 * @param max_lit_limit input size
 */
void lz4(const xf::compression::uintMemWidth_t* in,
         xf::compression::uintMemWidth_t* out,
         const uint32_t input_idx[PARALLEL_BLOCK],
         const uint32_t output_idx[PARALLEL_BLOCK],
         const uint32_t input_size[PARALLEL_BLOCK],
         uint32_t output_size[PARALLEL_BLOCK],
         uint32_t max_lit_limit[PARALLEL_BLOCK]) {
    hls::stream<uintInV_t> inStream[PARALLEL_BLOCK];
    hls::stream<bool> outStreamEos[PARALLEL_BLOCK];
    hls::stream<uintOutV_t> outStream[PARALLEL_BLOCK];
#pragma HLS STREAM variable = outStreamEos depth = 2
#pragma HLS STREAM variable = inStream depth = c_gmemBurstSize
#pragma HLS STREAM variable = outStream depth = c_gmemBurstSize

#pragma HLS BIND_STORAGE variable = outStreamEos type = FIFO impl = SRL
#pragma HLS BIND_STORAGE variable = inStream type = FIFO impl = SRL
#pragma HLS BIND_STORAGE variable = outStream type = FIFO impl = SRL

    hls::stream<uint32_t> compressedSize[PARALLEL_BLOCK];

#pragma HLS dataflow
    xf::compression::details::mm2multStreamSize<8, PARALLEL_BLOCK, GMEM_DWIDTH, GMEM_BURST_SIZE>(in, input_idx,
                                                                                                 inStream, input_size);

    for (uint8_t i = 0; i < PARALLEL_BLOCK; i++) {
#pragma HLS UNROLL
        // lz4Core is instantiated based on the PARALLEL_BLOCK
        lz4Core(inStream[i], outStream[i], outStreamEos[i], compressedSize[i], max_lit_limit, input_size[i], i);
    }

    xf::compression::details::multStream2MM<8, PARALLEL_BLOCK, GMEM_DWIDTH, GMEM_BURST_SIZE>(
        outStream, outStreamEos, compressedSize, output_idx, out, output_size);
}
//} // namespace end

extern "C" {
/**
 * @brief LZ4 compression kernel.
 *
 * @param in input stream width
 * @param out output stream width
 * @param compressd_size output size
 * @param in_block_size input size
 * @param block_size_in_kb input size
 * @param input_size input size
 */
void xilLz4Compress

    (const xf::compression::uintMemWidth_t* in,
     xf::compression::uintMemWidth_t* out,
     uint32_t* compressd_size,
     uint32_t* in_block_size,
     uint32_t block_size_in_kb,
     uint32_t input_size) {
#pragma HLS INTERFACE m_axi port = in offset = slave bundle = gmem0
#pragma HLS INTERFACE m_axi port = out offset = slave bundle = gmem0
#pragma HLS INTERFACE m_axi port = compressd_size offset = slave bundle = gmem1
#pragma HLS INTERFACE m_axi port = in_block_size offset = slave bundle = gmem1
#pragma HLS INTERFACE s_axilite port = in bundle = control
#pragma HLS INTERFACE s_axilite port = out bundle = control
#pragma HLS INTERFACE s_axilite port = compressd_size bundle = control
#pragma HLS INTERFACE s_axilite port = in_block_size bundle = control
#pragma HLS INTERFACE s_axilite port = block_size_in_kb bundle = control
#pragma HLS INTERFACE s_axilite port = input_size bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

    uint32_t block_idx = 0;
    uint32_t block_length = block_size_in_kb * 1024;
    uint32_t no_blocks = (input_size - 1) / block_length + 1;
    uint32_t max_block_size = block_size_in_kb * 1024;

    bool small_block[PARALLEL_BLOCK];
    uint32_t input_block_size[PARALLEL_BLOCK];
    uint32_t input_idx[PARALLEL_BLOCK];
    uint32_t output_idx[PARALLEL_BLOCK];
    uint32_t output_block_size[PARALLEL_BLOCK];
    uint32_t max_lit_limit[PARALLEL_BLOCK];
    uint32_t small_block_inSize[PARALLEL_BLOCK];
#pragma HLS ARRAY_PARTITION variable = input_block_size dim = 0 complete
#pragma HLS ARRAY_PARTITION variable = input_idx dim = 0 complete
#pragma HLS ARRAY_PARTITION variable = output_idx dim = 0 complete
#pragma HLS ARRAY_PARTITION variable = output_block_size dim = 0 complete
#pragma HLS ARRAY_PARTITION variable = max_lit_limit dim = 0 complete

    // Figure out total blocks & block sizes
    for (uint32_t i = 0; i < no_blocks; i += PARALLEL_BLOCK) {
        uint32_t nblocks = PARALLEL_BLOCK;
        if ((i + PARALLEL_BLOCK) > no_blocks) {
            nblocks = no_blocks - i;
        }

        for (uint32_t j = 0; j < PARALLEL_BLOCK; j++) {
            if (j < nblocks) {
                uint32_t inBlockSize = in_block_size[i + j];
                if (inBlockSize < MIN_BLOCK_SIZE) {
                    small_block[j] = 1;
                    small_block_inSize[j] = inBlockSize;
                    input_block_size[j] = 0;
                    input_idx[j] = 0;
                } else {
                    small_block[j] = 0;
                    input_block_size[j] = inBlockSize;
                    input_idx[j] = (i + j) * max_block_size;
                    output_idx[j] = (i + j) * max_block_size;
                }
            } else {
                input_block_size[j] = 0;
                input_idx[j] = 0;
            }
            output_block_size[j] = 0;
            max_lit_limit[j] = 0;
        }

        // Call for parallel compression
        lz4(in, out, input_idx, output_idx, input_block_size, output_block_size, max_lit_limit);

        for (uint32_t k = 0; k < nblocks; k++) {
            if (max_lit_limit[k]) {
                compressd_size[block_idx] = input_block_size[k];
            } else {
                compressd_size[block_idx] = output_block_size[k];
            }

            if (small_block[k] == 1) {
                compressd_size[block_idx] = small_block_inSize[k];
            }
            block_idx++;
        }
    }
}
}
