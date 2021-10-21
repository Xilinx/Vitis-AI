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
 * @file lz4_unpacker_kernel.cpp
 * @brief Source for LZ4 P2P uncompress kernel.
 *
 * This file is part of Vitis Data Compression Library.
 */

#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include "hls_stream.h"
#include <ap_int.h>

#include "lz4_unpacker_kernel.hpp"

/**
 * This value is used to set
 * uncompressed block size value.
 * 4th byte is always set to below
 * and placed as uncompressed byte
 */
#define NO_COMPRESS_BIT 128

/**
 * In case of uncompressed block
 * Values below are used to set
 * 3rd byte to following values
 * w.r.t various maximum block sizes
 * supported by standard
 */
#define BSIZE_NCOMP_64 1
#define BSIZE_NCOMP_256 4
#define BSIZE_NCOMP_1024 16
#define BSIZE_NCOMP_4096 64

/**
 * Below are the codes as per LZ4 standard for
 * various maximum block sizes supported.
 */
#define BSIZE_STD_64KB 0x40
#define BSIZE_STD_256KB 0x50
#define BSIZE_STD_1024KB 0x60
#define BSIZE_STD_4096KB 0x70

typedef ap_uint<GMEM_DWIDTH> uintMemWidth_t;

// Stream in_block_size, in_compress_size, block_start_idx to decompress kernel. And need to put Macro or use array
// based on number of compute units

extern "C" {
void xilLz4Unpacker(const xf::compression::uintMemWidth_t* in,
                    dt_blockInfo* unpacker_block_info,
                    dt_chunkInfo* unpacker_chunk_info,
                    uint32_t block_size_in_kb,
                    uint8_t first_chunk,
                    uint8_t total_no_cu,
                    uint32_t num_blocks) {
#pragma HLS INTERFACE m_axi port = in offset = slave bundle = gmem
#pragma HLS INTERFACE m_axi port = unpacker_block_info offset = slave bundle = gmem
#pragma HLS INTERFACE m_axi port = unpacker_chunk_info offset = slave bundle = gmem
#pragma HLS INTERFACE s_axilite port = in bundle = control
#pragma HLS INTERFACE s_axilite port = unpacker_block_info bundle = control
#pragma HLS INTERFACE s_axilite port = unpacker_chunk_info bundle = control
#pragma HLS INTERFACE s_axilite port = block_size_in_kb bundle = control
#pragma HLS INTERFACE s_axilite port = first_chunk bundle = control
#pragma HLS INTERFACE s_axilite port = total_no_cu bundle = control
#pragma HLS INTERFACE s_axilite port = num_blocks bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

    uint32_t block_size_in_bytes = block_size_in_kb * 1024;
    uint32_t max_no_blocks = num_blocks * total_no_cu;

    dt_chunkInfo cInfo;

    if (first_chunk) {
        uintMemWidth_t inTemp;
        /*Magic headers*/
        inTemp = in[0];
        uint8_t m1 = inTemp.range(7, 0);
        uint8_t m2 = inTemp.range(15, 8);
        uint8_t m3 = inTemp.range(23, 16);
        uint8_t m4 = inTemp.range(31, 24);

        /*Header checksum*/
        uint8_t hc = inTemp.range(39, 32);

        /*Block size*/
        uint32_t code = inTemp.range(47, 40);
        switch (code) {
            case BSIZE_STD_64KB:
                block_size_in_kb = 64;
                break;
            case BSIZE_STD_256KB:
                block_size_in_kb = 256;
                break;
            case BSIZE_STD_1024KB:
                block_size_in_kb = 1024;
                break;
            case BSIZE_STD_4096KB:
                block_size_in_kb = 4096;
                break;
            default:
                break;
        }
        block_size_in_bytes = block_size_in_kb * 1024;

        /*Original file size*/
        cInfo.originalSize = inTemp.range(111, 48);

        /*Calculate no of blocks based on original size of file*/
        cInfo.numBlocks = (cInfo.originalSize - 1) / block_size_in_bytes + 1;

        /*Initialize start index for first chunk*/
        cInfo.inStartIdx = 15;
    }

    uint32_t curr_no_blocks = (cInfo.numBlocks >= max_no_blocks) ? max_no_blocks : cInfo.numBlocks;

    cInfo.numBlocks = cInfo.numBlocks - curr_no_blocks;

    const int c_byte_size = 8;
    uint64_t inIdx = cInfo.inStartIdx;
    uint32_t Idx1 = (inIdx * c_byte_size) / GMEM_DWIDTH;
    uint32_t Idx2 = (inIdx * c_byte_size) % GMEM_DWIDTH;
    uint32_t compressed_size = 0;
    uint32_t compressed_size1 = 0;

    // struct object
    dt_blockInfo bInfo;

    for (uint32_t blkIdx = 0; blkIdx < curr_no_blocks; blkIdx++) {
#pragma HLS PIPELINE off
        if (Idx2 + 32 <= GMEM_DWIDTH) {
            uintMemWidth_t inTemp;
            inTemp = in[Idx1];
            compressed_size = inTemp.range(Idx2 + 32 - 1, Idx2);
        } else {
            uintMemWidth_t inTemp;
            uintMemWidth_t inTemp1;
            ap_uint<32> ctemp;
            inTemp = in[Idx1];
            int c_word_size = GMEM_DWIDTH / c_byte_size;
            inTemp1 = in[Idx1 + 1];
            ctemp = (inTemp1.range(Idx2 + 32 - GMEM_DWIDTH - 1, 0), inTemp.range(GMEM_DWIDTH - 1, Idx2));
            compressed_size = ctemp;
        }
        inIdx = inIdx + 4;
        uint32_t tmp;
        tmp = compressed_size;
        tmp >>= 24;
        if (tmp == NO_COMPRESS_BIT) {
            uint8_t b1 = compressed_size;
            uint8_t b2 = compressed_size >> 8;
            uint8_t b3 = compressed_size >> 16;
            if (b3 == BSIZE_NCOMP_64 || b3 == BSIZE_NCOMP_4096 || b3 == BSIZE_NCOMP_256 || b3 == BSIZE_NCOMP_1024) {
                compressed_size = block_size_in_bytes;
            } else {
                uint32_t size = 0;
                size = b3;
                size <<= 16;
                uint32_t temp = b2;
                temp <<= 8;
                size |= temp;
                temp = b1;
                size |= temp;
                compressed_size = size;
            }
        }
        bInfo.blockStartIdx = inIdx;
        bInfo.compressedSize = compressed_size;
        bInfo.blockSize = block_size_in_bytes;
        unpacker_block_info[blkIdx] = bInfo;
        // printf("blockStartIdx:%d\tcompressSize:%d\tblock_size_in_bytes:%d\n",
        // unpacker_block_info[blkIdx].blockStartIdx,
        //     unpacker_block_info[blkIdx].compressedSize, unpacker_block_info[blkIdx].blockSize);
        inIdx = inIdx + compressed_size;
        Idx1 = (inIdx * c_byte_size) / GMEM_DWIDTH;
        Idx2 = (inIdx * c_byte_size) % GMEM_DWIDTH;
    }
    cInfo.inStartIdx = inIdx;

    if (cInfo.numBlocks == 0) {
        unpacker_block_info[curr_no_blocks - 1].blockSize = cInfo.originalSize % (block_size_in_bytes);
        // If original size is multiple of block size
        if (unpacker_block_info[curr_no_blocks - 1].blockSize == 0) // If original size is multiple of block size
            unpacker_block_info[curr_no_blocks - 1].blockSize = block_size_in_bytes;
    }

    for (int i = 0; i < total_no_cu; i++) {
        if (curr_no_blocks > num_blocks)
            cInfo.numBlocksPerCU[i] = num_blocks;
        else
            cInfo.numBlocksPerCU[i] = curr_no_blocks;
        curr_no_blocks = curr_no_blocks - num_blocks;
    }

    *unpacker_chunk_info = cInfo;
}
}
