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

#ifndef _XFCOMPRESSION_LZ4_P2P_HPP_
#define _XFCOMPRESSION_LZ4_P2P_HPP_

/**
 * @file lz4_p2p.hpp
 * @brief Block and Chunk data required for LZ4 P2P.
 *
 * This file is part of XF Compression Library.
 */

#include <stdint.h>
#define GMEM_DATAWIDTH 512

// structure size explicitly made equal to 64Bytes so that it will match
// to Kernel Global Memory datawidth (512bit).
typedef struct unpackerBlockInfo {
    uint32_t compressedSize;
    uint32_t blockSize;
    uint32_t blockStartIdx;
    uint32_t padding[(GMEM_DATAWIDTH / 32) - 3];
} dt_blockInfo;

// structure size explicitly made equal to 64Bytes so that it will match
// to Kernel Global Memory datawidth (512bit).
typedef struct unpackerChunkInfo {
    uint32_t inStartIdx;
    uint32_t originalSize;
    uint32_t numBlocks;
    uint32_t numBlocksPerCU[2];
    uint32_t padding[(GMEM_DATAWIDTH / 32) - 5];
} dt_chunkInfo;

#endif // _XFCOMPRESSION_LZ4_P2P_HPP_
