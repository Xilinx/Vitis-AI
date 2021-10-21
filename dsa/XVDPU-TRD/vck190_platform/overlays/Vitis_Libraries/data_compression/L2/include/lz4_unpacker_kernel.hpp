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
#ifndef _XFCOMPRESSION_LZ4_UNPACKER_KERNEL_HPP_
#define _XFCOMPRESSION_LZ4_UNPACKER_KERNEL_HPP_

/**
 * @file lz4_unpacker_kernel.hpp
 * @brief Header for Unpacker kernel.
 *
 * This file is part of Vitis Data Compression Library.
 */

#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include "hls_stream.h"
#include <ap_int.h>

#include "mm2s.hpp"
#include "s2mm.hpp"
#include "stream_downsizer.hpp"
#include "stream_upsizer.hpp"
#include "lz4_p2p.hpp"
#define GMEM_DWIDTH 512

// Kernel top functions
extern "C" {

/**
 * @brief LZ4 unpacker kernel is responsible in unpacking LZ4 compressed block
 * information.
 *
 * @param in input stream width
 * @param in_block_size input block size
 * @param in_compress_size input compress size
 * @param block_start_idx start index of each input block
 * @param no_blocks_per_cu number of blocks for each compute unit
 * @param original_size original file size
 * @param in_start_index input start index
 * @param no_blocks number of blocks
 * @param block_size_in_kb size of each block
 * @param first_chunk first chunk to determine header
 * @param total_no_cu number of decompress compute units
 * @param num_blocks number of blocks based on host buffersize
 */
void xilLz4Unpacker(const xf::compression::uintMemWidth_t* in,
                    dt_blockInfo* bObj,
                    dt_chunkInfo* cObj,
                    uint32_t block_size_in_kb,
                    uint8_t first_chunk,
                    uint8_t total_no_cu,
                    uint32_t num_blocks);
}

#endif // _XFCOMPRESSION_LZ4_UNPACKER_KERNEL_HPP_
