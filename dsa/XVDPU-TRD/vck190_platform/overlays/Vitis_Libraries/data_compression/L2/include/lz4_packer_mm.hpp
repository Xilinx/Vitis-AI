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
#ifndef _XFCOMPRESSION_LZ4_PACKER_MM_HPP_
#define _XFCOMPRESSION_LZ4_PACKER_MM_HPP_

/**
 * @file lz4_packer_mm.hpp
 * @brief Header for LZ4 packer kernel.
 *
 * This file is part of Vitis Data Compression Library.
 */

#include <ap_int.h>
#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include "hls_stream.h"
#include "lz4_packer.hpp"

#define GMEM_DWIDTH 512
#define GMEM_BURST_SIZE 16

typedef ap_uint<GMEM_DWIDTH> uint512_t;

// Kernel top functions
extern "C" {
/**
 * @brief LZ4 packer kernel takes the raw data as input and compresses the data
 * in block based fashion and writes the output to global memory.
 *
 * @param in input raw data
 * @param out output compressed data
 * @param compressd_size compressed output size of each block
 * @param in_block_size input block size of each block
 * @param encoded_size encoded size of each block
 * @param orig_input_data raw input data
 * @param block_size_in_kb input block size in bytes
 * @param no_blocks number of input blocks
 * @param xxhashVal Hash Value
 * @param input_size Total Input File Size
 */
void xilLz4Packer(uint512_t* in,
                  uint512_t* out,
                  uint32_t* compressd_size,
                  uint32_t* in_block_size,
                  uint32_t* encoded_size,
                  uint512_t* orig_input_data,
                  uint32_t block_size_in_kb,
                  uint32_t no_blocks,
                  uint32_t xxhashVal,
                  uint32_t input_size);
}
#endif // _XFCOMPRESSION_LZ4_PACKER_MM_HPP_
