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
#ifndef _XFCOMPRESSION_SNAPPY_DECOMPRESS_MM_HPP_
#define _XFCOMPRESSION_SNAPPY_DECOMPRESS_MM_HPP_

/**
 * @file snappy_decompress_mm.hpp
 * @brief C++ Header for snappy decompression kernel.
 *
 * This file is part of Vitis Data Compression Library.
 */
#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include "hls_stream.h"
#include <ap_int.h>

#include "lz_decompress.hpp"
#include "mm2s.hpp"
#include "s2mm.hpp"
#include "stream_downsizer.hpp"
#include "stream_upsizer.hpp"

#include "snappy_decompress_details.hpp"

#define GMEM_DWIDTH 64
#define GMEM_BURST_SIZE 512

#ifndef PARALLEL_BYTE
#define PARALLEL_BYTE 8
#endif

#define MAX_OFFSET 65536
#define HISTORY_SIZE MAX_OFFSET

extern "C" {
/**
 * @brief Snappy decompression kernel takes compressed data as input and process in
 * block based fashion and writes the raw data to global memory.
 *
 * @param in input compressed data
 * @param out output raw data
 * @param in_block_size input block size of each block
 * @param in_compress_size compress size of each block
 * @param block_size_in_kb block size in bytes
 * @param no_blocks number of blocks
 */
void xilSnappyDecompress(const ap_uint<PARALLEL_BYTE * 8>* in,
                         ap_uint<PARALLEL_BYTE * 8>* out,
                         uint32_t* in_block_size,
                         uint32_t* in_compress_size,
                         uint32_t block_size_in_kb,
                         uint32_t no_blocks);
}
#endif // _XFCOMPRESSION_SNAPPY_DECOMPRESS_MM_HPP_
