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
#ifndef _XFCOMPRESSION_ZLIB_COMPRESS_MM_HPP_
#define _XFCOMPRESSION_ZLIB_COMPRESS_MM_HPP_

/**
 * @file zlib_compress_multi_engine_mm.hpp
 * @brief Header for Zlib compression kernel.
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
#include "zlib_compress.hpp"

#ifndef GMEM_DWIDTH
#define GMEM_DWIDTH 256
#endif

#ifndef GMEM_BURST_SIZE
#define GMEM_BURST_SIZE 32
#endif

#ifndef ZLIB_BLOCK_SIZE
#define ZLIB_BLOCK_SIZE 0x8000 // 32 KB
#endif

#ifndef PARALLEL_BLOCK
#define PARALLEL_BLOCK 4
#endif

// Kernel top functions
extern "C" {
/**
 * @brief ZLIB compression kernel takes the raw data as input and compresses the data
 * in parallel block based fashion and writes the output to global memory.
 *
 * @param in input raw data
 * @param out output compressed data
 * @param compressd_size compressed output size of each block
 * @param input_size input data size
 */
void xilZlibCompressFull(const ap_uint<GMEM_DWIDTH>* in,
                         ap_uint<GMEM_DWIDTH>* out,
                         uint32_t* compressd_size,
                         uint32_t input_size);
}
#endif // _XFCOMPRESSION_ZLIB_COMPRESS_MM_HPP_
