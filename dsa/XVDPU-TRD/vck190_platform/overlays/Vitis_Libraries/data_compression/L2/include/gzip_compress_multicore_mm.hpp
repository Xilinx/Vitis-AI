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
#ifndef _XFCOMPRESSION_GZIP_COMPRESS_MULTICORE_MM_HPP_
#define _XFCOMPRESSION_GZIP_COMPRESS_MULTICORE_MM_HPP_

/**
 * @file gzip_compress_multicore_mm.hpp
 * @brief Header for Gzip compression multicore kernel.
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
#define GMEM_DWIDTH 64
#endif

#ifndef GMEM_BURST_SIZE
#define GMEM_BURST_SIZE 128
#endif

#ifndef NUM_CORES
#define NUM_CORES 8 // Octacore by default
#endif

#ifndef BLOCKSIZE_IN_KB
#define BLOCKSIZE_IN_KB 32
#endif

// Kernel top functions
extern "C" {
/**
 * @brief GZIP compression kernel takes the raw data as input from DDR and compresses the data
 * using num cores and writes the output to global memory.
 *
 * @param in input raw data
 * @param out output compressed data
 * @param compressd_size compressed output size of each block
 * @param input_size input data size
 */
void xilGzipCompBlock(const ap_uint<GMEM_DWIDTH>* in,
                      ap_uint<GMEM_DWIDTH>* out,
                      uint32_t* compressd_size,
                      uint32_t* checksumData,
                      uint32_t input_size,
                      bool checksumType);
}
#endif // _XFCOMPRESSION_GZIP_COMPRESS_MULTICORE_MM_HPP_
