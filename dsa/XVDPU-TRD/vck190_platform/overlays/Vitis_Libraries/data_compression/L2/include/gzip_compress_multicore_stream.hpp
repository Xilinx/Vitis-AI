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
#ifndef _XFCOMPRESSION_GZIP_COMPRESS_MULTICORE_STREAM_HPP_
#define _XFCOMPRESSION_GZIP_COMPRESS_MULTICORE_STREAM_HPP_

/**
 * @file gzip_compress_multicore_stream.hpp
 * @brief Header for Gzip multicore streaming compression kernel.
 *
 * This file is part of Vitis Data Compression Library.
 */

#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include "hls_stream.h"
#include <ap_int.h>
#include "compress_utils.hpp"

#include "kernel_stream_utils.hpp"
#include "zlib_compress.hpp"
#include "checksum_wrapper.hpp"

#ifndef GMEM_DWIDTH
#define GMEM_DWIDTH 64
#endif

#ifndef URAM_BUFFER
#define URAM_BUFFER 0 // 0: BRAM; 1: URAM
#endif

#ifndef STRTGY
#define STRTGY 0 // GZIP
#endif

#ifndef NUM_CORES
#define NUM_CORES 8 // Octacore by default
#endif

#ifndef BLOCKSIZE_IN_KB
#define BLOCKSIZE_IN_KB 32 // 32KB by default
#endif

// Kernel top functions
extern "C" {
/**
 * @brief GZIP streaming compression kernel takes the raw data as input from axi interface and compresses the data
 * using num cores and writes the output to an axi interface.
 *
 * @param inaxistream input raw data
 * @param outaxistream output compressed data
 */
void xilGzipComp(hls::stream<ap_axiu<GMEM_DWIDTH, 0, 0, 0> >& inaxistream,
                 hls::stream<ap_axiu<GMEM_DWIDTH, 0, 0, 0> >& outaxistream);
}
#endif // _XFCOMPRESSION_GZIP_COMPRESS_MULTICORE_STREAM_HPP_
