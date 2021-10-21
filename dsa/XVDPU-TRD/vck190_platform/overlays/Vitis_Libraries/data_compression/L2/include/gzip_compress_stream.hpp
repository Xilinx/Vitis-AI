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
#ifndef _XFCOMPRESSION_GZIP_COMPRESS_STREAM_HPP_
#define _XFCOMPRESSION_GZIP_COMPRESS_STREAM_HPP_

/**
 * @file gzip_compress_stream.hpp
 * @brief Header for Gzip compression kernel.
 *
 * This file is part of Vitis Data Compression Library.
 */

#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include "hls_stream.h"
#include <ap_int.h>

#include "zlib_compress.hpp"

#ifndef GMEM_IN_DWIDTH
#define GMEM_IN_DWIDTH 8
#endif

#ifndef GMEM_OUT_DWIDTH
#define GMEM_OUT_DWIDTH 16
#endif

#ifndef ZLIB_BLOCK_SIZE
#define ZLIB_BLOCK_SIZE 32768
#endif

#ifndef STRATEGY
#define STRATEGY 0 // Gzip
#endif

#ifndef MIN_BLCK_SIZE
#define MIN_BLCK_SIZE 1024
#endif

// Kernel top functions
extern "C" {
/**
 * @brief GZIP compression kernel takes the raw data as input and compresses the data
 * in block based fashion and writes the output to global memory.
 *
 * @param inStream input raw data
 * @param outStream output compressed data
 * @param inSizeStream input data size
 */
void xilGzipCompressStreaming(hls::stream<ap_axiu<GMEM_IN_DWIDTH, 0, 0, 0> >& inStream,
                              hls::stream<ap_axiu<GMEM_OUT_DWIDTH, 0, 0, 0> >& outStream,
                              hls::stream<ap_axiu<32, 0, 0, 0> >& inSizeStream);
}
#endif // _XFCOMPRESSION_GZIP_COMPRESS_STREAM_HPP_
