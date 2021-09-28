/*
 * (c) Copyright 2021 Xilinx, Inc. All rights reserved.
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
#ifndef _XFCOMPRESSION_ZSTD_COMPRESS_STREAM_HPP_
#define _XFCOMPRESSION_ZSTD_COMPRESS_STREAM_HPP_

/**
 * @file zstd_compress_stream.hpp
 * @brief Header for Zstd compression kernel.
 *
 * This file is part of Vitis Data Compression Library.
 */

#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include "hls_stream.h"
#include <ap_int.h>

#include "zstd_compress.hpp"
#include "ap_axi_sdata.h"

#ifndef STREAM_IN_DWIDTH
#define STREAM_IN_DWIDTH 8
#endif

#ifndef STREAM_OUT_DWIDTH
#define STREAM_OUT_DWIDTH 32
#endif

// ZStd Block size and Window Size (lz history size)
#ifndef ZSTD_BLOCK_SIZE_KB
#define ZSTD_BLOCK_SIZE_KB 32
#endif

// const int c_streamDWidth = 8 * MULTIPLE_BYTES;
// window size is kept equal to block size in this design
constexpr int c_windowSize = ZSTD_BLOCK_SIZE_KB * 1024;
constexpr int c_blockSize = ZSTD_BLOCK_SIZE_KB * 1024;

#ifndef MIN_BLCK_SIZE
#define MIN_BLCK_SIZE 128
#endif

// Kernel top functions
extern "C" {
/**
 * @brief ZSTD compression kernel takes input data from axi stream and compresses it
 * into multiple frames having 1 block each and writes the compressed data to output axi stream.
 *
 *
 * @param inStream input raw data
 * @param outStream output compressed data
 */
void xilZstdCompress(hls::stream<ap_axiu<STREAM_IN_DWIDTH, 0, 0, 0> >& axiInStream,
                     hls::stream<ap_axiu<STREAM_OUT_DWIDTH, 0, 0, 0> >& axiOutStream);
}
#endif // _XFCOMPRESSION_ZSTD_COMPRESS_STREAM_HPP_
