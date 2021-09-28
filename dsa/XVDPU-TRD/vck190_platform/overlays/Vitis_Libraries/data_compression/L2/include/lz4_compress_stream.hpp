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
#ifndef _XFCOMPRESSION_LZ4_COMPRESS_STREAM_HPP_
#define _XFCOMPRESSION_LZ4_COMPRESS_STREAM_HPP_

/**
 * @file lz4_compress_stream.hpp
 * @brief Header for LZ4 compression stream based kernel.
 *
 * This file is part of Vitis Data Compression Library.
 */

#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include "hls_stream.h"
#include "ap_axi_sdata.h"
#include <ap_int.h>

#include "kernel_stream_utils.hpp"
#include "lz_compress.hpp"
#include "lz_optional.hpp"
#include "lz4_compress.hpp"

#define MAX_LIT_COUNT 4096
#define MIN_MATCH 4
#define LZ_MAX_OFFSET_LIMIT 65536
#define OFFSET_WINDOW LZ_MAX_OFFSET_LIMIT
#define PARALLEL_BLOCK 8
#define MAX_MATCH_LEN 255
#define MATCH_LEN 6

// Kernel top functions
extern "C" {
/**
 * @brief LZ4 compression streaming kernel. It takes input from axi kernel stream and writes compressed
 *        data back to output axi kernel stream.
 *
 * @param inaxistream   Input axi kernel stream
 * @param outaxistream  Output axi kernel stream
 * @param inputSize     Input compressed data size
 */
void xilLz4CompressStream(hls::stream<ap_axiu<8, 0, 0, 0> >& inaxistream,
                          hls::stream<ap_axiu<8, 0, 0, 0> >& outaxistream,
                          uint32_t inputSize);
}

#endif // _XFCOMPRESSION_LZ4_COMPRESS_STREAM_HPP_
