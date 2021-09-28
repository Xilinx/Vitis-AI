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
#ifndef _XFCOMPRESSION_SNAPPY_COMPRESS_STREAM_HPP_
#define _XFCOMPRESSION_SNAPPY_COMPRESS_STREAM_HPP_

/**
 * @file snappy_compress_kernel.hpp
 * @brief C++ Header for snappy compression kernel.
 *
 * This file is part of XF Compression Library.
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

#include "snappy_compress.hpp"

#define PARALLEL_BLOCK 8
#define MAX_MATCH_LEN 64
#define OFFSET_WINDOW 65536
#define MIN_MATCH 4
#define LZ_MAX_OFFSET_LIMIT 65536
#define MATCH_LEN 6

#ifdef LARGE_LIT_RANGE
#define MAX_LIT_COUNT 4090
#define MAX_LIT_STREAM_SIZE 4096
#else
#define MAX_LIT_COUNT 60
#define MAX_LIT_STREAM_SIZE 64
#endif

extern "C" {
/**
 * @brief Snappy compression streaming kernel takes the raw data as input from kernel axi stream
 * and compresses the data in block based fashion and writes the output to kernel axi stream.
 *
 * @param inaxistream input kernel axi stream for raw data
 * @param outaxistream output kernel axi stream for compressed data
 * @param inputSize input data size
 */
void xilSnappyCompressStream(hls::stream<ap_axiu<8, 0, 0, 0> >& inaxistream,
                             hls::stream<ap_axiu<8, 0, 0, 0> >& outaxistream,
                             uint32_t inputSize);
}

#endif // _XFCOMPRESSION_SNAPPY_COMPRESS_STREAM_HPP_
