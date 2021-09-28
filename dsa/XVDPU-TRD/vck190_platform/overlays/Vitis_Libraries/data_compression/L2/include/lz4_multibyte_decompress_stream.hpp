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
#ifndef _XFCOMPRESSION_LZ4_DECOMPRESS_STREAM_HPP_
#define _XFCOMPRESSION_LZ4_DECOMPRESS_STREAM_HPP_

/**
 * @file lz4_multibyte_decompress_stream.hpp
 * @brief C++ Header for lz4 multibyte decompression stream kernel.
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
#include "lz_decompress.hpp"
#include "lz4_decompress.hpp"

#define MAX_OFFSET 65536
#define HISTORY_SIZE MAX_OFFSET

#ifndef MULTIPLE_BYTES
#define MULTIPLE_BYTES 8
#endif

extern "C" {
/**
 * @brief Snappy decompression streaming kernel takes compressed data as input from kernel axi stream
 * and process in block based fashion and writes the raw data to global memory.
 *
 * @param inaxistream input kernel axi stream for compressed data
 * @param outaxistream output kernel axi stream for decompressed data
 * @param inputSize input data size
 */
void xilLz4DecompressStream(hls::stream<ap_axiu<MULTIPLE_BYTES * 8, 0, 0, 0> >& inaxistream,
                            hls::stream<ap_axiu<MULTIPLE_BYTES * 8, 0, 0, 0> >& outaxistream,
                            hls::stream<ap_axiu<32, 0, 0, 0> >& outaxistreamsize,
                            uint32_t inputSize);
}
#endif // _XFCOMPRESSION_SNAPPY_DECOMPRESS_STREAM_HPP_
