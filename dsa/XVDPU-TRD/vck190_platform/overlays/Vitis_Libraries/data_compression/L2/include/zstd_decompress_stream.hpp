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

#ifndef _XFCOMPRESSION_ZSTD_DECOMPRESS_STREAM_HPP_
#define _XFCOMPRESSION_ZSTD_DECOMPRESS_STREAM_HPP_

/**
 * @file zstd_decompress_stream.hpp
 * @brief Header for zstd decompression streaming kernel.
 *
 * This file is part of Vitis Data Compression Library.
 */

#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include <ap_int.h>
#include "hls_stream.h"
#include "zstd_decompress.hpp"
#include "ap_axi_sdata.h"

// ZStd Block size and Window Size (lz history size)
#ifndef ZSTD_BLOCK_SIZE_KB
#define ZSTD_BLOCK_SIZE_KB 32
#endif

#ifndef MULTIPLE_BYTES
#define MULTIPLE_BYTES 4
#endif

const int c_streamDWidth = 8 * MULTIPLE_BYTES;
// window size is kept equal to block size in this design
const int c_windowSize = ZSTD_BLOCK_SIZE_KB * 1024;

extern "C" {
/**
 * @brief This is full ZStandard decompression streaming kernel function. It supports all block
 * sizes and supports window size upto 128KB. It takes entire ZStd compressed file as input
 * and produces decompressed file at the kernel output stream. This kernel does not use DDR memory,
 * it uses streams instead. Intermediate data is stored in internal BRAMs and stream FIFOs, which helps
 * to attain better decompression throughput.
 *
 * @param input_size input size
 * @param inaxistreamd input kernel axi stream
 * @param outaxistreamd output kernel axi stream
 * @param sizestreamd output size kernel axi stream
 *
 */
void xilZstdDecompressStream(hls::stream<ap_axiu<c_streamDWidth, 0, 0, 0> >& inaxistreamd,
                             hls::stream<ap_axiu<c_streamDWidth, 0, 0, 0> >& outaxistreamd);
}
#endif // _XFCOMPRESSION_ZSTD_DECOMPRESS_STREAM_HPP_
