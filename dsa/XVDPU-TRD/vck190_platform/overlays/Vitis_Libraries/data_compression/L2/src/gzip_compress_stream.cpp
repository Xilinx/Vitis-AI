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
/**
 * @file gzip_compress_stream.cpp
 * @brief Source for GZIP compression kernel.
 *
 * This file is part of Vitis Data Compression Library.
 */

#ifndef _XFCOMPRESSION_GZIP_COMPRESS_STREAM_CPP_
#define _XFCOMPRESSION_GZIP_COMPRESS_STREAM_CPP_

#include "gzip_compress_stream.hpp"

extern "C" {
/**
 * @brief Gzip compression kernel.
 *
 * @param inStream input raw data
 * @param outStream output compressed data
 * @param inSizeStream input data size
 */
void xilGzipCompressStreaming(hls::stream<ap_axiu<GMEM_IN_DWIDTH, 0, 0, 0> >& inStream,
                              hls::stream<ap_axiu<GMEM_OUT_DWIDTH, 0, 0, 0> >& outStream,
                              hls::stream<ap_axiu<32, 0, 0, 0> >& inSizeStream) {
#ifndef DISABLE_FREE_RUNNING_KERNEL
#pragma HLS interface ap_ctrl_none port = return
#endif

    xf::compression::zlibCompressStreaming<GMEM_IN_DWIDTH, GMEM_OUT_DWIDTH, ZLIB_BLOCK_SIZE, STRATEGY, MIN_BLCK_SIZE>(
        inStream, outStream, inSizeStream);
}
}
#endif
