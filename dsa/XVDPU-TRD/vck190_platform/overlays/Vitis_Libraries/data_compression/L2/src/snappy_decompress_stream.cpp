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
 * @file snappy_decompress_stream.cpp
 * @brief Source for snappy decompression kernel.
 *
 * This file is part of Vitis Compression Library.
 */
#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include "hls_stream.h"
#include <ap_int.h>

#include "snappy_decompress_stream.hpp"

typedef ap_uint<8> streamDt;

extern "C" {

void xilSnappyDecompressStream(hls::stream<ap_axiu<8, 0, 0, 0> >& inaxistream,
                               hls::stream<ap_axiu<8, 0, 0, 0> >& outaxistream,
                               uint32_t inputSize,
                               uint32_t outputSize) {
#pragma HLS interface axis port = inaxistream
#pragma HLS interface axis port = outaxistream
#pragma HLS interface s_axilite port = inputSize bundle = control
#pragma HLS interface s_axilite port = outputSize bundle = control
#pragma HLS interface s_axilite port = return bundle = control

    hls::stream<xf::compression::compressd_dt> decompressedStream("decompressedStream");
    hls::stream<streamDt> inStream("inStream");
    hls::stream<streamDt> outStream("outStream");

#pragma HLS STREAM variable = inStream depth = 2
#pragma HLS STREAM variable = outStream depth = 2
#pragma HLS STREAM variable = decompressedStream depth = 8

#pragma HLS dataflow

    xf::compression::details::kStreamRead<8>(inaxistream, inStream, inputSize);

    xf::compression::snappyDecompress(inStream, decompressedStream, inputSize);
    xf::compression::lzDecompress<HISTORY_SIZE>(decompressedStream, outStream, outputSize);

    xf::compression::details::kStreamWriteFixedSize<8>(outaxistream, outStream, outputSize);
}
}
