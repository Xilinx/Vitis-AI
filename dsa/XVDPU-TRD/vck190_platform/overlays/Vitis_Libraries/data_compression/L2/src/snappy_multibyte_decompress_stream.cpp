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

#include "snappy_multibyte_decompress_stream.hpp"

const int c_parallelBit = MULTIPLE_BYTES * 8;
const int historySize = HISTORY_SIZE;

extern "C" {

void xilSnappyDecompressStream(hls::stream<ap_axiu<c_parallelBit, 0, 0, 0> >& inaxistream,
                               hls::stream<ap_axiu<c_parallelBit, 0, 0, 0> >& outaxistream,
                               hls::stream<ap_axiu<32, 0, 0, 0> >& outaxistreamsize,
                               uint32_t inputSize) {
#pragma HLS interface axis port = inaxistream
#pragma HLS interface axis port = outaxistream
#pragma HLS interface axis port = outaxistreamsize
#pragma HLS interface s_axilite port = inputSize bundle = control
#pragma HLS interface s_axilite port = return bundle = control

    hls::stream<ap_uint<c_parallelBit> > inStream("inStream");
    hls::stream<ap_uint<c_parallelBit + 8> > decompressedStream("decompressedStream");
    hls::stream<uint32_t> decStreamSize;
#pragma HLS STREAM variable = inStream depth = 32
#pragma HLS STREAM variable = decompressedStream depth = 32

#pragma HLS BIND_STORAGE variable = inStream type = FIFO impl = SRL
#pragma HLS BIND_STORAGE variable = decompressedStream type = FIFO impl = SRL

#pragma HLS dataflow

    xf::compression::details::kStreamRead<c_parallelBit>(inaxistream, inStream, inputSize);

    xf::compression::snappyDecompressEngine<MULTIPLE_BYTES, historySize>(inStream, decompressedStream, decStreamSize,
                                                                         inputSize);

    xf::compression::details::kStreamWriteMultiByteSize<c_parallelBit>(outaxistream, outaxistreamsize,
                                                                       decompressedStream, decStreamSize);
}
}
