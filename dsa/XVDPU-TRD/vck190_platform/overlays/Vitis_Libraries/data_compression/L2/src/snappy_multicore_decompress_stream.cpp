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

#include "snappy_multicore_decompress_stream.hpp"

const int c_parallelBit = MULTIPLE_BYTES * 8;
const int historySize = HISTORY_SIZE;

extern "C" {

void xilSnappyDecompressStream(hls::stream<ap_axiu<c_parallelBit, 0, 0, 0> >& inaxistream,
                               hls::stream<ap_axiu<32, 0, 0, 0> >& inaxistreamsize,
                               hls::stream<ap_axiu<c_parallelBit, 0, 0, 0> >& outaxistream,
                               hls::stream<ap_axiu<32, 0, 0, 0> >& outaxistreamsize) {
// For free running kernel, user needs to specify ap_ctrl_none for return port.
// This will create the kernel without AXI lite interface. Kernel will always be
// in running states.
#ifndef DISABLE_FREE_RUNNING_KERNEL
#pragma HLS interface ap_ctrl_none port = return
#endif

    hls::stream<ap_uint<c_parallelBit> > inStream("inStream");
    hls::stream<uint32_t> inSizeStream;
    hls::stream<ap_uint<c_parallelBit + 8> > decompressedStream("decompressedStream");
    hls::stream<uint32_t> decStreamSize;
#pragma HLS STREAM variable = inStream depth = 32
#pragma HLS STREAM variable = decompressedStream depth = 32

#pragma HLS BIND_STORAGE variable = inStream type = FIFO impl = SRL
#pragma HLS BIND_STORAGE variable = decompressedStream type = FIFO impl = SRL

#pragma HLS dataflow
    xf::compression::details::kStreamDataRead<c_parallelBit>(inaxistream, inaxistreamsize, inStream, inSizeStream);

    xf::compression::snappyMultiCoreDecompress<NUM_CORES, MULTIPLE_BYTES, historySize>(
        inStream, inSizeStream, decompressedStream, decStreamSize);

    xf::compression::details::kStreamWriteMultiByteSize<c_parallelBit>(outaxistream, outaxistreamsize,
                                                                       decompressedStream, decStreamSize);
}
}
