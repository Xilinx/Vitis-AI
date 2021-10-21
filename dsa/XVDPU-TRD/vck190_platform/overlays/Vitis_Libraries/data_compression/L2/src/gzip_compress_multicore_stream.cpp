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
 * @file gzip_compress_multicore_stream.cpp
 * @brief Source for Gzip compression multicore kernel.
 *
 * This file is part of Vitis Data Compression Library.
 */

#include "gzip_compress_multicore_stream.hpp"

extern "C" {
/**
 * @brief GZIP streaming compression kernel takes the raw data as input from axi interface and compresses the data
 * using num cores and writes the output to an axi interface.
 *
 * @param inaxistream input raw data
 * @param outaxistream output compressed data
 * @param insizeaxistream input size
 * @param outsizeaxistream compressed size
 */

void xilGzipComp(hls::stream<ap_axiu<GMEM_DWIDTH, 0, 0, 0> >& inaxistream,
                 hls::stream<ap_axiu<GMEM_DWIDTH, 0, 0, 0> >& outaxistream) {
// For free running kernel, user needs to specify ap_ctrl_none for return port.
// This will create the kernel without AXI lite interface. Kernel will always be
// in running states.
#ifndef DISABLE_FREE_RUNNING_KERNEL
#pragma HLS interface ap_ctrl_none port = return
#endif

#pragma HLS dataflow
    xf::compression::gzipMulticoreCompressAxiStream<BLOCKSIZE_IN_KB, NUM_CORES, STRTGY>(inaxistream, outaxistream);
}
}
