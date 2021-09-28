/*
 * Copyright 2020 Xilinx, Inc.
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
 */

#include "convert.hpp"

#ifndef __SYNTHESIS__
#include <iostream>
#endif

extern "C" void convertCsrCsc_kernel(int vertexNum,
                                     int edgeNum,
                                     uint512 offsetG1[V],
                                     uint512 indexG1[E],
                                     uint512 offsetG2[V],
                                     DT indexG2[E * 16],
                                     uint512 offsetG2Tmp1[V],
                                     uint512 offsetG2Tmp2[V]) {
#pragma HLS INTERFACE m_axi offset = slave bundle = gmem0_0 port = offsetG1 latency = 64 num_read_outstanding = \
    16 max_read_burst_length = 32
#pragma HLS INTERFACE m_axi offset = slave bundle = gmem0_1 port = indexG1 latency = 64 num_read_outstanding = \
    16 max_read_burst_length = 32
#pragma HLS INTERFACE m_axi offset = slave bundle = gmem0_2 port = offsetG2 latency = 64 num_write_outstanding = \
    16 max_write_burst_length = 32
#pragma HLS INTERFACE m_axi offset = slave bundle = gmem0_3 port = indexG2 latency = 64 num_write_outstanding = \
    16 max_write_burst_length = 32
#pragma HLS INTERFACE m_axi offset = slave bundle = gmem0_4 port = offsetG2Tmp1 latency = 64 num_read_outstanding = \
    16 max_read_burst_length = 32 num_write_outstanding = 16 max_write_burst_length = 32
#pragma HLS INTERFACE m_axi offset = slave bundle = gmem0_5 port = offsetG2Tmp2 latency = 64 num_write_outstanding = \
    16 num_read_outstanding = 16 max_write_burst_length = 32 max_read_burst_length = 32

#pragma HLS INTERFACE s_axilite port = vertexNum bundle = control
#pragma HLS INTERFACE s_axilite port = edgeNum bundle = control
#pragma HLS INTERFACE s_axilite port = offsetG1 bundle = control
#pragma HLS INTERFACE s_axilite port = indexG1 bundle = control
#pragma HLS INTERFACE s_axilite port = offsetG2 bundle = control
#pragma HLS INTERFACE s_axilite port = indexG2 bundle = control
#pragma HLS INTERFACE s_axilite port = offsetG2Tmp1 bundle = control
#pragma HLS INTERFACE s_axilite port = offsetG2Tmp2 bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

    const int cacheDepthBin = 0;  // cache line depth in Binary
    const int dataOneLineBin = 4; // data numbers in Binary of one buffType
    const int usURAM = 0;         // 0 represents use LUTRAM, 1 represents use URAM, 2 represents use BRAM

    xf::graph::convertCsrCsc<DT, V, E, cacheDepthBin, dataOneLineBin, usURAM>(
        edgeNum, vertexNum, offsetG1, indexG1, offsetG2, indexG2, offsetG2Tmp1, offsetG2Tmp2);
}
