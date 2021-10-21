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

#include "label_propagation_top.hpp"

#ifndef __SYNTHESIS__
#include <iostream>
#endif

extern "C" void LPKernel(int vertexNum,
                         int edgeNum,
                         int iterNum,
                         uint512 offsetCSR[V],
                         uint512 indexCSR[E],
                         uint512 offsetCSC[V],
                         uint512 indexCSC[E],
                         DT indexCSC2[E * K],
                         uint512 pingHashBuf[V],
                         uint512 pongHashBuf[V],
                         uint512 labelPing[V],
                         uint512 labelPong[V]) {
#pragma HLS INTERFACE m_axi offset = slave latency = 64 num_read_outstanding = 16 max_read_burst_length = 32 bundle = \
    gmem0_0 port = offsetCSR
#pragma HLS INTERFACE m_axi offset = slave latency = 64 num_read_outstanding = 16 max_read_burst_length = 32 bundle = \
    gmem0_1 port = indexCSR
#pragma HLS INTERFACE m_axi offset = slave latency = 64 num_write_outstanding = 16 num_read_outstanding = \
    16 max_write_burst_length = 32 max_read_burst_length = 32 bundle = gmem0_2 port = offsetCSC
#pragma HLS INTERFACE m_axi offset = slave latency = 64 num_write_outstanding = 16 num_read_outstanding = \
    16 max_write_burst_length = 32 max_read_burst_length = 32 bundle = gmem0_3 port = indexCSC
#pragma HLS INTERFACE m_axi offset = slave latency = 64 num_write_outstanding = 16 num_read_outstanding = \
    16 max_write_burst_length = 32 max_read_burst_length = 32 bundle = gmem0_4 port = indexCSC2
#pragma HLS INTERFACE m_axi offset = slave latency = 64 num_write_outstanding = 16 num_read_outstanding = \
    16 max_write_burst_length = 32 max_read_burst_length = 32 bundle = gmem0_5 port = pingHashBuf
#pragma HLS INTERFACE m_axi offset = slave latency = 64 num_write_outstanding = 16 num_read_outstanding = \
    16 max_write_burst_length = 32 max_read_burst_length = 32 bundle = gmem0_6 port = pongHashBuf
#pragma HLS INTERFACE m_axi offset = slave latency = 32 num_write_outstanding = 16 num_read_outstanding = \
    16 max_write_burst_length = 32 max_read_burst_length = 32 bundle = gmem0_7 port = labelPing
#pragma HLS INTERFACE m_axi offset = slave latency = 32 num_write_outstanding = 16 num_read_outstanding = \
    16 max_write_burst_length = 32 max_read_burst_length = 32 bundle = gmem0_8 port = labelPong

#pragma HLS INTERFACE s_axilite port = vertexNum bundle = control
#pragma HLS INTERFACE s_axilite port = edgeNum bundle = control
#pragma HLS INTERFACE s_axilite port = iterNum bundle = control
#pragma HLS INTERFACE s_axilite port = offsetCSR bundle = control
#pragma HLS INTERFACE s_axilite port = indexCSR bundle = control
#pragma HLS INTERFACE s_axilite port = offsetCSC bundle = control
#pragma HLS INTERFACE s_axilite port = indexCSC bundle = control
#pragma HLS INTERFACE s_axilite port = indexCSC2 bundle = control
#pragma HLS INTERFACE s_axilite port = pingHashBuf bundle = control
#pragma HLS INTERFACE s_axilite port = pongHashBuf bundle = control
#pragma HLS INTERFACE s_axilite port = labelPing bundle = control
#pragma HLS INTERFACE s_axilite port = labelPong bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

    const int cacheDepthBin = 0;  // cache line depth in Binary
    const int dataOneLineBin = 4; // data numbers in Binary of one buffType
    const int usURAM = 0;         // 0 represents use LUTRAM, 1 represents use URAM, 2 represents use BRAM

    xf::graph::convertCsrCsc<DT, V, E, cacheDepthBin, dataOneLineBin, usURAM>(
        edgeNum, vertexNum, offsetCSR, indexCSR, offsetCSC, indexCSC2, labelPing, labelPong);
    xf::graph::labelPropagation(edgeNum, vertexNum, iterNum, offsetCSR, indexCSR, offsetCSC, indexCSC, pingHashBuf,
                                pongHashBuf, labelPing, labelPong);
}
