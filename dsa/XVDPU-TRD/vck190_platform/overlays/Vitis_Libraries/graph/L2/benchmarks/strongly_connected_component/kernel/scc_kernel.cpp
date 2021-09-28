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

#include "scc_kernel.hpp"

extern "C" void scc_kernel(const int edgeNum,
                           const int vertexNum,

                           ap_uint<512>* columnG1,
                           ap_uint<512>* offsetG1,
                           ap_uint<512>* column512G2,
                           ap_uint<32>* column32G2,
                           ap_uint<512>* offsetG2,
                           ap_uint<512>* columnG3,
                           ap_uint<512>* offsetG3,

                           ap_uint<512>* offsetG2Tmp1,
                           ap_uint<512>* offsetG2Tmp2,

                           ap_uint<512>* colorMap512G1,
                           ap_uint<32>* colorMap32G1,
                           ap_uint<32>* queueG1,
                           ap_uint<512>* colorMap512G2,
                           ap_uint<32>* colorMap32G2,
                           ap_uint<32>* queueG2,
                           ap_uint<32>* queueG3,

                           ap_uint<32>* result) {
#pragma HLS INTERFACE m_axi offset = slave latency = 32 num_read_outstanding = 32 max_read_burst_length = 32 bundle = \
    gmem0_0 port = columnG1
#pragma HLS INTERFACE m_axi offset = slave latency = 32 num_read_outstanding = 32 max_read_burst_length = 2 bundle = \
    gmem0_1 port = offsetG1

#pragma HLS INTERFACE m_axi offset = slave latency = 32 num_write_outstanding = 16 num_read_outstanding = \
    16 max_write_burst_length = 32 max_read_burst_length = 32 bundle = gmem0_2 port = column512G2
#pragma HLS INTERFACE m_axi offset = slave latency = 32 num_write_outstanding = 32 max_write_burst_length = \
    32 bundle = gmem1_0 port = column32G2
#pragma HLS INTERFACE m_axi offset = slave latency = 32 num_write_outstanding = 16 num_read_outstanding = \
    16 max_write_burst_length = 32 max_read_burst_length = 32 bundle = gmem0_3 port = offsetG2

#pragma HLS INTERFACE m_axi offset = slave latency = 32 num_read_outstanding = 32 max_read_burst_length = 32 bundle = \
    gmem0_4 port = columnG3
#pragma HLS INTERFACE m_axi offset = slave latency = 32 num_read_outstanding = 32 max_read_burst_length = 2 bundle = \
    gmem0_5 port = offsetG3

#pragma HLS INTERFACE m_axi offset = slave latency = 32 num_write_outstanding = 16 num_read_outstanding = \
    16 max_write_burst_length = 32 max_read_burst_length = 32 bundle = gmem0_6 port = offsetG2Tmp1
#pragma HLS INTERFACE m_axi offset = slave latency = 32 num_write_outstanding = 16 num_read_outstanding = \
    16 max_write_burst_length = 32 max_read_burst_length = 32 bundle = gmem0_7 port = offsetG2Tmp2

#pragma HLS INTERFACE m_axi offset = slave latency = 32 num_read_outstanding = 32 max_read_burst_length = 2 bundle = \
    gmem0_8 port = colorMap512G1
#pragma HLS INTERFACE m_axi offset = slave latency = 32 num_write_outstanding = 16 num_read_outstanding = \
    8 max_write_burst_length = 32 max_read_burst_length = 64 bundle = gmem0_9 port = colorMap32G1
#pragma HLS INTERFACE m_axi offset = slave latency = 32 num_write_outstanding = 32 num_read_outstanding = \
    32 max_write_burst_length = 2 max_read_burst_length = 2 bundle = gmem0_10 port = queueG1

#pragma HLS INTERFACE m_axi offset = slave latency = 32 num_read_outstanding = 32 max_read_burst_length = 2 bundle = \
    gmem0_11 port = colorMap512G2
#pragma HLS INTERFACE m_axi offset = slave latency = 32 num_write_outstanding = 32 num_read_outstanding = \
    32 max_write_burst_length = 2 max_read_burst_length = 2 bundle = gmem0_12 port = colorMap32G2
#pragma HLS INTERFACE m_axi offset = slave latency = 32 num_write_outstanding = 32 num_read_outstanding = \
    32 max_write_burst_length = 2 max_read_burst_length = 2 bundle = gmem0_13 port = queueG2

#pragma HLS INTERFACE m_axi offset = slave latency = 32 num_read_outstanding = 2 max_read_burst_length = 64 bundle = \
    gmem0_14 port = queueG3

#pragma HLS INTERFACE m_axi offset = slave latency = 32 num_write_outstanding = 16 num_read_outstanding = \
    64 max_write_burst_length = 32 max_read_burst_length = 8 bundle = gmem0_15 port = result

#pragma HLS INTERFACE s_axilite port = edgeNum bundle = control
#pragma HLS INTERFACE s_axilite port = vertexNum bundle = control
#pragma HLS INTERFACE s_axilite port = columnG1 bundle = control
#pragma HLS INTERFACE s_axilite port = offsetG1 bundle = control
#pragma HLS INTERFACE s_axilite port = column512G2 bundle = control
#pragma HLS INTERFACE s_axilite port = column32G2 bundle = control
#pragma HLS INTERFACE s_axilite port = offsetG2 bundle = control
#pragma HLS INTERFACE s_axilite port = columnG3 bundle = control
#pragma HLS INTERFACE s_axilite port = offsetG3 bundle = control
#pragma HLS INTERFACE s_axilite port = offsetG2Tmp1 bundle = control
#pragma HLS INTERFACE s_axilite port = offsetG2Tmp2 bundle = control
#pragma HLS INTERFACE s_axilite port = colorMap512G1 bundle = control
#pragma HLS INTERFACE s_axilite port = colorMap32G1 bundle = control
#pragma HLS INTERFACE s_axilite port = queueG1 bundle = control
#pragma HLS INTERFACE s_axilite port = colorMap512G2 bundle = control
#pragma HLS INTERFACE s_axilite port = colorMap32G2 bundle = control
#pragma HLS INTERFACE s_axilite port = queueG2 bundle = control
#pragma HLS INTERFACE s_axilite port = queueG3 bundle = control
#pragma HLS INTERFACE s_axilite port = result bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

    const int cacheDepthBin = 14; // cache line depth in Binary
    const int dataOneLineBin = 4; // data numbers in Binary of one buffType
    const int usURAM = 1;         // 0 represents use LUTRAM, 1 represents use URAM, 2 represents use BRAM

    xf::graph::stronglyConnectedComponents<V, E, cacheDepthBin, dataOneLineBin, usURAM, MAXDEGREE>(
        edgeNum, vertexNum, offsetG1, columnG1, offsetG2, column512G2, column32G2, offsetG3, columnG3, offsetG2Tmp1,
        offsetG2Tmp2, colorMap512G1, colorMap32G1, queueG1, colorMap512G2, colorMap32G2, queueG2, queueG3, result);
}
