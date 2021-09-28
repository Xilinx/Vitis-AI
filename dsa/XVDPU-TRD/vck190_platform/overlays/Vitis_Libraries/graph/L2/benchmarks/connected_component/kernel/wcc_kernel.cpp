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

#include "wcc_kernel.hpp"

extern "C" void wcc_kernel(const int edgeNum,
                           const int vertexNum,

                           ap_uint<512>* columnG1,
                           ap_uint<512>* offsetG1,
                           ap_uint<512>* column512G2,
                           ap_uint<32>* column32G2,
                           ap_uint<512>* offsetG2,

                           ap_uint<512>* offsetG2Tmp1,
                           ap_uint<512>* offsetG2Tmp2,

                           ap_uint<512>* queue512,
                           ap_uint<32>* queue,

                           ap_uint<512>* result512,
                           ap_uint<32>* result32) {
#pragma HLS INTERFACE m_axi offset = slave latency = 32 num_read_outstanding = 32 max_read_burst_length = 8 bundle = \
    gmem0_0 port = columnG1
#pragma HLS INTERFACE m_axi offset = slave latency = 32 num_read_outstanding = 32 max_read_burst_length = 2 bundle = \
    gmem0_1 port = offsetG1
#pragma HLS INTERFACE m_axi offset = slave latency = 32 num_write_outstanding = 32 num_read_outstanding = \
    32 max_write_burst_length = 32 max_read_burst_length = 32 bundle = gmem0_2 port = column512G2
#pragma HLS INTERFACE m_axi offset = slave latency = 32 num_write_outstanding = 32 num_read_outstanding = \
    32 max_write_burst_length = 32 max_read_burst_length = 32 bundle = gmem0_3 port = column32G2
#pragma HLS INTERFACE m_axi offset = slave latency = 32 num_write_outstanding = 32 num_read_outstanding = \
    32 max_write_burst_length = 32 max_read_burst_length = 32 bundle = gmem0_4 port = offsetG2

#pragma HLS INTERFACE m_axi offset = slave latency = 32 num_write_outstanding = 32 num_read_outstanding = \
    32 max_write_burst_length = 32 max_read_burst_length = 32 bundle = gmem0_5 port = offsetG2Tmp1
#pragma HLS INTERFACE m_axi offset = slave latency = 32 num_write_outstanding = 32 num_read_outstanding = \
    32 max_write_burst_length = 32 max_read_burst_length = 32 bundle = gmem0_6 port = offsetG2Tmp2

#pragma HLS INTERFACE m_axi offset = slave latency = 32 num_read_outstanding = 32 max_read_burst_length = 2 bundle = \
    gmem0_7 port = queue512
#pragma HLS INTERFACE m_axi offset = slave latency = 32 num_write_outstanding = 32 max_write_burst_length = 2 bundle = \
    gmem0_8 port = queue

#pragma HLS INTERFACE m_axi offset = slave latency = 32 num_read_outstanding = 32 max_read_burst_length = 2 bundle = \
    gmem0_9 port = result512
#pragma HLS INTERFACE m_axi offset = slave latency = 32 num_write_outstanding = 32 max_write_burst_length = 2 bundle = \
    gmem0_10 port = result32

#pragma HLS INTERFACE s_axilite port = edgeNum bundle = control
#pragma HLS INTERFACE s_axilite port = vertexNum bundle = control
#pragma HLS INTERFACE s_axilite port = columnG1 bundle = control
#pragma HLS INTERFACE s_axilite port = offsetG1 bundle = control
#pragma HLS INTERFACE s_axilite port = column512G2 bundle = control
#pragma HLS INTERFACE s_axilite port = column32G2 bundle = control
#pragma HLS INTERFACE s_axilite port = offsetG2 bundle = control
#pragma HLS INTERFACE s_axilite port = offsetG2Tmp1 bundle = control
#pragma HLS INTERFACE s_axilite port = offsetG2Tmp2 bundle = control
#pragma HLS INTERFACE s_axilite port = queue512 bundle = control
#pragma HLS INTERFACE s_axilite port = queue bundle = control
#pragma HLS INTERFACE s_axilite port = result512 bundle = control
#pragma HLS INTERFACE s_axilite port = result32 bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

    const int cacheDepthBin = 14; // cache line depth in Binary
    const int dataOneLineBin = 4; // data numbers in Binary of one buffType
    const int usURAM = 1;         // 0 represents use LUTRAM, 1 represents use URAM, 2 represents use BRAM

    xf::graph::connectedComponents<V, E, cacheDepthBin, dataOneLineBin, usURAM, MAXDEGREE>(
        edgeNum, vertexNum, offsetG1, columnG1, offsetG2, column512G2, column32G2, offsetG2Tmp1, offsetG2Tmp2, queue512,
        queue, result512, result32);
}
