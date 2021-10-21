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

#include "triangle_count_kernel.hpp"

#ifndef __SYNTHESIS__
#include <iostream>
#endif

extern "C" void TC_kernel(int vertexNum,
                          int edgeNum,
                          uint512 offsetsG1[V],
                          uint512 rowsG1[E],
                          uint512 offsetsG2[V],
                          uint512 rowsG2[E],
                          uint512 offsetsG3[V * 2],
                          uint512 rowsG3[E],
                          uint64_t TC[N]) {
#pragma HLS INTERFACE m_axi offset = slave latency = 64 num_write_outstanding = 1 num_read_outstanding = \
    16 max_write_burst_length = 2 max_read_burst_length = 32 bundle = gmem0_0 port = offsetsG1
#pragma HLS INTERFACE m_axi offset = slave latency = 64 num_write_outstanding = 1 num_read_outstanding = \
    16 max_write_burst_length = 2 max_read_burst_length = 32 bundle = gmem0_1 port = rowsG1
#pragma HLS INTERFACE m_axi offset = slave latency = 64 num_write_outstanding = 1 num_read_outstanding = \
    16 max_write_burst_length = 2 max_read_burst_length = 32 bundle = gmem0_2 port = offsetsG2
#pragma HLS INTERFACE m_axi offset = slave latency = 64 num_write_outstanding = 1 num_read_outstanding = \
    16 max_write_burst_length = 2 max_read_burst_length = 32 bundle = gmem0_3 port = rowsG2
#pragma HLS INTERFACE m_axi offset = slave latency = 64 num_write_outstanding = 16 num_read_outstanding = \
    16 max_write_burst_length = 32 max_read_burst_length = 32 bundle = gmem0_4 port = offsetsG3
#pragma HLS INTERFACE m_axi offset = slave latency = 35 num_write_outstanding = 1 num_read_outstanding = \
    16 max_write_burst_length = 2 max_read_burst_length = 32 bundle = gmem0_5 port = rowsG3
#pragma HLS INTERFACE m_axi offset = slave latency = 35 num_write_outstanding = 1 num_read_outstanding = \
    1 max_write_burst_length = 2 max_read_burst_length = 2 bundle = gmem0_6 port = TC

#pragma HLS INTERFACE s_axilite port = vertexNum bundle = control
#pragma HLS INTERFACE s_axilite port = edgeNum bundle = control
#pragma HLS INTERFACE s_axilite port = offsetsG1 bundle = control
#pragma HLS INTERFACE s_axilite port = rowsG1 bundle = control
#pragma HLS INTERFACE s_axilite port = offsetsG2 bundle = control
#pragma HLS INTERFACE s_axilite port = rowsG2 bundle = control
#pragma HLS INTERFACE s_axilite port = offsetsG3 bundle = control
#pragma HLS INTERFACE s_axilite port = rowsG3 bundle = control
#pragma HLS INTERFACE s_axilite port = TC bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

#ifndef __SYNTHESIS__
    std::cout << "vertexNum=" << vertexNum << std::endl;
    std::cout << "edgeNum=" << edgeNum << std::endl;
#endif
    xf::graph::preProcessData(vertexNum, offsetsG1, offsetsG3);
    xf::graph::triangleCount<LEN, ML>(vertexNum, edgeNum, offsetsG1, rowsG1, offsetsG2, rowsG2, offsetsG3, rowsG3, TC);
#ifndef __SYNTHESIS__
    std::cout << "triangle count = " << TC[0] << std::endl;
#endif
}
