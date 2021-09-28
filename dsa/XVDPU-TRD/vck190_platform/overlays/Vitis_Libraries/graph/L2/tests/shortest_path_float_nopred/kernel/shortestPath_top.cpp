/*
 * Copyright 2019 Xilinx, Inc.
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

#include "shortestPath_top.hpp"

#ifndef __SYNTHESIS__
#include <iostream>
#endif

extern "C" void shortestPath_top(ap_uint<32>* config,
                                 ap_uint<512>* offset,
                                 ap_uint<512>* column,
                                 ap_uint<512>* weight,

                                 ap_uint<512>* ddrQue512,
                                 ap_uint<32>* ddrQue,

                                 ap_uint<512>* result512,
                                 ap_uint<32>* result,

                                 ap_uint<8>* info) {
    const int depth_E = E;
    const int depth_V = V;
// clang-format off
#pragma HLS INTERFACE m_axi offset = slave latency = 32 num_write_outstanding = 1 num_read_outstanding = \
    32 max_write_burst_length = 2 max_read_burst_length = 8 bundle = gmem0 port = config depth = 4
#pragma HLS INTERFACE m_axi offset = slave latency = 32 num_write_outstanding = 1 num_read_outstanding = \
    32 max_write_burst_length = 2 max_read_burst_length = 8 bundle = gmem0 port = offset depth = depth_V
#pragma HLS INTERFACE m_axi offset = slave latency = 32 num_write_outstanding = 1 num_read_outstanding = \
    32 max_write_burst_length = 2 max_read_burst_length = 32 bundle = gmem1 port = column depth = depth_E
#pragma HLS INTERFACE m_axi offset = slave latency = 32 num_write_outstanding = 1 num_read_outstanding = \
    32 max_write_burst_length = 2 max_read_burst_length = 32 bundle = gmem2 port = weight depth = depth_E
#pragma HLS INTERFACE m_axi offset = slave latency = 32 num_write_outstanding = 32 num_read_outstanding = \
    1 max_write_burst_length = 2 max_read_burst_length = 2 bundle = gmem3 port = ddrQue depth = depth_E*16
#pragma HLS INTERFACE m_axi offset = slave latency = 32 num_write_outstanding = 32 num_read_outstanding = \
    1 max_write_burst_length = 2 max_read_burst_length = 2 bundle = gmem3 port = ddrQue512 depth = depth_E
#pragma HLS INTERFACE m_axi offset = slave latency = 32 num_write_outstanding = 32 num_read_outstanding = \
    32 max_write_burst_length = 64 max_read_burst_length = 2 bundle = gmem4 port = result512 depth = depth_V
#pragma HLS INTERFACE m_axi offset = slave latency = 32 num_write_outstanding = 32 num_read_outstanding = \
    32 max_write_burst_length = 64 max_read_burst_length = 2 bundle = gmem4 port = info depth = 8
#pragma HLS INTERFACE m_axi offset = slave latency = 32 num_write_outstanding = 32 num_read_outstanding = \
    32 max_write_burst_length = 64 max_read_burst_length = 2 bundle = gmem4 port = result depth = depth_V*16

// clang-format on
#pragma HLS INTERFACE s_axilite port = config bundle = control
#pragma HLS INTERFACE s_axilite port = offset bundle = control
#pragma HLS INTERFACE s_axilite port = column bundle = control
#pragma HLS INTERFACE s_axilite port = weight bundle = control
#pragma HLS INTERFACE s_axilite port = ddrQue bundle = control
#pragma HLS INTERFACE s_axilite port = ddrQue512 bundle = control
#pragma HLS INTERFACE s_axilite port = result bundle = control
#pragma HLS INTERFACE s_axilite port = result512 bundle = control
#pragma HLS INTERFACE s_axilite port = info bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

#ifndef __SYNTHESIS__
    std::cout << "kernel call success" << std::endl;
#endif
    xf::graph::singleSourceShortestPath<32, MAXOUTDEGREE>(config, offset, column, weight, ddrQue512, ddrQue, result512,
                                                          result, info);
#ifndef __SYNTHESIS__
    std::cout << "kernel call finish" << std::endl;
#endif
}
