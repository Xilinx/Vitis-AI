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

#include "twoHop_kernel.hpp"

#ifndef __SYNTHESIS__
#include <iostream>
#endif

extern "C" void twoHop_kernel(unsigned numPairs,
                              ap_uint<64>* pair,

                              unsigned* offsetOneHop,
                              unsigned* indexOneHop,
                              unsigned* offsetTwoHop,
                              unsigned* indexTwoHop,

                              unsigned* cnt_res) {
// clang-format off
#pragma HLS INTERFACE m_axi offset = slave latency = 32 num_write_outstanding = 1 num_read_outstanding = \
    16 max_write_burst_length = 2 max_read_burst_length = 256 bundle = gmem0 port = pair

#pragma HLS INTERFACE m_axi offset = slave latency = 32 num_write_outstanding = 16 num_read_outstanding = \
    1 max_write_burst_length = 2 max_read_burst_length = 2 bundle = gmem1 port = cnt_res

#pragma HLS INTERFACE m_axi offset = slave latency = 64 num_write_outstanding = 1 num_read_outstanding = \
    64 max_write_burst_length = 2 max_read_burst_length = 2 bundle = gmem2 port = offsetOneHop

#pragma HLS INTERFACE m_axi offset = slave latency = 64 num_write_outstanding = 1 num_read_outstanding = \
    64 max_write_burst_length = 2 max_read_burst_length = 256 bundle = gmem3 port = indexOneHop

#pragma HLS INTERFACE m_axi offset = slave latency = 64 num_write_outstanding = 1 num_read_outstanding = \
    64 max_write_burst_length = 2 max_read_burst_length = 2 bundle = gmem4 port = offsetTwoHop

#pragma HLS INTERFACE m_axi offset = slave latency = 64 num_write_outstanding = 1 num_read_outstanding = \
    64 max_write_burst_length = 2 max_read_burst_length = 256 bundle = gmem5 port = indexTwoHop

#pragma HLS INTERFACE s_axilite port = numPairs bundle = control
#pragma HLS INTERFACE s_axilite port = pair bundle = control
#pragma HLS INTERFACE s_axilite port = offsetOneHop bundle = control
#pragma HLS INTERFACE s_axilite port = indexOneHop bundle = control
#pragma HLS INTERFACE s_axilite port = offsetTwoHop bundle = control
#pragma HLS INTERFACE s_axilite port = indexTwoHop bundle = control
#pragma HLS INTERFACE s_axilite port = cnt_res bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control
// clang-format on

#ifndef __SYNTHESIS__
    std::cout << "kernel call success" << std::endl;
#endif

    xf::graph::twoHop(numPairs, pair, offsetOneHop, indexOneHop, offsetTwoHop, indexTwoHop, cnt_res);

#ifndef __SYNTHESIS__
    std::cout << "kernel call finish" << std::endl;
#endif
}
