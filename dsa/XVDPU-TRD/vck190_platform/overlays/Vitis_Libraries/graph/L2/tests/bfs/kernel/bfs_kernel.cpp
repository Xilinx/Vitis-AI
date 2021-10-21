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

#include "bfs_kernel.hpp"

extern "C" void bfs_kernel(const int srcID,
                           const int vertexNum,

                           ap_uint<512>* column,
                           ap_uint<512>* offset,

                           ap_uint<512>* queue512,
                           ap_uint<32>* queue,
                           ap_uint<512>* color512,

                           ap_uint<32>* result_dt,
                           ap_uint<32>* result_ft,
                           ap_uint<32>* result_pt,
                           ap_uint<32>* result_lv) {
#pragma HLS INTERFACE m_axi offset = slave latency = 32 num_read_outstanding = 32 max_read_burst_length = 8 bundle = \
    gmem0_0 port = column
#pragma HLS INTERFACE m_axi offset = slave latency = 32 num_read_outstanding = 64 max_read_burst_length = 8 bundle = \
    gmem0_1 port = offset

#pragma HLS INTERFACE m_axi offset = slave latency = 32 num_write_outstanding = 64 num_read_outstanding = \
    2 max_write_burst_length = 2 max_read_burst_length = 2 bundle = gmem0_2 port = queue512
#pragma HLS INTERFACE m_axi offset = slave latency = 32 num_write_outstanding = 64 num_read_outstanding = \
    2 max_write_burst_length = 2 max_read_burst_length = 2 bundle = gmem0_2 port = queue
#pragma HLS INTERFACE m_axi offset = slave latency = 32 num_write_outstanding = 64 num_read_outstanding = \
    64 max_write_burst_length = 32 max_read_burst_length = 2 bundle = gmem1_0 port = color512

#pragma HLS INTERFACE m_axi offset = slave latency = 32 num_write_outstanding = 64 num_read_outstanding = \
    64 max_write_burst_length = 32 max_read_burst_length = 2 bundle = gmem1_0 port = result_dt
#pragma HLS INTERFACE m_axi offset = slave latency = 32 num_write_outstanding = 64 max_write_burst_length = 2 bundle = \
    gmem1_1 port = result_ft
#pragma HLS INTERFACE m_axi offset = slave latency = 32 num_write_outstanding = 64 max_write_burst_length = 2 bundle = \
    gmem1_2 port = result_pt
#pragma HLS INTERFACE m_axi offset = slave latency = 32 num_write_outstanding = 64 max_write_burst_length = 2 bundle = \
    gmem1_3 port = result_lv

#pragma HLS INTERFACE s_axilite port = srcID bundle = control
#pragma HLS INTERFACE s_axilite port = vertexNum bundle = control
#pragma HLS INTERFACE s_axilite port = column bundle = control
#pragma HLS INTERFACE s_axilite port = offset bundle = control
#pragma HLS INTERFACE s_axilite port = queue512 bundle = control
#pragma HLS INTERFACE s_axilite port = queue bundle = control
#pragma HLS INTERFACE s_axilite port = color512 bundle = control
#pragma HLS INTERFACE s_axilite port = result_dt bundle = control
#pragma HLS INTERFACE s_axilite port = result_ft bundle = control
#pragma HLS INTERFACE s_axilite port = result_pt bundle = control
#pragma HLS INTERFACE s_axilite port = result_lv bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

    xf::graph::bfs<MAXDEGREE>(srcID, vertexNum, offset, column, queue512, queue, color512, result_dt, result_ft,
                              result_pt, result_lv);
}
