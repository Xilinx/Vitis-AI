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
/**
 * @file dut.cpp
 *
 * @brief This file contains top function of test case.
 */

#include "kernel_louvain.hpp"
#include "louvain_coloring.hpp"

extern "C" void kernel_louvain(int64_t* config0,
                               DWEIGHT* config1,
                               ap_uint<CSRWIDTHS>* offsets,
                               ap_uint<CSRWIDTHS>* indices,
                               ap_uint<CSRWIDTHS>* weights,
                               ap_uint<COLORWIDTHS>* colorAxi,
                               ap_uint<COLORWIDTHS>* colorInx,
                               ap_uint<DWIDTHS>* cidPrev,
                               ap_uint<DWIDTHS>* cidSizePrev,
                               ap_uint<DWIDTHS>* totPrev,
                               ap_uint<DWIDTHS>* cidCurr,
                               ap_uint<DWIDTHS>* cidSizeCurr,
                               ap_uint<DWIDTHS>* totCurr,
                               ap_uint<DWIDTHS>* cidSizeUpdate,
                               ap_uint<DWIDTHS>* totUpdate,
                               ap_uint<DWIDTHS>* cWeight,
                               ap_uint<CSRWIDTHS>* offsetsDup,
                               ap_uint<CSRWIDTHS>* indicesDup,
                               ap_uint<8>* flag,
                               ap_uint<8>* flagUpdate) {
    DWEIGHT constant_recip = 0;
#pragma HLS INTERFACE m_axi offset = slave bundle = gmem0 port = config0 latency = 32 num_read_outstanding = \
    64 max_read_burst_length = 64 num_write_outstanding = 64 max_write_burst_length = 32 depth = 4

#pragma HLS INTERFACE m_axi offset = slave bundle = gmem0 port = config1 latency = 32 num_read_outstanding = \
    64 max_read_burst_length = 64 num_write_outstanding = 64 max_write_burst_length = 32 depth = 4

#pragma HLS INTERFACE m_axi offset = slave bundle = gmem0 port = offsets latency = 32 num_read_outstanding = \
    64 max_read_burst_length = 64 num_write_outstanding = 64 max_write_burst_length = 32 depth = depthVertex

#pragma HLS INTERFACE m_axi offset = slave bundle = gmem1 port = indices latency = 32 num_read_outstanding = \
    64 max_read_burst_length = 64 num_write_outstanding = 64 max_write_burst_length = 32 depth = depthEdge

#pragma HLS INTERFACE m_axi offset = slave bundle = gmem2 port = weights latency = 32 num_read_outstanding = \
    64 max_read_burst_length = 64 num_write_outstanding = 64 max_write_burst_length = 32 depth = depthEdge

#pragma HLS INTERFACE m_axi offset = slave bundle = gmem3 port = colorAxi latency = 32 num_read_outstanding = \
    64 max_read_burst_length = 32 num_write_outstanding = 64 max_write_burst_length = 32 depth = depthVertex

#pragma HLS INTERFACE m_axi offset = slave bundle = gmem4 port = colorInx latency = 32 num_read_outstanding = \
    64 max_read_burst_length = 32 num_write_outstanding = 64 max_write_burst_length = 32 depth = depthVertex

#pragma HLS INTERFACE m_axi offset = slave bundle = gmem5 port = cidPrev latency = 32 num_read_outstanding = \
    64 max_read_burst_length = 32 num_write_outstanding = 64 max_write_burst_length = 32 depth = depthVertex

#pragma HLS INTERFACE m_axi offset = slave bundle = gmem6 port = cidSizePrev latency = 32 num_read_outstanding = \
    64 max_read_burst_length = 32 num_write_outstanding = 64 max_write_burst_length = 32 depth = depthVertex

#pragma HLS INTERFACE m_axi offset = slave bundle = gmem7 port = totPrev latency = 32 num_read_outstanding = \
    64 max_read_burst_length = 32 num_write_outstanding = 64 max_write_burst_length = 32 depth = depthVertex

#pragma HLS INTERFACE m_axi offset = slave bundle = gmem8 port = cidCurr latency = 32 num_read_outstanding = \
    64 max_read_burst_length = 32 num_write_outstanding = 64 max_write_burst_length = 32 depth = depthVertex

#pragma HLS INTERFACE m_axi offset = slave bundle = gmem9 port = cidSizeCurr latency = 32 num_read_outstanding = \
    64 max_read_burst_length = 32 num_write_outstanding = 64 max_write_burst_length = 32 depth = depthVertex

#pragma HLS INTERFACE m_axi offset = slave bundle = gmem10 port = totCurr latency = 32 num_read_outstanding = \
    64 max_read_burst_length = 32 num_write_outstanding = 64 max_write_burst_length = 32 depth = depthVertex

#pragma HLS INTERFACE m_axi offset = slave bundle = gmem11 port = cidSizeUpdate latency = 32 num_read_outstanding = \
    64 max_read_burst_length = 32 num_write_outstanding = 64 max_write_burst_length = 32 depth = depthVertex

#pragma HLS INTERFACE m_axi offset = slave bundle = gmem12 port = totUpdate latency = 32 num_read_outstanding = \
    64 max_read_burst_length = 32 num_write_outstanding = 64 max_write_burst_length = 32 depth = depthVertex

#pragma HLS INTERFACE m_axi offset = slave bundle = gmem13 port = cWeight latency = 32 num_read_outstanding = \
    64 max_read_burst_length = 32 num_write_outstanding = 64 max_write_burst_length = 32 depth = depthVertex

#pragma HLS INTERFACE m_axi offset = slave bundle = gmem14 port = offsetsDup latency = 32 num_read_outstanding = \
    64 max_read_burst_length = 64 num_write_outstanding = 64 max_write_burst_length = 32 depth = depthVertex

#pragma HLS INTERFACE m_axi offset = slave bundle = gmem15 port = indicesDup latency = 32 num_read_outstanding = \
    64 max_read_burst_length = 64 num_write_outstanding = 64 max_write_burst_length = 32 depth = depthEdge

#pragma HLS INTERFACE m_axi offset = slave bundle = gmem16 port = flag latency = 32 num_read_outstanding = \
    32 max_read_burst_length = 2 num_write_outstanding = 32 max_write_burst_length = 2 depth = depthVertex

#pragma HLS INTERFACE m_axi offset = slave bundle = gmem17 port = flagUpdate latency = 32 num_read_outstanding = \
    32 max_read_burst_length = 2 num_write_outstanding = 32 max_write_burst_length = 2 depth = depthVertex

#pragma HLS INTERFACE s_axilite port = config0 bundle = control
#pragma HLS INTERFACE s_axilite port = config1 bundle = control
#pragma HLS INTERFACE s_axilite port = offsets bundle = control
#pragma HLS INTERFACE s_axilite port = indices bundle = control
#pragma HLS INTERFACE s_axilite port = weights bundle = control
#pragma HLS INTERFACE s_axilite port = colorAxi bundle = control
#pragma HLS INTERFACE s_axilite port = colorInx bundle = control
#pragma HLS INTERFACE s_axilite port = cidPrev bundle = control
#pragma HLS INTERFACE s_axilite port = cidSizePrev bundle = control
#pragma HLS INTERFACE s_axilite port = totPrev bundle = control
#pragma HLS INTERFACE s_axilite port = cidCurr bundle = control
#pragma HLS INTERFACE s_axilite port = cidSizeCurr bundle = control
#pragma HLS INTERFACE s_axilite port = totCurr bundle = control
#pragma HLS INTERFACE s_axilite port = cidSizeUpdate bundle = control
#pragma HLS INTERFACE s_axilite port = totUpdate bundle = control
#pragma HLS INTERFACE s_axilite port = cWeight bundle = control
#pragma HLS INTERFACE s_axilite port = offsetsDup bundle = control
#pragma HLS INTERFACE s_axilite port = indicesDup bundle = control
#pragma HLS INTERFACE s_axilite port = flag bundle = control
#pragma HLS INTERFACE s_axilite port = flagUpdate bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

    xf::graph::kernelLouvainTop(config0, config1, offsets, indices, weights, colorAxi, colorInx, cidPrev, cidSizePrev,
                                totPrev, cidCurr, cidSizeCurr, totCurr, cidSizeUpdate, totUpdate, cWeight, offsetsDup,
                                indicesDup, flag, flagUpdate);
}
