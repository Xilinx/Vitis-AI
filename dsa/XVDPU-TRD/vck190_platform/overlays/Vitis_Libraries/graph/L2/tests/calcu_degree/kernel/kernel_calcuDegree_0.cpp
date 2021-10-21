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

#include "kernel_calcuDegree.hpp"

#ifdef KERNEL0
extern "C" void kernel_calcuDegree_0(int nrows, int nnz, buffType* degreeCSR, buffType* indiceCSC) {
#pragma HLS INTERFACE m_axi offset = slave bundle = gmem1 port = degreeCSR latency = 125 num_read_outstanding = \
    32 max_read_burst_length = 2 num_write_outstanding = 32 max_write_burst_length = 2
#pragma HLS INTERFACE m_axi offset = slave bundle = gmem3 port = indiceCSC latency = 125 num_read_outstanding = \
    32 max_read_burst_length = 32
#pragma HLS INTERFACE s_axilite port = degreeCSR bundle = control
#pragma HLS INTERFACE s_axilite port = indiceCSC bundle = control
#pragma HLS INTERFACE s_axilite port = nnz bundle = control
#pragma HLS INTERFACE s_axilite port = nrows bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

    const int cacheDepthBin = 15; // cache line depth in Binary
    const int dataOneLineBin = 4; // data numbers in Binary of one buffType
    const int usURAM = 1;         // 0 represents use LUTRAM, 1 represents use URAM, 2 represents use BRAM

    xf::graph::calcuDegree<maxVertex / 2, maxEdge / 2, cacheDepthBin, dataOneLineBin, usURAM>(nrows, nnz, indiceCSC,
                                                                                              degreeCSR);
}
#endif
