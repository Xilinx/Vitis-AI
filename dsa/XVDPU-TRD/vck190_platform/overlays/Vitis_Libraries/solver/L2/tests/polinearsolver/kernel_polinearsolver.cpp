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

#include "xf_solver_L2.hpp"
#define NCU 2
#define MAXN 4
#define LDA 4
#define LDB 2

extern "C" void kernel_polinearsolver_0(int na, int nb, double* dataA, double* dataB) {
#pragma HLS INTERFACE m_axi port = dataA bundle = gmem0 offset = slave num_read_outstanding = \
    16 max_read_burst_length = 32
#pragma HLS INTERFACE m_axi port = dataB bundle = gmem1 offset = slave num_read_outstanding = \
    16 max_read_burst_length = 32

#pragma HLS INTERFACE s_axilite port = na bundle = control
#pragma HLS INTERFACE s_axilite port = nb bundle = control
#pragma HLS INTERFACE s_axilite port = dataA bundle = control
#pragma HLS INTERFACE s_axilite port = dataB bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

    int info;
    // Calling for cholesky core function
    xf::solver::polinearsolver<double, MAXN, NCU>(na, dataA, nb, dataB, LDA, LDB, info);
}
