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
#define MA 4  // Symmetric Matrix Row size
#define NA MA // Symmetric Matrix Col size

extern "C" void kernel_syevj_0(double dataA[NA * NA], double sigma[NA], double dataU[NA * NA], int matrixSize) {
#pragma HLS INTERFACE m_axi port = dataA bundle = gmem0 offset = slave num_read_outstanding = \
    16 max_read_burst_length = 32
#pragma HLS INTERFACE m_axi port = sigma bundle = gmem1 offset = slave num_read_outstanding = \
    16 max_read_burst_length = 32
#pragma HLS INTERFACE m_axi port = dataU bundle = gmem2 offset = slave num_read_outstanding = \
    16 max_read_burst_length = 32

#pragma HLS INTERFACE s_axilite port = matrixSize bundle = control
#pragma HLS INTERFACE s_axilite port = dataA bundle = control
#pragma HLS INTERFACE s_axilite port = sigma bundle = control
#pragma HLS INTERFACE s_axilite port = dataU bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

#pragma HLS dataflow

    // p Calling for svd core function
    int info;
    xf::solver::syevj<double, NA, 2>(matrixSize, dataA, matrixSize, sigma, dataU, matrixSize, info);
}
