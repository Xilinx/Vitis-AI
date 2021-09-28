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
#define MA 16 // Matrix Row size
#define NA 16 // Matrix Col size
#define NCU 4 // Number of compute units

extern "C" void kernel_geqrf_0(double* dataA, double* tau) {
// extern "C" void kernel_geqrf_0(double dataA[MA*NA], double tau[NA]) {
#pragma HLS INTERFACE m_axi port = dataA bundle = gmem0 offset = slave num_read_outstanding = \
    16 max_read_burst_length = 32
#pragma HLS INTERFACE m_axi port = tau bundle = gmem1 offset = slave num_read_outstanding = 16 max_read_burst_length = \
    32

#pragma HLS INTERFACE s_axilite port = dataA bundle = control
#pragma HLS INTERFACE s_axilite port = tau bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

    //#pragma HLS data_pack variable=dataA
    //#pragma HLS data_pack variable=tau

    // matrix initialization
    double dataA_2D[MA][NA];

    // Matrix transform from 1D to 2D
    int k = 0;
    for (int i = 0; i < MA; ++i) {
        for (int j = 0; j < NA; ++j) {
#pragma HLS pipeline
            dataA_2D[i][j] = dataA[k];
            k++;
        }
    }

    // Calling for QR
    xf::solver::geqrf<double, MA, NA, NCU>(MA, NA, dataA_2D, NA, tau);

    // Matrix transform from 2D to 1D
    k = 0;
    for (int i = 0; i < MA; ++i) {
        for (int j = 0; j < NA; ++j) {
#pragma HLS pipeline
            dataA[k] = dataA_2D[i][j];
            k++;
        }
    }
}
