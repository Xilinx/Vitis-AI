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

#define NRC 16
#define NCU 1

//#define NRC 1024
//#define NCU 16

extern "C" void kernel_gtsv_0(int n, double* matDiagLow, double* matDiag, double* matDiagUp, double* rhs) {
#pragma HLS INTERFACE m_axi offset = slave bundle = gmem0 port = matDiagLow latency = 64 num_read_outstanding = \
    16 num_write_outstanding = 16 max_read_burst_length = 64 max_write_burst_length = 64 depth = 16

#pragma HLS INTERFACE m_axi offset = slave bundle = gmem1 port = matDiag latency = 64 num_read_outstanding = \
    16 num_write_outstanding = 16 max_read_burst_length = 64 max_write_burst_length = 64 depth = 16

#pragma HLS INTERFACE m_axi offset = slave bundle = gmem2 port = matDiagUp latency = 64 num_read_outstanding = \
    16 num_write_outstanding = 16 max_read_burst_length = 64 max_write_burst_length = 64 depth = 16

#pragma HLS INTERFACE m_axi offset = slave bundle = gmem3 port = rhs latency = 64 num_read_outstanding = \
    16 num_write_outstanding = 16 max_read_burst_length = 64 max_write_burst_length = 64 depth = 16

#pragma HLS INTERFACE s_axilite port = n bundle = control
#pragma HLS INTERFACE s_axilite port = matDiagLow bundle = control
#pragma HLS INTERFACE s_axilite port = matDiag bundle = control
#pragma HLS INTERFACE s_axilite port = matDiagUp bundle = control
#pragma HLS INTERFACE s_axilite port = rhs bundle = control

#pragma HLS INTERFACE s_axilite port = return bundle = control

    xf::solver::gtsv<double, NRC, NCU>(n, matDiagLow, matDiag, matDiagUp, rhs);
};
