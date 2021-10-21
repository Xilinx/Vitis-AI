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

extern "C" {

void kernel_getrf_0(double* A, int* P) {
// clang-format off
#pragma HLS INTERFACE m_axi offset = slave bundle = gmem0 port = A latency = 64 \
  num_read_outstanding = 16 num_write_outstanding = 16 \
  max_read_burst_length = 64 max_write_burst_length = 64 depth=16*16

#pragma HLS INTERFACE m_axi offset = slave bundle = gmem0 port = P latency = 64 \
  num_read_outstanding = 16 num_write_outstanding = 16 \
  max_read_burst_length = 64 max_write_burst_length = 64 depth=16

// clang-format on
#pragma HLS INTERFACE s_axilite port = A bundle = control
#pragma HLS INTERFACE s_axilite port = P bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

    int info;
    xf::solver::getrf<double, NRC, NCU>(NRC, A, NRC, P, info);
};
};
