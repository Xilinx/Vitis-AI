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

/**
 *  @brief FPGA FD accelerator kernel
 *
 *  $DateTime: 2018/02/05 02:36:41 $
 */

#include <assert.h>
#include "ap_fixed.h"
#include "xf_fintech/fd_bs_local_volatility_solver.hpp"

extern "C" {

void fd_bs_lv_kernel(ap_uint<512>* xGrid,
                     ap_uint<512>* tGrid,
                     ap_uint<512>* sigma,
                     ap_uint<512>* rate,
                     ap_uint<512>* initialCondition,
                     float theta,
                     FD_DATA_TYPE boundaryLower,
                     FD_DATA_TYPE boundaryUpper,
                     unsigned int tSteps,
                     ap_uint<512>* solution) {
#pragma HLS INTERFACE m_axi port = xGrid offset = slave bundle = gmem0
#pragma HLS INTERFACE m_axi port = tGrid offset = slave bundle = gmem0
#pragma HLS INTERFACE m_axi port = sigma offset = slave bundle = gmem0
#pragma HLS INTERFACE m_axi port = rate offset = slave bundle = gmem0
#pragma HLS INTERFACE m_axi port = initialCondition offset = slave bundle = gmem0
#pragma HLS INTERFACE m_axi port = solution offset = slave bundle = gmem0

#pragma HLS INTERFACE s_axilite port = xGrid bundle = control
#pragma HLS INTERFACE s_axilite port = tGrid bundle = control
#pragma HLS INTERFACE s_axilite port = sigma bundle = control
#pragma HLS INTERFACE s_axilite port = rate bundle = control
#pragma HLS INTERFACE s_axilite port = initialCondition bundle = control
#pragma HLS INTERFACE s_axilite port = theta bundle = control
#pragma HLS INTERFACE s_axilite port = boundaryLower bundle = control
#pragma HLS INTERFACE s_axilite port = boundaryUpper bundle = control
#pragma HLS INTERFACE s_axilite port = tSteps bundle = control
#pragma HLS INTERFACE s_axilite port = solution bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

    xf::fintech::FdBsLvSolver<FD_DATA_TYPE, FD_DATA_EQ_TYPE, FD_N_SIZE, FD_M_SIZE>(
        xGrid, tGrid, sigma, rate, initialCondition, theta, boundaryLower, boundaryUpper, tSteps, solution);
}

} // extern C
