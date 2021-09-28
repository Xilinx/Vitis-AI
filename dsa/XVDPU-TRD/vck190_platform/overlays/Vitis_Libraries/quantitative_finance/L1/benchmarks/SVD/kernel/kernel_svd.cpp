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
#include "kernel_svd.hpp"
extern "C" void kernel_svd_0(
    double dataA[NA * NA], double sigma[NA], double dataU[NA * NA], double dataV[NA * NA], int diagSize1) {
#pragma HLS INTERFACE m_axi port = dataA bundle = gmem0 offset = slave num_read_outstanding = \
    16 max_read_burst_length = 32
#pragma HLS INTERFACE m_axi port = sigma bundle = gmem1 offset = slave num_read_outstanding = \
    16 max_read_burst_length = 32
#pragma HLS INTERFACE m_axi port = dataU bundle = gmem2 offset = slave num_read_outstanding = \
    16 max_read_burst_length = 32
#pragma HLS INTERFACE m_axi port = dataV bundle = gmem3 offset = slave num_read_outstanding = \
    16 max_read_burst_length = 32

#pragma HLS INTERFACE s_axilite port = diagSize1 bundle = control
#pragma HLS INTERFACE s_axilite port = dataA bundle = control
#pragma HLS INTERFACE s_axilite port = sigma bundle = control
#pragma HLS INTERFACE s_axilite port = dataU bundle = control
#pragma HLS INTERFACE s_axilite port = dataV bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

#pragma HLS data_pack variable = dataA
#pragma HLS data_pack variable = sigma
#pragma HLS data_pack variable = dataU
#pragma HLS data_pack variable = dataV

    double dataA_new[NA][NA];
    double sigma_new[NA][NA];
    double dataU_new[NA][NA];
    double dataV_new[NA][NA];
    int k = 0;
    for (int i = 0; i < NA; ++i) {
        for (int j = 0; j < NA; ++j) {
#pragma HLS pipeline
            dataA_new[i][j] = dataA[k];
            k++;
        }
    }
    for (int i = 0; i < 1; ++i) {
        xf::fintech::svd<double, NA>(dataA_new, sigma_new, dataU_new, dataV_new);
    }
    k = 0;
    for (int i = 0; i < NA; ++i) {
        for (int j = 0; j < NA; ++j) {
#pragma HLS pipeline
            dataU[k] = dataU_new[i][j];
            dataV[k] = dataV_new[i][j];
            k++;
            if (j == 0) {
                sigma[i] = sigma_new[i][i];
            }
        }
    }
}
