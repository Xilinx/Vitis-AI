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
 * @file rtmbackward.cpp
 * @brief It defines the backward kernel function
 */

#include "rtmbackward.hpp"
extern "C" void rtmbackward(const unsigned int p_z,
                            const unsigned int p_x,
                            const unsigned int p_t,
                            const unsigned int p_recz,
                            const RTM_dataType* p_rec,
                            const RTM_dataType* p_coefz,
                            const RTM_dataType* p_coefx,
                            const RTM_dataType* p_taperz,
                            const RTM_dataType* p_taperx,
                            const RTM_interface* p_v2dt2,
                            RTM_interface* p_p0,
                            RTM_interface* p_p1,
                            RTM_interface* p_r0,
                            RTM_interface* p_r1,
                            RTM_interface* p_i0,
                            RTM_interface* p_i1,
                            RTM_upbType* p_upb) {
#pragma HLS INTERFACE s_axilite port = p_x bundle = control
#pragma HLS INTERFACE s_axilite port = p_z bundle = control
#pragma HLS INTERFACE s_axilite port = p_t bundle = control
#pragma HLS INTERFACE s_axilite port = p_recz bundle = control
#pragma HLS INTERFACE s_axilite port = p_rec bundle = control
#pragma HLS INTERFACE s_axilite port = p_coefx bundle = control
#pragma HLS INTERFACE s_axilite port = p_coefz bundle = control
#pragma HLS INTERFACE s_axilite port = p_taperx bundle = control
#pragma HLS INTERFACE s_axilite port = p_taperz bundle = control
#pragma HLS INTERFACE s_axilite port = p_v2dt2 bundle = control
#pragma HLS INTERFACE s_axilite port = p_p0 bundle = control
#pragma HLS INTERFACE s_axilite port = p_p1 bundle = control
#pragma HLS INTERFACE s_axilite port = p_r0 bundle = control
#pragma HLS INTERFACE s_axilite port = p_r1 bundle = control
#pragma HLS INTERFACE s_axilite port = p_i0 bundle = control
#pragma HLS INTERFACE s_axilite port = p_i1 bundle = control
#pragma HLS INTERFACE s_axilite port = p_upb bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

#pragma HLS INTERFACE m_axi port = p_coefx offset = slave bundle = gmemDDR
#pragma HLS INTERFACE m_axi port = p_coefz offset = slave bundle = gmemDDR
#pragma HLS INTERFACE m_axi port = p_taperx offset = slave bundle = gmemDDR
#pragma HLS INTERFACE m_axi port = p_taperz offset = slave bundle = gmemDDR

#pragma HLS INTERFACE m_axi port = p_upb offset = slave bundle = gmemDDR
#pragma HLS INTERFACE m_axi port = p_rec offset = slave bundle = gmem_rec

#pragma HLS INTERFACE m_axi port = p_p0 offset = slave bundle = gmem_p0
#pragma HLS INTERFACE m_axi port = p_p1 offset = slave bundle = gmem_p1
#pragma HLS INTERFACE m_axi port = p_r0 offset = slave bundle = gmem_r0
#pragma HLS INTERFACE m_axi port = p_r1 offset = slave bundle = gmem_r1
#pragma HLS INTERFACE m_axi port = p_i0 offset = slave bundle = gmem_i0
#pragma HLS INTERFACE m_axi port = p_i1 offset = slave bundle = gmem_i1
#pragma HLS INTERFACE m_axi port = p_v2dt2 offset = slave bundle = gmem_v2dt2

    RTM_TYPE l_f[RTM_numBSMs];
#pragma HLS ARRAY_PARTITION variable = l_f complete dim = 1
    RTM_TYPE l_r[RTM_numBSMs];
#pragma HLS ARRAY_PARTITION variable = l_r complete dim = 1

    for (int i = 0; i < RTM_numBSMs; i++) {
        l_r[i].setDim(p_z, p_x);
        l_r[i].setCoef(p_coefz, p_coefx);
        l_r[i].setTaper(p_taperz, p_taperx);
        l_r[i].setReceiver(p_recz);

        l_f[i].setDim(p_z, p_x);
        l_f[i].setCoef(p_coefz, p_coefx);
        l_f[i].setTaper(p_taperz, p_taperx);
        l_f[i].setReceiver(p_recz);
    }
    for (int t = 0; t < p_t / RTM_numBSMs; t++) {
        backward(l_f, l_r, p_t / RTM_numBSMs - 1 - t, p_t, p_v2dt2, p_rec, p_upb, p_p0, p_p1, p_r0, p_r1, p_i0, p_i1);
    }
}
