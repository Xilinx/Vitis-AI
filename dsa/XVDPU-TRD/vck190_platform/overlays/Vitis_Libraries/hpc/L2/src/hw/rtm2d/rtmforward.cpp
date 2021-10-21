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
 * @file rtmforward.cpp
 * @brief It defines the forward kernel function
 */
#include "rtmforward.hpp"
extern "C" void rtmforward(const unsigned int p_z,
                           const unsigned int p_x,
                           const unsigned int p_t,
                           const unsigned int p_srcz,
                           const unsigned int p_srcx,
                           const RTM_dataType* p_src,
                           const RTM_dataType* p_coefz,
                           const RTM_dataType* p_coefx,
                           const RTM_dataType* p_taperz,
                           const RTM_dataType* p_taperx,
                           const RTM_interface* p_v2dt2,
                           RTM_interface* p_p0,
                           RTM_interface* p_p1,
                           RTM_upbType* p_upb) {
#pragma HLS INTERFACE s_axilite port = p_x bundle = control
#pragma HLS INTERFACE s_axilite port = p_z bundle = control
#pragma HLS INTERFACE s_axilite port = p_t bundle = control
#pragma HLS INTERFACE s_axilite port = p_srcx bundle = control
#pragma HLS INTERFACE s_axilite port = p_srcz bundle = control
#pragma HLS INTERFACE s_axilite port = p_src bundle = control
#pragma HLS INTERFACE s_axilite port = p_coefx bundle = control
#pragma HLS INTERFACE s_axilite port = p_coefz bundle = control
#pragma HLS INTERFACE s_axilite port = p_taperx bundle = control
#pragma HLS INTERFACE s_axilite port = p_taperz bundle = control
#pragma HLS INTERFACE s_axilite port = p_v2dt2 bundle = control
#pragma HLS INTERFACE s_axilite port = p_p0 bundle = control
#pragma HLS INTERFACE s_axilite port = p_p1 bundle = control
#pragma HLS INTERFACE s_axilite port = p_upb bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

#pragma HLS INTERFACE m_axi port = p_coefx offset = slave bundle = gmemDDR
#pragma HLS INTERFACE m_axi port = p_coefz offset = slave bundle = gmemDDR
#pragma HLS INTERFACE m_axi port = p_taperx offset = slave bundle = gmemDDR
#pragma HLS INTERFACE m_axi port = p_taperz offset = slave bundle = gmemDDR
#pragma HLS INTERFACE m_axi port = p_src offset = slave bundle = gmemDDR
#pragma HLS INTERFACE m_axi port = p_upb offset = slave bundle = gmemDDR

#pragma HLS INTERFACE m_axi port = p_p0 offset = slave bundle = gmem_p0
#pragma HLS INTERFACE m_axi port = p_p1 offset = slave bundle = gmem_p1
#pragma HLS INTERFACE m_axi port = p_v2dt2 offset = slave bundle = gmem_v2dt2

#ifndef __SYNTHESIS__
    assert(p_x * p_z % RTM_parEntries == 0);
#endif

    RTM_TYPE l_s[RTM_numFSMs];
#pragma HLS ARRAY_PARTITION variable = l_s complete dim = 1
    RTM_dataType l_src[RTM_numFSMs];
#pragma HLS ARRAY_PARTITION variable = l_src complete dim = 1
    for (int i = 0; i < RTM_numFSMs; i++) {
        l_s[i].setDim(p_z, p_x);
        l_s[i].setCoef(p_coefz, p_coefx);
        l_s[i].setTaper(p_taperz, p_taperx);
        l_s[i].setSrc(p_srcz, p_srcx);
    }
    for (int t = 0; t < p_t / RTM_numFSMs; t++) {
        for (int i = 0; i < RTM_numFSMs; i++) l_src[i] = p_src[t * RTM_numFSMs + i];

        forward(l_s, t, l_src, p_v2dt2, p_upb, p_p0, p_p1);
    }
}
