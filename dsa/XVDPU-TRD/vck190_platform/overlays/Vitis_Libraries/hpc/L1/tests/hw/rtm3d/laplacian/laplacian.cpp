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
#include "laplacian.hpp"

void top_f(STENCEIL_TYPE* s, const t_InType* p_img, t_InType* p_out) {
#pragma HLS DATAFLOW
    hls::stream<t_InType> l_str[NT + 1];
#pragma HLS ARRAY_PARTITION variable = l_str complete dim = 1
    xf::blas::mem2stream<t_InType>(s[0].getCube(), p_img, l_str[0]);
    for (int i = 0; i < NT; i++)
#pragma HLS UNROLL
        s[i].laplacian(l_str[i], l_str[i + 1]);
    xf::blas::stream2mem<t_InType>(s[0].getCube(), l_str[NT], p_out);
}

extern "C" void top(const unsigned int p_z,
                    const unsigned int p_y,
                    const unsigned int p_x,
                    const DATATYPE p_coefz[ORDER + 1],
                    const DATATYPE p_coefy[ORDER + 1],
                    const DATATYPE p_coefx[ORDER + 1],
                    const t_InType p_img[M_z * M_y * M_x / nPE],
                    t_InType p_out[M_z * M_y * M_x / nPE]) {
    STENCEIL_TYPE l_s[NT];
#pragma HLS ARRAY_PARTITION variable = l_s complete dim = 1
    for (int i = 0; i < NT; i++) {
        l_s[i].setDim(p_z, p_y, p_x);
        l_s[i].setCoef(p_coefz, p_coefy, p_coefx);
    }
    top_f(l_s, p_img, p_out);
}
