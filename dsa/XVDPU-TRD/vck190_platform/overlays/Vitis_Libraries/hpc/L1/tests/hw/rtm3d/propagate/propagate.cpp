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
#include "propagate.hpp"

void top_f(STENCEIL_TYPE* s, const t_PairInType* p_img, t_PairInType* p_out, const t_InType* p_v2dt2) {
#pragma HLS DATAFLOW
    hls::stream<t_InType> l_v2dt2[N_INST + 1];
#pragma HLS ARRAY_PARTITION variable = l_v2dt2 complete dim = 1
#pragma HLS stream variable = l_v2dt2 depth = STENCEIL_TYPE::t_FifoDepth
#pragma HLS RESOURCE variable = l_v2dt2 core = RAM_2P_URAM

    hls::stream<t_PairInType> l_img[N_INST + 1];
#pragma HLS ARRAY_PARTITION variable = l_img complete dim = 1
    int l_cube = s[0].getCube();
    xf::blas::mem2stream<t_InType>(l_cube, p_v2dt2, l_v2dt2[0]);
    xf::blas::mem2stream<t_PairInType>(l_cube, p_img, l_img[0]);
    for (int i = 0; i < N_INST; i++)
#pragma HLS UNROLL
        s[i].propagate(l_v2dt2[i], l_v2dt2[i + 1], l_img[i], l_img[i + 1]);
    xf::hpc::dataConsumer(l_cube, l_v2dt2[N_INST]);
    xf::blas::stream2mem<t_PairInType>(l_cube, l_img[N_INST], p_out);
}

extern "C" void top(const unsigned int p_z,
                    const unsigned int p_y,
                    const unsigned int p_x,
                    const DATATYPE p_coefz[ORDER + 1],
                    const DATATYPE p_coefy[ORDER + 1],
                    const DATATYPE p_coefx[ORDER + 1],
                    const t_InType p_v2dt2[M_z * M_y * M_x / nPE],
                    t_PairInType p_in[M_z * M_y * M_x / nPE],
                    t_PairInType p_out[M_z * M_y * M_x / nPE]) {
    STENCEIL_TYPE l_s[N_INST];
#pragma HLS ARRAY_PARTITION variable = l_s complete dim = 1
    for (int i = 0; i < N_INST; i++) {
        l_s[i].setDim(p_z, p_y, p_x);
        l_s[i].setCoef(p_coefz, p_coefy, p_coefx);
    }
    top_f(l_s, p_in, p_out, p_v2dt2);
}
