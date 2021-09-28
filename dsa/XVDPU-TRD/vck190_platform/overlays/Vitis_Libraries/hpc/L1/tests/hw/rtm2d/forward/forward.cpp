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

#include "forward.hpp"

template <unsigned int t_NumInstance, typename t_DataType, typename t_PairInType, typename t_UpbInType>
void rtmforward(RTM_TYPE s[t_NumInstance],
                const bool p_sel,
                const unsigned int p_t,
                const t_DataType* p_src,
                const IN_TYPE* p_v2dt2,
                t_UpbInType* p_upb,
                t_PairInType* p_p0,
                t_PairInType* p_p1,
                t_PairInType* p_pp0,
                t_PairInType* p_pp1) {
    const unsigned int l_num = s[0].getArea();
    const unsigned int l_upbSize = s[0].getX();

    hls::stream<t_PairInType> l_p[t_NumInstance + 1];
#pragma HLS ARRAY_PARTITION variable = l_p dim = 1 complete
#pragma HLS stream variable = l_p depth = 4
    hls::stream<IN_TYPE> l_v2dt2, l_vt[t_NumInstance + 1];
#pragma HLS ARRAY_PARTITION variable = l_vt dim = 1 complete
#pragma HLS stream depth = RTM_TYPE::t_FifoDepth variable = l_vt
    hls::stream<t_UpbInType> l_upb[t_NumInstance];
#pragma HLS ARRAY_PARTITION variable = l_upb dim = 1 complete

#pragma HLS DATAFLOW
    memSelStream<t_PairInType>(l_num, p_sel, p_p0, p_p1, l_p[0]);
    xf::blas::mem2stream(l_num, p_v2dt2, l_vt[0]);
    for (int i = 0; i < t_NumInstance; i++) {
#pragma HLS UNROLL
        s[i].forward(p_src[i], l_upb[i], l_vt[i], l_vt[i + 1], l_p[i], l_p[i + 1]);
    }
    dataConsumer(l_num, l_vt[t_NumInstance]);
    RTM_TYPE::saveUpb<t_NumInstance>(l_upbSize, p_t, l_upb, p_upb);
    streamSelMem<t_PairInType>(l_num, p_sel, p_pp1, p_pp0, l_p[t_NumInstance]);
}

extern "C" void top(const bool p_sel,
                    const unsigned int p_z,
                    const unsigned int p_x,
                    const unsigned int p_t,
                    const unsigned int p_srcz,
                    const unsigned int p_srcx,
                    const DATATYPE p_src[NTime],
                    const DATATYPE p_coefz[ORDER + 1],
                    const DATATYPE p_coefx[ORDER + 1],
                    const DATATYPE p_taperz[NZB],
                    const DATATYPE p_taperx[NXB],
                    const IN_TYPE p_v2dt2[WIDTH * HEIGHT / nPE],
                    PAIRIN_TYPE p_p0[WIDTH * HEIGHT / nPE],
                    PAIRIN_TYPE p_p1[WIDTH * HEIGHT / nPE],
                    PAIRIN_TYPE p_pp0[WIDTH * HEIGHT / nPE],
                    PAIRIN_TYPE p_pp1[WIDTH * HEIGHT / nPE],
                    UPB_TYPE p_upb[WIDTH * NTime]) {
    RTM_TYPE l_s[NUM_INST];
#pragma HLS ARRAY_PARTITION variable = l_s complete dim = 1
    DATATYPE l_src[NUM_INST];
#pragma HLS ARRAY_PARTITION variable = l_src complete dim = 1

    for (int i = 0; i < NUM_INST; i++) {
        l_s[i].setDim(p_z, p_x);
        l_s[i].setCoef(p_coefz, p_coefx);
        l_s[i].setSrc(p_srcz, p_srcx);
        l_s[i].setTaper(p_taperz, p_taperx);
    }
    for (int i = 0; i < NUM_INST; i++) {
        l_src[i] = p_src[p_t * NUM_INST + i];
    }
    rtmforward<NUM_INST, DATATYPE, PAIRIN_TYPE, UPB_TYPE>(l_s, p_sel, p_t, l_src, p_v2dt2, p_upb, p_p0, p_p1, p_pp0,
                                                          p_pp1);
}
