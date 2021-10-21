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

#include "backward.hpp"
template <unsigned int t_NumInstance, typename t_DataType, typename t_TypeInt, typename t_PairType, typename t_UpbType>
void rtmbackward(RTM_TYPE sf[t_NumInstance],
                 RTM_TYPE sr[t_NumInstance],
                 const bool p_sel,
                 const unsigned int p_t,
                 const unsigned int p_T,
                 const t_TypeInt* p_v2dt2,
                 const t_DataType* p_rec,
                 const t_UpbType* p_upb,
                 t_PairType* p_p0,
                 t_PairType* p_p1,
                 t_PairType* p_pp0,
                 t_PairType* p_pp1,
                 t_PairType* p_r0,
                 t_PairType* p_r1,
                 t_PairType* p_rr0,
                 t_PairType* p_rr1,
                 t_TypeInt* p_i0,
                 t_TypeInt* p_i1,
                 t_TypeInt* p_ii0,
                 t_TypeInt* p_ii1) {
    const unsigned int l_num = sr[0].getArea();
    const unsigned int l_upbSize = sr[0].getX();
    const unsigned int l_recSize = sr[0].getX() - 2 * sr[0].getXB();
    const unsigned int l_imgSize = l_recSize * (sr[0].getZ() - 2 * sr[0].getZB()) / nPE;
    int l_t = p_t * t_NumInstance;

#pragma HLS DATAFLOW
    // Velocity model
    hls::stream<t_TypeInt> l_v2dt2, l_pvt[t_NumInstance + 1], l_rvt[t_NumInstance + 1];
#pragma HLS stream variable = l_v2dt2 depth = 4
#pragma HLS ARRAY_PARTITION variable = l_pvt dim = 1 complete
#pragma HLS ARRAY_PARTITION variable = l_rvt dim = 1 complete
#pragma HLS stream depth = RTM_TYPE::t_FifoDepth variable = l_pvt
#pragma HLS stream depth = RTM_TYPE::t_FifoDepth variable = l_rvt

    xf::blas::mem2stream(l_num, p_v2dt2, l_v2dt2);
    duplicate(l_num, l_v2dt2, l_pvt[0], l_rvt[0]);
    hls::stream<t_TypeInt> l_cp[t_NumInstance], l_cr[t_NumInstance];
#pragma HLS ARRAY_PARTITION variable = l_cp dim = 1 complete
#pragma HLS ARRAY_PARTITION variable = l_cr dim = 1 complete
#pragma HLS stream variable = l_cp depth = 32
#pragma HLS stream variable = l_cr depth = 32

    // Direct Wave
    hls::stream<t_PairType> l_p[t_NumInstance + 1];
#pragma HLS ARRAY_PARTITION variable = l_p dim = 1 complete
    hls::stream<t_UpbType> l_upb[t_NumInstance];
#pragma HLS ARRAY_PARTITION variable = l_upb dim = 1 complete
    RTM_TYPE::loadUpb<t_NumInstance>(l_upbSize, p_t, l_upb, p_upb);
    memSelStream<t_PairType>(l_num, p_sel, p_p0, p_p1, l_p[0]);
    for (int i = 0; i < t_NumInstance; i++) {
#pragma HLS UNROLL
        sf[i].backwardF(l_upb[i], l_pvt[i], l_pvt[i + 1], l_p[i], l_p[i + 1], l_cp[i],
                        (l_t + t_NumInstance - i) == p_T || p_T - 1 == (l_t + t_NumInstance - i));
    }
    dataConsumer(l_num, l_pvt[t_NumInstance]);
    streamSelMem<t_PairType>(l_num, p_sel, p_pp1, p_pp0, l_p[t_NumInstance]);

    // Receiver Wave
    hls::stream<t_PairType> l_r[t_NumInstance + 1];
#pragma HLS ARRAY_PARTITION variable = l_r dim = 1 complete
    hls::stream<t_DataType> l_rec[t_NumInstance];
#pragma HLS ARRAY_PARTITION variable = l_rec dim = 1 complete
    RTM_TYPE::loadUpb<t_NumInstance>(l_recSize, p_t, l_rec, p_rec);
    memSelStream<t_PairType>(l_num, p_sel, p_r0, p_r1, l_r[0]);
    for (int i = 0; i < t_NumInstance; i++) {
#pragma HLS UNROLL
        sr[i].backwardR(l_rec[i], l_rvt[i], l_rvt[i + 1], l_r[i], l_r[i + 1], l_cr[i]);
    }

    dataConsumer(l_num, l_rvt[t_NumInstance]);
    streamSelMem<t_PairType>(l_num, p_sel, p_rr1, p_rr0, l_r[t_NumInstance]);

    // Image reBuild
    hls::stream<t_TypeInt> l_i[t_NumInstance + 1];
#pragma HLS ARRAY_PARTITION variable = l_i dim = 1 complete
#pragma HLS stream depth = RTM_TYPE::t_FifoDepth variable = l_i
    memSelStream<t_TypeInt>(l_imgSize, p_sel, p_i0, p_i1, l_i[0]);
    for (int i = 0; i < t_NumInstance; i++) {
#pragma HLS UNROLL
        sr[i].crossCorrelation(l_cp[i], l_cr[i], l_i[i], l_i[i + 1]);
    }
    streamSelMem<t_TypeInt>(l_imgSize, p_sel, p_ii1, p_ii0, l_i[t_NumInstance]);
}

extern "C" void top(const bool p_sel,
                    const unsigned int p_z,
                    const unsigned int p_x,
                    const unsigned int p_t,
                    const unsigned int p_T,
                    const unsigned int p_recz,
                    const DATATYPE p_rec[NX * NTime],
                    const DATATYPE p_coefz[ORDER + 1],
                    const DATATYPE p_coefx[ORDER + 1],
                    const DATATYPE p_taperz[NZB],
                    const DATATYPE p_taperx[NXB],
                    const IN_TYPE p_v2dt2[WIDTH * HEIGHT / nPE],
                    PAIRIN_TYPE p_p0[WIDTH * HEIGHT / nPE],
                    PAIRIN_TYPE p_p1[WIDTH * HEIGHT / nPE],
                    PAIRIN_TYPE p_pp0[WIDTH * HEIGHT / nPE],
                    PAIRIN_TYPE p_pp1[WIDTH * HEIGHT / nPE],
                    PAIRIN_TYPE p_r0[WIDTH * HEIGHT / nPE],
                    PAIRIN_TYPE p_r1[WIDTH * HEIGHT / nPE],
                    PAIRIN_TYPE p_rr0[WIDTH * HEIGHT / nPE],
                    PAIRIN_TYPE p_rr1[WIDTH * HEIGHT / nPE],
                    IN_TYPE p_i0[NX * NZ / nPE],
                    IN_TYPE p_i1[NX * NZ / nPE],
                    IN_TYPE p_ii0[NX * NZ / nPE],
                    IN_TYPE p_ii1[NX * NZ / nPE],
                    const UPB_TYPE p_upb[WIDTH * NTime]) {
    RTM_TYPE l_p[NUM_INST];
    RTM_TYPE l_r[NUM_INST];
#pragma HLS ARRAY_PARTITION variable = l_p complete dim = 1
#pragma HLS ARRAY_PARTITION variable = l_r complete dim = 1

    for (int i = 0; i < NUM_INST; i++) {
        l_p[i].setDim(p_z, p_x);
        l_p[i].setCoef(p_coefz, p_coefx);
        l_p[i].setTaper(p_taperz, p_taperx);
        l_p[i].setReceiver(p_recz);
        l_r[i].setDim(p_z, p_x);
        l_r[i].setCoef(p_coefz, p_coefx);
        l_r[i].setTaper(p_taperz, p_taperx);
        l_r[i].setReceiver(p_recz);
    }
    rtmbackward<NUM_INST, DATATYPE, IN_TYPE, PAIRIN_TYPE, UPB_TYPE>(l_p, l_r, p_sel, p_t, p_T, p_v2dt2, p_rec, p_upb,
                                                                    p_p0, p_p1, p_pp0, p_pp1, p_r0, p_r1, p_rr0, p_rr1,
                                                                    p_i0, p_i1, p_ii0, p_ii1);
}
