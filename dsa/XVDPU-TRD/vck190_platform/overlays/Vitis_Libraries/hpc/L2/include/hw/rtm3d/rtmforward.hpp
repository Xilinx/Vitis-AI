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

#ifndef _RTMFORWARD_HPP_
#define _RTMFORWARD_HPP_

#include "xf_blas.hpp"
#include "rtm.hpp"

namespace xf {
namespace hpc {
namespace rtm {

/**
 * @brief forward function is composed by multiple forward streaming modules
 *
 * @param p_sel determins the port id to read or write
 * @param p_domain the domain partition object
 *
 * @param p_sm an array of streaming module objects
 *
 * @param p_t  the start time step in the current process
 *
 * @param p_src  the source wavefiled
 *
 * @param p_v2dt2  the velocity model v^2 * dt^2
 *
 * @param p_pi0  the first input memory of pressure wavefield at t-1
 * @param p_pi1  the second input memory of pressure wavefield at t-1
 *
 * @param p_po0  the first output memory of pressure wavefield at t-1
 * @param p_po1  the second output memory of pressure wavefield at t-1
 *
 * @param p_ppi0  the first input memory of pressure wavefield at t-2
 * @param p_ppi1  the second input memory of pressure wavefield at t-2
 *
 * @param p_ppo0  the first output memory of pressure wavefield at t-2
 * @param p_ppo1  the second output memory of pressure wavefield at t-2
 *
 * @param p_upb  the memory to store the upper-boundary data
 */

template <int t_NumFSMs, typename t_DataType, typename t_InType, typename t_UpbType, typename t_RTM, typename t_Domain>
void forward(bool p_sel,
             const t_Domain& p_domain,
             t_RTM p_sm[t_NumFSMs],
             const unsigned int p_t,
             const t_DataType* p_src,
             const t_InType* p_v2dt2,
             t_InType* p_pi0,
             t_InType* p_pi1,
             t_InType* p_po0,
             t_InType* p_po1,
             t_InType* p_ppi0,
             t_InType* p_ppi1,
             t_InType* p_ppo0,
             t_InType* p_ppo1,
             t_UpbType* p_upb) {
    const int l_entries = p_sm[0].getCube();

    hls::stream<t_InType> l_vt_in, l_vt_out;
    hls::stream<t_InType> l_pp_in, l_pp_out;

    hls::stream<t_UpbType> l_upb[t_NumFSMs];
#pragma HLS ARRAY_PARTITION variable = l_upb complete dim = 1

    hls::stream<t_InType> l_p[t_NumFSMs + 1];
#pragma HLS ARRAY_PARTITION variable = l_p complete dim = 1

#pragma HLS DATAFLOW

    p_domain.template memSelStream(p_sel, p_pi1, p_pi0, l_p[0]);
    p_domain.template memSelStream(p_sel, p_ppi1, p_ppi0, l_pp_in);

    p_domain.template mem2stream(p_v2dt2, l_vt_in);

#if RTM_numFSMs == 1
    p_sm[0].forward(p_src[0], l_upb[0], l_vt_in, l_vt_out, l_pp_in, l_p[0], l_pp_out, l_p[1]);
#else

    hls::stream<t_InType> l_vt[t_NumFSMs - 1];
#pragma HLS ARRAY_PARTITION variable = l_vt complete dim = 1
#pragma HLS stream depth = t_RTM::t_FifoDepth variable = l_vt
#pragma HLS RESOURCE variable = l_vt core = fifo_uram

    hls::stream<t_InType> l_pp[t_NumFSMs - 1];
#pragma HLS ARRAY_PARTITION variable = l_pp complete dim = 1
#pragma HLS stream depth = t_RTM::t_FifoDepth variable = l_pp
#pragma HLS RESOURCE variable = l_pp core = fifo_uram

    p_sm[0].forward(p_src[0], l_upb[0], l_vt_in, l_vt[0], l_pp_in, l_p[0], l_pp[0], l_p[1]);
    for (int i = 1; i < t_NumFSMs - 1; i++) {
#pragma HLS UNROLL
        p_sm[i].forward(p_src[i], l_upb[i], l_vt[i - 1], l_vt[i], l_pp[i - 1], l_p[i], l_pp[i], l_p[i + 1]);
    }
    p_sm[t_NumFSMs - 1].forward(p_src[t_NumFSMs - 1], l_upb[t_NumFSMs - 1], l_vt[t_NumFSMs - 2], l_vt_out,
                                l_pp[t_NumFSMs - 2], l_p[t_NumFSMs - 1], l_pp_out, l_p[t_NumFSMs]);
#endif
    xf::hpc::dataConsumer(l_entries, l_vt_out);

    p_sm[0].template saveUpb<t_NumFSMs>(p_t, l_upb, p_upb);
    p_domain.template streamSelMem(p_sel, l_p[t_NumFSMs], p_po0, p_po1);
    p_domain.template streamSelMem(p_sel, l_pp_out, p_ppo0, p_ppo1);
}

/**
 * @brief forward function  composed by multiple forward streaming modules
 *
 * @param p_sel determins the port id to read or write
 * @param p_domain the domain partition object
 *
 * @param p_sm  an array of streaming module objects
 *
 * @param p_src  the source wavefiled
 *
 * @param p_v2dt2  the velocity model v^2 * dt^2
 *
 * @param p_pi0  the first input memory of pressure wavefield at t-1
 * @param p_pi1  the second input memory of pressure wavefield at t-1
 *
 * @param p_po0  the first output memory of pressure wavefield at t-1
 * @param p_po1  the second output memory of pressure wavefield at t-1
 *
 * @param p_ppi0  the first input memory of pressure wavefield at t-2
 * @param p_ppi1  the second input memory of pressure wavefield at t-2
 *
 * @param p_ppo0  the first output memory of pressure wavefield at t-2
 * @param p_ppo1  the second output memory of pressure wavefield at t-2
 */

template <int t_NumFSMs, typename t_DataType, typename t_InType, typename t_RTM, typename t_Domain>
void forward(bool p_sel,
             const t_Domain& p_domain,
             t_RTM p_sm[t_NumFSMs],
             const t_DataType* p_src,
             const t_InType* p_v2dt2,
             t_InType* p_pi0,
             t_InType* p_pi1,
             t_InType* p_po0,
             t_InType* p_po1,
             t_InType* p_ppi0,
             t_InType* p_ppi1,
             t_InType* p_ppo0,
             t_InType* p_ppo1) {
    const int l_entries = p_sm[0].getCube();

    hls::stream<t_InType> l_vt_in, l_vt_out;
    hls::stream<t_InType> l_pp_in, l_pp_out;

    hls::stream<t_InType> l_p[t_NumFSMs + 1];
#pragma HLS ARRAY_PARTITION variable = l_p complete dim = 1

#pragma HLS DATAFLOW

    p_domain.template memSelStream(p_sel, p_pi1, p_pi0, l_p[0]);
    p_domain.template memSelStream(p_sel, p_ppi1, p_ppi0, l_pp_in);

    p_domain.template mem2stream(p_v2dt2, l_vt_in);

#if RTM_numFSMs == 1
    p_sm[0].forward(p_src[0], l_vt_in, l_vt_out, l_pp_in, l_p[0], l_pp_out, l_p[1]);
#else

    hls::stream<t_InType> l_vt[t_NumFSMs - 1];
#pragma HLS ARRAY_PARTITION variable = l_vt complete dim = 1
#pragma HLS stream depth = t_RTM::t_FifoDepth variable = l_vt
#pragma HLS RESOURCE variable = l_vt core = fifo_uram

    hls::stream<t_InType> l_pp[t_NumFSMs - 1];
#pragma HLS ARRAY_PARTITION variable = l_pp complete dim = 1
#pragma HLS stream depth = t_RTM::t_FifoDepth variable = l_pp
#pragma HLS RESOURCE variable = l_pp core = fifo_uram

    p_sm[0].forward(p_src[0], l_vt_in, l_vt[0], l_pp_in, l_p[0], l_pp[0], l_p[1]);
    for (int i = 1; i < t_NumFSMs - 1; i++) {
#pragma HLS UNROLL
        p_sm[i].forward(p_src[i], l_vt[i - 1], l_vt[i], l_pp[i - 1], l_p[i], l_pp[i], l_p[i + 1]);
    }
    p_sm[t_NumFSMs - 1].forward(p_src[t_NumFSMs - 1], l_vt[t_NumFSMs - 2], l_vt_out, l_pp[t_NumFSMs - 2],
                                l_p[t_NumFSMs - 1], l_pp_out, l_p[t_NumFSMs]);
#endif
    xf::hpc::dataConsumer(l_entries, l_vt_out);

    p_domain.template streamSelMem(p_sel, l_p[t_NumFSMs], p_po0, p_po1);
    p_domain.template streamSelMem(p_sel, l_pp_out, p_ppo0, p_ppo1);
}
}
}
}
#endif
