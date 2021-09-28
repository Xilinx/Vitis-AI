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
#ifndef XF_BLAS_GEMM_SYSTOLIC_ARRAY_KERNEL_HPP
#define XF_BLAS_GEMM_SYSTOLIC_ARRAY_KERNEL_HPP

#include "blasInstr.hpp"
#include "gemmKernel.hpp"
namespace xf {
namespace blas {

template <typename t_DataType,
          unsigned int t_ParEntriesN,
          typename t_TagType = TaggedDualWideType<t_DataType, t_ParEntriesN> >
void dataFwd(hls::stream<typename t_TagType::t_TypeInt>& p_dataIn,
             hls::stream<typename t_TagType::t_TypeInt>& p_dataOut) {
    bool l_exit = false;
    while (!l_exit) {
#pragma HLS PIPELINE
        t_TagType l_c = p_dataIn.read();
        l_exit = l_c.m_exit;
        p_dataOut.write(l_c);
    }
}

template <typename t_DataType>
void dataTagFwd(hls::stream<t_DataType>& p_dataIn,
                hls::stream<ap_uint<2> >& p_tagIn,
                hls::stream<t_DataType>& p_dataOut,
                hls::stream<ap_uint<2> >& p_tagOut) {
    bool l_exit = false;
    while (!l_exit) {
#pragma HLS PIPELINE
        t_DataType l_dat = p_dataIn.read();
        p_dataOut.write(l_dat);
        ap_uint<2> l_tag = p_tagIn.read();
        p_tagOut.write(l_tag);
        l_exit = l_tag[0];
    }
}

template <typename t_DataType, unsigned int t_ParEntries, unsigned int t_KparWords, unsigned int t_MemBits>
#ifdef GemmStreamSeq
void gemmSystolicArray(hls::stream<ap_uint<t_MemBits> >& p_a,
                       hls::stream<ap_uint<2> >& p_tag,
                       hls::stream<ap_uint<t_MemBits> >& p_b,
                       hls::stream<ap_uint<t_MemBits> >& p_bOut,
                       hls::stream<t_DataType> p_aOut[t_ParEntries],
                       hls::stream<ap_uint<2> > p_tagOut[t_ParEntries]) {
    SystolicArray<t_DataType, t_KparWords * t_ParEntries, t_ParEntries> sa;
    sa.systolicArray(p_a, p_tag, p_b, p_aOut, p_tagOut, p_bOut);
}

#else
void gemmSystolicArray(hls::stream<ap_uint<t_MemBits> >& p_a,
                       hls::stream<ap_uint<2> >& p_tag,
                       hls::stream<ap_uint<t_MemBits> >& p_b,
                       hls::stream<ap_uint<t_MemBits> >& p_bOut,
                       hls::stream<t_DataType>& p_aOut_0,
                       hls::stream<t_DataType>& p_aOut_1,
                       hls::stream<t_DataType>& p_aOut_2,
                       hls::stream<t_DataType>& p_aOut_3,
                       hls::stream<t_DataType>& p_aOut_4,
                       hls::stream<t_DataType>& p_aOut_5,
                       hls::stream<t_DataType>& p_aOut_6,
                       hls::stream<t_DataType>& p_aOut_7,
                       hls::stream<t_DataType>& p_aOut_8,
                       hls::stream<t_DataType>& p_aOut_9,
                       hls::stream<t_DataType>& p_aOut_10,
                       hls::stream<t_DataType>& p_aOut_11,
                       hls::stream<t_DataType>& p_aOut_12,
                       hls::stream<t_DataType>& p_aOut_13,
                       hls::stream<t_DataType>& p_aOut_14,
                       hls::stream<t_DataType>& p_aOut_15,
                       hls::stream<ap_uint<2> >& p_tagOut_0,
                       hls::stream<ap_uint<2> >& p_tagOut_1,
                       hls::stream<ap_uint<2> >& p_tagOut_2,
                       hls::stream<ap_uint<2> >& p_tagOut_3,
                       hls::stream<ap_uint<2> >& p_tagOut_4,
                       hls::stream<ap_uint<2> >& p_tagOut_5,
                       hls::stream<ap_uint<2> >& p_tagOut_6,
                       hls::stream<ap_uint<2> >& p_tagOut_7,
                       hls::stream<ap_uint<2> >& p_tagOut_8,
                       hls::stream<ap_uint<2> >& p_tagOut_9,
                       hls::stream<ap_uint<2> >& p_tagOut_10,
                       hls::stream<ap_uint<2> >& p_tagOut_11,
                       hls::stream<ap_uint<2> >& p_tagOut_12,
                       hls::stream<ap_uint<2> >& p_tagOut_13,
                       hls::stream<ap_uint<2> >& p_tagOut_14,
                       hls::stream<ap_uint<2> >& p_tagOut_15) {
#pragma HLS DATAFLOW
#pragma HLS INTERFACE ap_ctrl_none port = return

    hls::stream<t_DataType> l_aOut[t_ParEntries];
    hls::stream<ap_uint<2> > l_tagOut[t_ParEntries];
#pragma HLS ARRAY_PARTITION variable = l_aOut dim = 1 complete
#pragma HLS ARRAY_PARTITION variable = l_tagOut dim = 1 complete
    SystolicArray<t_DataType, t_KparWords * t_ParEntries, t_ParEntries> sa;
    sa.systolicArray(p_a, p_tag, p_b, l_aOut, l_tagOut, p_bOut);
    dataTagFwd<t_DataType>(l_aOut[0], l_tagOut[0], p_aOut_0, p_tagOut_0);
    dataTagFwd<t_DataType>(l_aOut[1], l_tagOut[1], p_aOut_1, p_tagOut_1);
    dataTagFwd<t_DataType>(l_aOut[2], l_tagOut[2], p_aOut_2, p_tagOut_2);
    dataTagFwd<t_DataType>(l_aOut[3], l_tagOut[3], p_aOut_3, p_tagOut_3);
    dataTagFwd<t_DataType>(l_aOut[4], l_tagOut[4], p_aOut_4, p_tagOut_4);
    dataTagFwd<t_DataType>(l_aOut[5], l_tagOut[5], p_aOut_5, p_tagOut_5);
    dataTagFwd<t_DataType>(l_aOut[6], l_tagOut[6], p_aOut_6, p_tagOut_6);
    dataTagFwd<t_DataType>(l_aOut[7], l_tagOut[7], p_aOut_7, p_tagOut_7);
    dataTagFwd<t_DataType>(l_aOut[8], l_tagOut[8], p_aOut_8, p_tagOut_8);
    dataTagFwd<t_DataType>(l_aOut[9], l_tagOut[9], p_aOut_9, p_tagOut_9);
    dataTagFwd<t_DataType>(l_aOut[10], l_tagOut[10], p_aOut_10, p_tagOut_10);
    dataTagFwd<t_DataType>(l_aOut[11], l_tagOut[11], p_aOut_11, p_tagOut_11);
    dataTagFwd<t_DataType>(l_aOut[12], l_tagOut[12], p_aOut_12, p_tagOut_12);
    dataTagFwd<t_DataType>(l_aOut[13], l_tagOut[13], p_aOut_13, p_tagOut_13);
    dataTagFwd<t_DataType>(l_aOut[14], l_tagOut[14], p_aOut_14, p_tagOut_14);
    dataTagFwd<t_DataType>(l_aOut[15], l_tagOut[15], p_aOut_15, p_tagOut_15);
}
#endif
}
}

#endif
