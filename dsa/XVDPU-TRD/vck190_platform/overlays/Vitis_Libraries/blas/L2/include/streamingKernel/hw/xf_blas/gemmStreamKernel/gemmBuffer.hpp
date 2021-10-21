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
#ifndef XF_BLAS_GEMM_STREAM_KERNEL_HPP
#define XF_BLAS_GEMM_STREAM_KERNEL_HPP

#include "blasInstr.hpp"
#include "gemmKernel.hpp"
namespace xf {
namespace blas {

template <typename t_DataType,
          unsigned int t_MemWordsPerInstr,
          unsigned int t_ParEntries,
          unsigned int t_MparWords,
          unsigned int t_KparWords,
          unsigned int t_NparWords,
          unsigned int t_MemBits>
void gemmBuffer(hls::stream<ap_uint<t_MemBits> >& p_aStr,
                hls::stream<ap_uint<t_MemBits> >& p_bStr,
                hls::stream<ap_uint<t_MemBits> >& p_xStr,
                hls::stream<ap_uint<t_MemBits> >& p_cStr,
                hls::stream<ap_uint<t_MemBits + 2> >& p_tagA,
                hls::stream<ap_uint<t_MemBits> >& p_tagB,
                hls::stream<ap_uint<t_MemBits> >& p_sumStr) {
    GemmKernel<t_DataType, t_DataType, t_ParEntries, t_ParEntries, t_MparWords, t_KparWords, t_NparWords> l_gemmKernel;

    ap_uint<t_MemBits> l_instrA[t_MemWordsPerInstr];
    ap_uint<t_MemBits> l_instrB[t_MemWordsPerInstr];
    ap_uint<t_MemBits> l_instrX[t_MemWordsPerInstr];
#pragma HLS ARRAY_PARTITION variable = l_instrA complete dim = 1
#pragma HLS ARRAY_PARTITION variable = l_instrB complete dim = 1
#pragma HLS ARRAY_PARTITION variable = l_instrX complete dim = 1
    for (unsigned int i = 0; i < t_MemWordsPerInstr; ++i) {
#pragma HLS PIPELINE
        l_instrA[i] = p_aStr.read();
        l_instrB[i] = p_bStr.read();
        l_instrX[i] = p_xStr.read();
        p_cStr.write(l_instrA[i]);
    }

    uint16_t l_opCode = getOpCode<t_MemWordsPerInstr, t_MemBits>(l_instrA);
    while (l_opCode != OpCodeType::OpControl) {
        MemInstr<t_MemWordsPerInstr * t_MemBits / 8> l_memInstr;

        l_memInstr.template loadMem<t_MemBits / 8>(l_instrA);
        GemmInstr<t_MemWordsPerInstr * t_MemBits / 8> l_gemmInstr;
        l_gemmInstr.load(l_memInstr);

        const unsigned int l_aColBlocks = l_gemmInstr.m_k / (t_ParEntries * t_KparWords);
        const unsigned int l_aRowBlocks = l_gemmInstr.m_m / (t_ParEntries * t_MparWords);
        const unsigned int l_bColBlocks = l_gemmInstr.m_n / (t_ParEntries * t_NparWords);
        const unsigned int l_transpBlocks = l_aColBlocks * l_aRowBlocks * l_bColBlocks * t_MparWords;
        l_gemmKernel.GemmBuffers(p_aStr, p_bStr, p_xStr, p_cStr, p_tagA, p_tagB, p_sumStr, l_aColBlocks, l_aRowBlocks,
                                 l_bColBlocks, l_transpBlocks, 1);

        for (unsigned int i = 0; i < t_MemWordsPerInstr; ++i) {
#pragma HLS PIPELINE
            l_instrA[i] = p_aStr.read();
            l_instrB[i] = p_bStr.read();
            l_instrX[i] = p_xStr.read();
            p_cStr.write(l_instrA[i]);
        }
        l_opCode = getOpCode<t_MemWordsPerInstr, t_MemBits>(l_instrA);
    }
}
}
}

#endif
