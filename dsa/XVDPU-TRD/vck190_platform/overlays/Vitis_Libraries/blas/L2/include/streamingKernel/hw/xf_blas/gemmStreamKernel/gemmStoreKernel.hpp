/**********
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
 * **********/

#ifndef XF_BLAS_GEMM_STORE_KERNELS_HPP
#define XF_BLAS_GEMM_STORE_KERNELS_HPP

#ifndef __cplusplus
#error "BLAS Library only works with C++."
#endif

#include "blasInstr.hpp"
#include "gemmMatMoverL2.hpp"
namespace xf {
namespace blas {

template <unsigned int t_ResOffsetBytes,
          unsigned int t_MemWordsPerInstr,
          unsigned int t_ParEntries,
          unsigned int t_MparWords,
          unsigned int t_KparWords,
          unsigned int t_NparWords,
          unsigned int t_MemBits>
void gemmStore(hls::stream<ap_uint<t_MemBits> >& p_cStr,
               hls::stream<ap_uint<32> >& p_resStr,
               ap_uint<t_MemBits>* p_ptr) {
    static unsigned int t_MemWordBytes = t_MemBits / 8;
    static unsigned int t_ResOffset = t_ResOffsetBytes / t_MemWordBytes;
    static unsigned int t_InstrBytes = t_MemWordsPerInstr * t_MemWordBytes;

    ap_uint<t_MemBits> l_instrA[t_MemWordsPerInstr];
    ap_uint<t_MemBits> l_instrB[t_MemWordsPerInstr];
    ap_uint<t_MemBits> l_instrX[t_MemWordsPerInstr];
#pragma HLS ARRAY_PARTITION variable = l_instrA complete dim = 1
#pragma HLS ARRAY_PARTITION variable = l_instrB complete dim = 1
#pragma HLS ARRAY_PARTITION variable = l_instrX complete dim = 1
    for (unsigned int i = 0; i < t_MemWordsPerInstr; ++i) {
#pragma HLS PIPELINE
        l_instrA[i] = p_cStr.read();
    }
    uint16_t l_opCode = getOpCode<t_MemWordsPerInstr, t_MemBits>(l_instrA);
    while (l_opCode != OpCodeType::OpControl) {
        MemInstr<t_MemWordsPerInstr * t_MemBits / 8> l_memInstr;
        l_memInstr.template loadMem<t_MemBits / 8>(l_instrA);
        GemmInstr<t_MemWordsPerInstr * t_MemBits / 8> l_gemmInstr;
        l_gemmInstr.load(l_memInstr);

        ap_uint<t_MemBits>* l_cPtr = p_ptr + l_gemmInstr.m_cOffset * 8 / t_MemBits;

        static const unsigned int t_AblockRows = t_ParEntries * t_MparWords;
        unsigned int l_mBlocks = l_gemmInstr.m_m / (t_ParEntries * t_MparWords);
        unsigned int l_nBlocks = l_gemmInstr.m_n / (t_ParEntries * t_NparWords);
        unsigned int l_nParWords = l_gemmInstr.m_n / t_ParEntries;
        unsigned int l_cBlockOffset = l_nParWords * t_AblockRows;

        for (unsigned int ar = 0; ar < l_mBlocks; ++ar) {
            for (unsigned int bc = 0; bc < l_nBlocks; ++bc) {
                // storeC
                gemmStoreBlock<t_ParEntries, t_MparWords, t_NparWords, t_MemBits>(p_cStr, l_nParWords,
                                                                                  l_cPtr + bc * t_NparWords);
            }
            l_cPtr += l_cBlockOffset;
        }
        for (unsigned int i = 0; i < t_MemWordsPerInstr; ++i) {
#pragma HLS PIPELINE
            l_instrA[i] = p_cStr.read();
        }
        l_opCode = getOpCode<t_MemWordsPerInstr, t_MemBits>(l_instrA);
    }

    uint32_t l_cycles = (uint32_t)(p_resStr.read());
    encodeResInstr<t_MemWordsPerInstr, t_MemBits>(l_cycles, l_instrX);
    for (unsigned int i = 0; i < t_MemWordsPerInstr; ++i) {
#pragma HLS PIPELINE
        p_ptr[t_ResOffset + i] = l_instrX[i];
    }
}
}
}
#endif
