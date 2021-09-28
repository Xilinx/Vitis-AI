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

#ifndef XF_BLAS_GEMM_LOAD_KERNELS_HPP
#define XF_BLAS_GEMM_LOAD_KERNELS_HPP

#ifndef __cplusplus
#error "BLAS Library only works with C++."
#endif

#include "blasInstr.hpp"
#include "gemmMatMoverL2.hpp"
namespace xf {
namespace blas {

template <unsigned int t_MaxNumInstrs,
          unsigned int t_MemWordsPerInstr,
          unsigned int t_ParEntries,
          unsigned int t_MparWords,
          unsigned int t_KparWords,
          unsigned int t_NparWords,
          unsigned int t_MemBits>
void gemmLoad(ap_uint<t_MemBits>* p_ptr,
              hls::stream<ap_uint<t_MemBits> >& p_aStr,
              hls::stream<ap_uint<t_MemBits> >& p_bStr,
              hls::stream<ap_uint<t_MemBits> >& p_xStr,
              hls::stream<ap_uint<16> >& p_opCodeStr) {
    static const unsigned int t_MemWordBytes = t_MemBits / 8;

    ap_uint<t_MemBits> l_progMem[t_MaxNumInstrs][t_MemWordsPerInstr];
#pragma HLS ARRAY_PARTITION variable = l_progMem complete dim = 2

    // load Instructions
    loadInstr<t_MaxNumInstrs, t_MemWordsPerInstr, t_MemBits>(p_ptr, l_progMem);

    unsigned int l_pc = 0;
    uint16_t l_opCode = OpCodeType::OpControl;
    ap_uint<t_MemBits> l_instr[t_MemWordsPerInstr];
    do {
#pragma HLS ARRAY_PARTITION variable = l_instr complete dim = 1
        getInstr<t_MaxNumInstrs, t_MemWordsPerInstr, t_MemBits>(l_progMem, l_pc, l_instr);
        l_opCode = getOpCode<t_MemWordsPerInstr, t_MemBits>(l_instr);
        switch (l_opCode) {
            case OpCodeType::OpGemm: {
                p_opCodeStr.write(l_opCode);
                for (unsigned int i = 0; i < t_MemWordsPerInstr; ++i) {
#pragma HLS PIPELINE
                    p_aStr.write(l_instr[i]);
                    p_bStr.write(l_instr[i]);
                    p_xStr.write(l_instr[i]);
                }
                MemInstr<t_MemWordsPerInstr * t_MemBits / 8> l_memInstr;
                l_memInstr.template loadMem<t_MemBits / 8>(l_instr);
                GemmInstr<t_MemWordsPerInstr * t_MemBits / 8> l_gemmInstr;
                l_gemmInstr.load(l_memInstr);
                ap_uint<t_MemBits>* l_aPtr = p_ptr + l_gemmInstr.m_aOffset * 8 / t_MemBits;
                ap_uint<t_MemBits>* l_bPtr = p_ptr + l_gemmInstr.m_bOffset * 8 / t_MemBits;
                ap_uint<t_MemBits>* l_xPtr = p_ptr + l_gemmInstr.m_xOffset * 8 / t_MemBits;
                gemmLoadDat<t_ParEntries, t_MparWords, t_KparWords, t_NparWords, t_MemBits>(
                    l_aPtr, l_bPtr, l_xPtr, l_gemmInstr.m_m, l_gemmInstr.m_k, l_gemmInstr.m_n, p_aStr, p_bStr, p_xStr);
                break;
            }
            case OpCodeType::OpControl: {
                break;
            }
            default: {
#ifndef __SYNTHESIS__
                assert(false);
#endif
            }
        }
        l_pc++;
    } while ((l_pc < t_MaxNumInstrs) && (l_opCode != OpCodeType::OpControl));

    encodeControlInstr<t_MemWordsPerInstr, t_MemBits>(l_instr);
    for (unsigned int i = 0; i < t_MemWordsPerInstr; ++i) {
#pragma HLS PIPELINE
        p_aStr.write(l_instr[i]);
        p_bStr.write(l_instr[i]);
        p_xStr.write(l_instr[i]);
    }
    p_opCodeStr.write(OpCodeType::OpControl);
}
}
}
#endif
