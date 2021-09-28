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

#ifndef XF_BLAS_BLASINSTR_HPP
#define XF_BLAS_BLASINSTR_HPP

#ifndef __cplusplus
#error "BLAS Library only works with C++."
#endif

#include <cstdint>
#include "ap_int.h"
#include "hls_stream.h"
#include "types.hpp"
#include "ISA.hpp"

namespace xf {
namespace blas {

template <unsigned int t_MaxNumInstrs, unsigned int t_MemWordsPerInstr, unsigned int t_MemBits>
void loadInstr(ap_uint<t_MemBits>* p_ptr, ap_uint<t_MemBits> p_progMem[t_MaxNumInstrs][t_MemWordsPerInstr]) {
#pragma HLS INLINE
    for (unsigned int i = 0; i < t_MaxNumInstrs; ++i) {
        for (unsigned int j = 0; j < t_MemWordsPerInstr; ++j) {
#pragma HLS PIPELINE
            p_progMem[i][j] = p_ptr[i * t_MemWordsPerInstr + j];
        }
    }
}

template <unsigned int t_MaxNumInstrs, unsigned int t_MemWordsPerInstr, unsigned int t_MemBits>
void getInstr(ap_uint<t_MemBits> p_progMem[t_MaxNumInstrs][t_MemWordsPerInstr],
              const unsigned int p_id,
              ap_uint<t_MemBits> p_instr[t_MemWordsPerInstr]) {
#pragma HLS INLINE
    for (unsigned int i = 0; i < t_MemWordsPerInstr; ++i) {
#pragma HLS UNROLL
        p_instr[i] = p_progMem[p_id][i];
    }
}

template <unsigned int t_MemWordsPerInstr, unsigned int t_MemBits>
uint16_t getOpCode(ap_uint<t_MemBits> p_instr[t_MemWordsPerInstr]) {
#pragma HLS INLINE
    ap_uint<t_MemBits> l_val = p_instr[0];
    uint16_t l_opCode = l_val.range(15, 0);
    return l_opCode;
}

template <unsigned int t_MemWordsPerInstr, unsigned int t_MemBits>
void decodeGemmLdStInstr(ap_uint<t_MemBits> p_instr[t_MemWordsPerInstr],
                         uint32_t& p_aOffset,
                         uint32_t& p_bOffset,
                         uint32_t& p_xOffset,
                         uint32_t& p_aWrOffset,
                         uint32_t& p_bWrOffset,
                         uint32_t& p_xWrOffset,
                         uint32_t& p_m,
                         uint32_t& p_k,
                         uint32_t& p_n) {
#pragma HLS INLINE
    static const unsigned int t_Bits = t_MemBits * t_MemWordsPerInstr;
    static const unsigned int t_Bytes = t_Bits / 8;
    static const unsigned int t_MemWordBytes = t_MemBits / 8;
    ap_uint<t_Bits> l_val;
    for (unsigned int i = 0; i < t_MemWordsPerInstr; ++i) {
#pragma HLS UNROLL
        l_val.range((i + 1) * t_MemBits - 1, i * t_MemBits) = p_instr[i];
    }
    WideType<uint8_t, t_Bytes> l_wVal(l_val);
    MemInstr<t_Bytes> l_memInstr;
    for (unsigned int i = 0; i < t_Bytes; ++i) {
#pragma HLS UNROLL
        l_memInstr[i] = l_wVal[i];
    }
    GemmLdStInstr<t_Bytes> l_gemmLdStInstr;
    l_gemmLdStInstr.load(l_memInstr);
    p_aOffset = l_gemmLdStInstr.m_aOffset / t_MemWordBytes;
    p_bOffset = l_gemmLdStInstr.m_bOffset / t_MemWordBytes;
    p_xOffset = l_gemmLdStInstr.m_xOffset / t_MemWordBytes;
    p_aWrOffset = l_gemmLdStInstr.m_aWrOffset / t_MemWordBytes;
    p_bWrOffset = l_gemmLdStInstr.m_bWrOffset / t_MemWordBytes;
    p_xWrOffset = l_gemmLdStInstr.m_xWrOffset / t_MemWordBytes;
    p_m = l_gemmLdStInstr.m_m;
    p_k = l_gemmLdStInstr.m_k;
    p_n = l_gemmLdStInstr.m_n;
}

template <unsigned int t_MemWordsPerInstr, unsigned int t_MemBits>
void encodeControlInstr(ap_uint<t_MemBits> p_instr[t_MemWordsPerInstr]) {
#pragma HLS INLINE
    static const unsigned int t_Bits = t_MemBits * t_MemWordsPerInstr;
    static const unsigned int t_Bytes = t_Bits / 8;
    static const unsigned int t_MemWordBytes = t_MemBits / 8;
    ControlInstr<t_Bytes> l_controlInstr(true, false);
    MemInstr<t_Bytes> l_memInstr;
    l_controlInstr.store(l_memInstr);
    WideType<uint8_t, t_Bytes> l_wVal;
    for (unsigned int i = 0; i < t_Bytes; ++i) {
#pragma HLS UNROLL
        l_wVal[i] = l_memInstr[i];
    }
    ap_uint<t_Bits> l_val = l_wVal;
    for (unsigned int i = 0; i < t_MemWordsPerInstr; ++i) {
#pragma HLS UNROLL
        p_instr[i] = l_val.range((i + 1) * t_MemBits - 1, i * t_MemBits);
    }
}
template <unsigned int t_MemWordsPerInstr, unsigned int t_MemBits>
void encodeResInstr(uint32_t p_cycles, ap_uint<t_MemBits> p_instr[t_MemWordsPerInstr]) {
#pragma HLS INLINE
    static const unsigned int t_Bits = t_MemBits * t_MemWordsPerInstr;
    static const unsigned int t_Bytes = t_Bits / 8;
    static const unsigned int t_MemWordBytes = t_MemBits / 8;
    ResInstr<t_Bytes> l_resInstr(0, p_cycles);
    MemInstr<t_Bytes> l_memInstr;
    l_resInstr.store(l_memInstr);
    WideType<uint8_t, t_Bytes> l_wVal;
    for (unsigned int i = 0; i < t_Bytes; ++i) {
#pragma HLS UNROLL
        l_wVal[i] = l_memInstr[i];
    }
    ap_uint<t_Bits> l_val = l_wVal;
    for (unsigned int i = 0; i < t_MemWordsPerInstr; ++i) {
#pragma HLS UNROLL
        p_instr[i] = l_val.range((i + 1) * t_MemBits - 1, i * t_MemBits);
    }
}
}
}
#endif
