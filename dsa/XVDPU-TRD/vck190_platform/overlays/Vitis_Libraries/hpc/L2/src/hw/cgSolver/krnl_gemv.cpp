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
#include "token.hpp"
#include "interface.hpp"
#include "krnl_gemv.hpp"
#include "update_gemv.hpp"
#include "loadMats.hpp"
#include "timer.hpp"

void proc_gemv(uint32_t p_size,
               hls::stream<typename xf::blas::WideType<CG_dataType, 1>::t_TypeInt>& p_dot,
               CG_interface* p_A0,
#if CG_numChannels > 1
               CG_interface* p_A1,
#endif
#if CG_numChannels > 2
               CG_interface* p_A2,
               CG_interface* p_A3,
#endif
#if CG_numChannels > 4
               CG_interface* p_A4,
               CG_interface* p_A5,
               CG_interface* p_A6,
               CG_interface* p_A7,
#endif
#if CG_numChannels > 8
               CG_interface* p_A8,
               CG_interface* p_A9,
               CG_interface* p_Aa,
               CG_interface* p_Ab,
               CG_interface* p_Ac,
               CG_interface* p_Ad,
               CG_interface* p_Ae,
               CG_interface* p_Af,
#endif
               CG_interface* p_pk,
               CG_vecInterface* p_pkc,
               CG_vecInterface* p_Apk) {
    hls::stream<CG_interface> l_strA[CG_numChannels];
    hls::stream<CG_interface> l_strP;
    hls::stream<CG_vecInterface> l_strPc, l_Apk;
#pragma HLS DATAFLOW
    xf::blas::loadMats<CG_dataType, CG_parEntries>(p_size / CG_numChannels, p_size, p_A0,
#if CG_numChannels > 1
                                                   p_A1,
#endif
#if CG_numChannels > 2
                                                   p_A2, p_A3,
#endif
#if CG_numChannels > 4
                                                   p_A4, p_A5, p_A6, p_A7,
#endif
#if CG_numChannels > 8
                                                   p_A8, p_A9, p_Aa, p_Ab, p_Ac, p_Ad, p_Ae, p_Af,
#endif
                                                   l_strA);
    xf::blas::mem2stream(p_size / CG_parEntries, p_pk, l_strP, p_size / CG_numChannels);
    xf::blas::mem2stream(p_size / CG_vecParEntries, p_pkc, l_strPc);
    xf::hpc::cg::update_gemv<CG_dataType, CG_parEntries, CG_vecParEntries, CG_numChannels>(
        p_size, p_dot, l_strA, l_strP, l_strPc, l_Apk, p_size / CG_numChannels);
    xf::blas::stream2mem<CG_vecInterface>(p_size / CG_vecParEntries, l_Apk, p_Apk);
}

extern "C" void krnl_gemv(CG_interface* p_A0,
#if CG_numChannels > 1
                          CG_interface* p_A1,
#endif
#if CG_numChannels > 2
                          CG_interface* p_A2,
                          CG_interface* p_A3,
#endif
#if CG_numChannels > 4
                          CG_interface* p_A4,
                          CG_interface* p_A5,
                          CG_interface* p_A6,
                          CG_interface* p_A7,
#endif
#if CG_numChannels > 8
                          CG_interface* p_A8,
                          CG_interface* p_A9,
                          CG_interface* p_Aa,
                          CG_interface* p_Ab,
                          CG_interface* p_Ac,
                          CG_interface* p_Ad,
                          CG_interface* p_Ae,
                          CG_interface* p_Af,
#endif
                          CG_interface* p_pk,
                          CG_vecInterface* p_pkc,
                          CG_vecInterface* p_Apk,
                          hls::stream<ap_uint<CG_tkStrWidth> >& p_tokenIn,
                          hls::stream<ap_uint<CG_tkStrWidth> >& p_tokenOut) {
    POINTER(p_A0, gmem_A0)
#if CG_numChannels > 1
    POINTER(p_A1, gmem_A1)
#endif
#if CG_numChannels > 2
    POINTER(p_A2, gmem_A2)
    POINTER(p_A3, gmem_A3)
#endif
#if CG_numChannels > 4
    POINTER(p_A4, gmem_A4)
    POINTER(p_A5, gmem_A5)
    POINTER(p_A6, gmem_A6)
    POINTER(p_A7, gmem_A7)
#endif
#if CG_numChannels > 8
    POINTER(p_A8, gmem_A8)
    POINTER(p_A9, gmem_A9)
    POINTER(p_Aa, gmem_Aa)
    POINTER(p_Ab, gmem_Ab)
    POINTER(p_Ac, gmem_Ac)
    POINTER(p_Ad, gmem_Ad)
    POINTER(p_Ae, gmem_Ae)
    POINTER(p_Af, gmem_Af)
#endif
    POINTER(p_pk, gmem_pk)
    POINTER(p_pkc, gmem_pkc)
    POINTER(p_Apk, gmem_Apk)
    AXIS(p_tokenIn)
    AXIS(p_tokenOut)
    SCALAR(return )

    xf::hpc::cg::Token<CG_dataType> l_token;
    xf::hpc::StreamInstr<sizeof(l_token)> l_cs;
    l_token.read_decode(p_tokenIn, l_cs);

    while (!l_token.getExit()) {
        uint32_t l_size = l_token.getVecSize();
        hls::stream<typename xf::blas::WideType<CG_dataType, 1>::t_TypeInt> l_dotStr;
        proc_gemv(l_size, l_dotStr, p_A0,
#if CG_numChannels > 1
                  p_A1,
#endif
#if CG_numChannels > 2
                  p_A2, p_A3,
#endif
#if CG_numChannels > 4
                  p_A4, p_A5, p_A6, p_A7,
#endif
#if CG_numChannels > 8
                  p_A8, p_A9, p_Aa, p_Ab, p_Ac, p_Ad, p_Ae, p_Af,
#endif
                  p_pk, p_pkc, p_Apk);

        xf::blas::WideType<CG_dataType, 1> l_r = l_dotStr.read();
        CG_dataType l_dot = l_r[0];
        l_token.setAlpha(l_token.getRZ() / l_dot);
        l_token.encode_write(p_tokenOut, l_cs);
        l_token.read_decode(p_tokenIn, l_cs);
    }
    l_token.encode_write(p_tokenOut, l_cs);
}
