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
#include "interface.hpp"
#include "loadMats.hpp"
#include "gemvMult.hpp"
#include "streamOps.hpp"
#include "gemvMultKernel.hpp"

extern "C" void krnl_gemv(uint32_t p_m,
                          uint32_t p_n,
                          BLAS_interface* p_A0,
#if BLAS_numChannels > 1
                          BLAS_interface* p_A1,
#endif
#if BLAS_numChannels > 2
                          BLAS_interface* p_A2,
                          BLAS_interface* p_A3,
#endif
#if BLAS_numChannels > 4
                          BLAS_interface* p_A4,
                          BLAS_interface* p_A5,
                          BLAS_interface* p_A6,
                          BLAS_interface* p_A7,
#endif
#if BLAS_numChannels > 8
                          BLAS_interface* p_A8,
                          BLAS_interface* p_A9,
                          BLAS_interface* p_Aa,
                          BLAS_interface* p_Ab,
                          BLAS_interface* p_Ac,
                          BLAS_interface* p_Ad,
                          BLAS_interface* p_Ae,
                          BLAS_interface* p_Af,
#endif
                          BLAS_interface* p_pk,
                          BLAS_interface* p_Apk) {

    POINTER(p_A0, gmem_A0)
#if BLAS_numChannels > 1
    POINTER(p_A1, gmem_A1)
#endif
#if BLAS_numChannels > 2
    POINTER(p_A2, gmem_A2)
    POINTER(p_A3, gmem_A3)
#endif
#if BLAS_numChannels > 4
    POINTER(p_A4, gmem_A4)
    POINTER(p_A5, gmem_A5)
    POINTER(p_A6, gmem_A6)
    POINTER(p_A7, gmem_A7)
#endif
#if BLAS_numChannels > 8
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
    POINTER(p_Apk, gmem_Apk)
    SCALAR(p_m)
    SCALAR(p_n)
    SCALAR(return )

    hls::stream<BLAS_interface> l_strA[BLAS_numChannels];
#pragma HLS ARRAY_PARTITION variable = l_strA dim = 1 complete
    hls::stream<BLAS_interface> l_strP;
    hls::stream<BLAS_interface> l_Apk;
#pragma HLS DATAFLOW

    xf::blas::loadMats<BLAS_dataType, BLAS_parEntries>(p_m / BLAS_numChannels, p_n, p_A0,
#if BLAS_numChannels > 1
                                                       p_A1,
#endif
#if BLAS_numChannels > 2
                                                       p_A2, p_A3,
#endif
#if BLAS_numChannels > 4
                                                       p_A4, p_A5, p_A6, p_A7,
#endif
#if BLAS_numChannels > 8
                                                       p_A8, p_A9, p_Aa, p_Ab, p_Ac, p_Ad, p_Ae, p_Af,
#endif
                                                       l_strA);
    xf::blas::mem2stream(p_n / BLAS_parEntries, p_pk, l_strP, p_m / BLAS_numChannels);
    xf::blas::gemv<BLAS_dataType, BLAS_parEntries, BLAS_numChannels>(p_m / BLAS_numChannels, p_n, l_strA, l_strP,
                                                                     l_Apk);
    xf::blas::stream2mem<BLAS_interface>(p_m / BLAS_parEntries, l_Apk, p_Apk);
}
