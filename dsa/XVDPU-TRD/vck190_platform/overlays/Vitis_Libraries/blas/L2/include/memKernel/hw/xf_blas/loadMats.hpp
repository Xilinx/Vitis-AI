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
#include "xf_blas.hpp"

#ifndef XF_BLAS_LOADMATS_HPP
#define XF_BLAS_LOADMATS_HPP

namespace xf {
namespace blas {

template <typename t_DataType, int t_ParEntries>
void loadMats(uint32_t p_m,
              uint32_t p_n,
              typename xf::blas::WideType<t_DataType, t_ParEntries>::t_TypeInt* p_A0,
              hls::stream<typename xf::blas::WideType<t_DataType, t_ParEntries>::t_TypeInt> p_strA[1]) {
    constexpr int t_LogParEntries = xf::blas::mylog2(t_ParEntries);
#pragma HLS DATAFLOW
    xf::blas::mem2stream(p_m * p_n >> t_LogParEntries, p_A0, p_strA[0x0]);
}

template <typename t_DataType, int t_ParEntries>
void loadMats(uint32_t p_m,
              uint32_t p_n,
              typename xf::blas::WideType<t_DataType, t_ParEntries>::t_TypeInt* p_A0,
              typename xf::blas::WideType<t_DataType, t_ParEntries>::t_TypeInt* p_A1,
              hls::stream<typename xf::blas::WideType<t_DataType, t_ParEntries>::t_TypeInt> p_strA[2]) {
    constexpr int t_LogParEntries = xf::blas::mylog2(t_ParEntries);
#pragma HLS DATAFLOW
    xf::blas::mem2stream(p_m * p_n >> t_LogParEntries, p_A0, p_strA[0x0]);
    xf::blas::mem2stream(p_m * p_n >> t_LogParEntries, p_A1, p_strA[0x1]);
}

template <typename t_DataType, int t_ParEntries>
void loadMats(uint32_t p_m,
              uint32_t p_n,
              typename xf::blas::WideType<t_DataType, t_ParEntries>::t_TypeInt* p_A0,
              typename xf::blas::WideType<t_DataType, t_ParEntries>::t_TypeInt* p_A1,
              typename xf::blas::WideType<t_DataType, t_ParEntries>::t_TypeInt* p_A2,
              typename xf::blas::WideType<t_DataType, t_ParEntries>::t_TypeInt* p_A3,
              hls::stream<typename xf::blas::WideType<t_DataType, t_ParEntries>::t_TypeInt> p_strA[4]) {
    constexpr int t_LogParEntries = xf::blas::mylog2(t_ParEntries);
#pragma HLS DATAFLOW
    xf::blas::mem2stream(p_m * p_n >> t_LogParEntries, p_A0, p_strA[0x0]);
    xf::blas::mem2stream(p_m * p_n >> t_LogParEntries, p_A1, p_strA[0x1]);
    xf::blas::mem2stream(p_m * p_n >> t_LogParEntries, p_A2, p_strA[0x2]);
    xf::blas::mem2stream(p_m * p_n >> t_LogParEntries, p_A3, p_strA[0x3]);
}

template <typename t_DataType, int t_ParEntries>
void loadMats(uint32_t p_m,
              uint32_t p_n,
              typename xf::blas::WideType<t_DataType, t_ParEntries>::t_TypeInt* p_A0,
              typename xf::blas::WideType<t_DataType, t_ParEntries>::t_TypeInt* p_A1,
              typename xf::blas::WideType<t_DataType, t_ParEntries>::t_TypeInt* p_A2,
              typename xf::blas::WideType<t_DataType, t_ParEntries>::t_TypeInt* p_A3,
              typename xf::blas::WideType<t_DataType, t_ParEntries>::t_TypeInt* p_A4,
              typename xf::blas::WideType<t_DataType, t_ParEntries>::t_TypeInt* p_A5,
              typename xf::blas::WideType<t_DataType, t_ParEntries>::t_TypeInt* p_A6,
              typename xf::blas::WideType<t_DataType, t_ParEntries>::t_TypeInt* p_A7,
              hls::stream<typename xf::blas::WideType<t_DataType, t_ParEntries>::t_TypeInt> p_strA[8]) {
    constexpr int t_LogParEntries = xf::blas::mylog2(t_ParEntries);
#pragma HLS DATAFLOW
    xf::blas::mem2stream(p_m * p_n >> t_LogParEntries, p_A0, p_strA[0x0]);
    xf::blas::mem2stream(p_m * p_n >> t_LogParEntries, p_A1, p_strA[0x1]);
    xf::blas::mem2stream(p_m * p_n >> t_LogParEntries, p_A2, p_strA[0x2]);
    xf::blas::mem2stream(p_m * p_n >> t_LogParEntries, p_A3, p_strA[0x3]);
    xf::blas::mem2stream(p_m * p_n >> t_LogParEntries, p_A4, p_strA[0x4]);
    xf::blas::mem2stream(p_m * p_n >> t_LogParEntries, p_A5, p_strA[0x5]);
    xf::blas::mem2stream(p_m * p_n >> t_LogParEntries, p_A6, p_strA[0x6]);
    xf::blas::mem2stream(p_m * p_n >> t_LogParEntries, p_A7, p_strA[0x7]);
}

template <typename t_DataType, int t_ParEntries>
void loadMats(uint32_t p_m,
              uint32_t p_n,
              typename xf::blas::WideType<t_DataType, t_ParEntries>::t_TypeInt* p_A0,
              typename xf::blas::WideType<t_DataType, t_ParEntries>::t_TypeInt* p_A1,
              typename xf::blas::WideType<t_DataType, t_ParEntries>::t_TypeInt* p_A2,
              typename xf::blas::WideType<t_DataType, t_ParEntries>::t_TypeInt* p_A3,
              typename xf::blas::WideType<t_DataType, t_ParEntries>::t_TypeInt* p_A4,
              typename xf::blas::WideType<t_DataType, t_ParEntries>::t_TypeInt* p_A5,
              typename xf::blas::WideType<t_DataType, t_ParEntries>::t_TypeInt* p_A6,
              typename xf::blas::WideType<t_DataType, t_ParEntries>::t_TypeInt* p_A7,
              typename xf::blas::WideType<t_DataType, t_ParEntries>::t_TypeInt* p_A8,
              typename xf::blas::WideType<t_DataType, t_ParEntries>::t_TypeInt* p_A9,
              typename xf::blas::WideType<t_DataType, t_ParEntries>::t_TypeInt* p_Aa,
              typename xf::blas::WideType<t_DataType, t_ParEntries>::t_TypeInt* p_Ab,
              typename xf::blas::WideType<t_DataType, t_ParEntries>::t_TypeInt* p_Ac,
              typename xf::blas::WideType<t_DataType, t_ParEntries>::t_TypeInt* p_Ad,
              typename xf::blas::WideType<t_DataType, t_ParEntries>::t_TypeInt* p_Ae,
              typename xf::blas::WideType<t_DataType, t_ParEntries>::t_TypeInt* p_Af,
              hls::stream<typename xf::blas::WideType<t_DataType, t_ParEntries>::t_TypeInt> p_strA[16]) {
    constexpr int t_LogParEntries = xf::blas::mylog2(t_ParEntries);
#pragma HLS DATAFLOW
    xf::blas::mem2stream(p_m * p_n >> t_LogParEntries, p_A0, p_strA[0x0]);
    xf::blas::mem2stream(p_m * p_n >> t_LogParEntries, p_A1, p_strA[0x1]);
    xf::blas::mem2stream(p_m * p_n >> t_LogParEntries, p_A2, p_strA[0x2]);
    xf::blas::mem2stream(p_m * p_n >> t_LogParEntries, p_A3, p_strA[0x3]);
    xf::blas::mem2stream(p_m * p_n >> t_LogParEntries, p_A4, p_strA[0x4]);
    xf::blas::mem2stream(p_m * p_n >> t_LogParEntries, p_A5, p_strA[0x5]);
    xf::blas::mem2stream(p_m * p_n >> t_LogParEntries, p_A6, p_strA[0x6]);
    xf::blas::mem2stream(p_m * p_n >> t_LogParEntries, p_A7, p_strA[0x7]);
    xf::blas::mem2stream(p_m * p_n >> t_LogParEntries, p_A8, p_strA[0x8]);
    xf::blas::mem2stream(p_m * p_n >> t_LogParEntries, p_A9, p_strA[0x9]);
    xf::blas::mem2stream(p_m * p_n >> t_LogParEntries, p_Aa, p_strA[0xa]);
    xf::blas::mem2stream(p_m * p_n >> t_LogParEntries, p_Ab, p_strA[0xb]);
    xf::blas::mem2stream(p_m * p_n >> t_LogParEntries, p_Ac, p_strA[0xc]);
    xf::blas::mem2stream(p_m * p_n >> t_LogParEntries, p_Ad, p_strA[0xd]);
    xf::blas::mem2stream(p_m * p_n >> t_LogParEntries, p_Ae, p_strA[0xe]);
    xf::blas::mem2stream(p_m * p_n >> t_LogParEntries, p_Af, p_strA[0xf]);
}
}
}
#endif
