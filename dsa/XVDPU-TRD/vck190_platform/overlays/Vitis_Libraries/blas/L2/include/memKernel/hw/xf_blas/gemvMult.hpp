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
#include "streamOps.hpp"
#ifndef XF_BLAS_KERNEL_GEMV_HPP
#define XF_BLAS_KERNEL_GEMV_HPP

namespace xf {
namespace blas {

template <typename t_DataType, int t_ParEntries, int t_NumPorts = 16>
void gemv(uint32_t p_m,
          uint32_t p_n,
          hls::stream<typename xf::blas::WideType<t_DataType, t_ParEntries>::t_TypeInt> p_A[t_NumPorts],
          hls::stream<typename xf::blas::WideType<t_DataType, t_ParEntries>::t_TypeInt>& p_pk,
          hls::stream<typename xf::blas::WideType<t_DataType, t_ParEntries>::t_TypeInt>& p_Apk) {
#pragma HLS DATAFLOW
    typedef typename xf::blas::WideType<t_DataType, t_ParEntries>::t_TypeInt t_TypeInt;
    constexpr int t_LogParEntries = xf::blas::mylog2(t_ParEntries);

    hls::stream<typename xf::blas::WideType<t_DataType, t_ParEntries>::t_TypeInt> l_pk[t_NumPorts];
#pragma HLS stream variable = l_pk depth = 32

    hls::stream<typename xf::blas::WideType<t_DataType, 1>::t_TypeInt> l_strR[t_NumPorts], l_strC;
#pragma HLS stream variable = l_strR depth = 32

    // xf::blas::duplicateStream<t_NumPorts>(p_m * p_n / t_ParEntries, p_pk, l_pk);
    hpc::streamFwd<t_NumPorts>(p_m * p_n / t_ParEntries, p_pk, l_pk);
    for (int i = 0; i < t_NumPorts; i++)
#pragma HLS UNROLL
        xf::blas::gemv<t_DataType, t_LogParEntries>(p_m, p_n, p_A[i], l_pk[i], l_strR[i]);

    hpc::collectStream<t_DataType, t_NumPorts>(p_m, l_strR, l_strC);
    hpc::stream2wide<sizeof(t_DataType) * 8, t_ParEntries>(p_m * t_NumPorts / t_ParEntries, l_strC, p_Apk);
}
}
}
#endif
