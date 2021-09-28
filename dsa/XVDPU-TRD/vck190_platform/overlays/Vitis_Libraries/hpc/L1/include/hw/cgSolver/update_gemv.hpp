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
#include "streamOps.hpp"
#include "xf_blas.hpp"

#ifndef XF_HPC_CG_UPDATE_GEMV_HPP
#define XF_HPC_CG_UPDATE_GEMV_HPP

namespace xf {
namespace hpc {
namespace cg {

template <typename t_DataType, int t_ParEntries, int t_VecParEntries, int t_NumPorts = 16>
void update_gemv(uint32_t p_size,
                 hls::stream<typename xf::blas::WideType<t_DataType, 1>::t_TypeInt>& p_dot,
                 hls::stream<typename xf::blas::WideType<t_DataType, t_ParEntries>::t_TypeInt> p_A[t_NumPorts],
                 hls::stream<typename xf::blas::WideType<t_DataType, t_ParEntries>::t_TypeInt>& p_pk0,
                 hls::stream<typename xf::blas::WideType<t_DataType, t_VecParEntries>::t_TypeInt>& p_pk1,
                 hls::stream<typename xf::blas::WideType<t_DataType, t_VecParEntries>::t_TypeInt>& p_Apk,
                 uint32_t p_m) {
    constexpr int t_LogParEntries = xf::blas::mylog2(t_ParEntries);
    typedef typename xf::blas::WideType<t_DataType, t_ParEntries>::t_TypeInt t_TypeInt;

    hls::stream<typename xf::blas::WideType<t_DataType, t_ParEntries>::t_TypeInt> l_pk[t_NumPorts];
#pragma HLS stream variable = l_pk depth = 32

    hls::stream<typename xf::blas::WideType<t_DataType, 1>::t_TypeInt> l_strC, l_strR[t_NumPorts];
#pragma HLS stream variable = l_strR depth = 32
    hls::stream<typename xf::blas::WideType<t_DataType, t_VecParEntries>::t_TypeInt> l_strAp, l_strAp0;
#pragma HLS DATAFLOW

    streamFwd<t_NumPorts>(p_size * p_size / t_NumPorts / t_ParEntries, p_pk0, l_pk);
    for (int i = 0; i < t_NumPorts; i++)
#pragma HLS UNROLL
        xf::blas::gemv<t_DataType, t_LogParEntries>(p_m, p_size, p_A[i], l_pk[i], l_strR[i]);

    collectStream<t_DataType, t_NumPorts, 1>(p_size / t_NumPorts, l_strR, l_strC);
    stream2wide<sizeof(t_DataType) * 8, t_VecParEntries>(p_size / t_VecParEntries, l_strC, l_strAp);
    duplicate(p_size / t_VecParEntries, l_strAp, l_strAp0, p_Apk);
    xf::blas::DotHelper<t_DataType, xf::blas::mylog2(t_VecParEntries)>::dot(p_size, 1, p_pk1, l_strAp0, p_dot);
}
}
}
}
#endif
