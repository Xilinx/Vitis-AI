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
#ifndef XF_BLAS_GEMM_MUL_HPP
#define XF_BLAS_GEMM_MUL_HPP
#include "blasInstr.hpp"
#include "gemmKernel.hpp"
namespace xf {

namespace blas {

template <typename t_DataType,
          unsigned int t_ParEntries,
          unsigned int t_KparWords,
          typename t_WideType = WideType<t_DataType, t_ParEntries> >
void gemmMuls(hls::stream<t_DataType>& p_aIn,
              hls::stream<ap_uint<2> >& p_tagIn,
              hls::stream<typename t_WideType::t_TypeInt>& p_bIn,
              hls::stream<typename t_WideType::t_TypeInt>& p_bOut,
              hls::stream<typename t_WideType::t_TypeInt>& p_out) {
#pragma HLS INTERFACE ap_ctrl_none port = return
    SystolicArray<t_DataType, t_KparWords * t_ParEntries, t_ParEntries> sa;
    sa.muls(p_aIn, p_tagIn, p_bIn, p_bOut, p_out);
}
}
}
#endif
