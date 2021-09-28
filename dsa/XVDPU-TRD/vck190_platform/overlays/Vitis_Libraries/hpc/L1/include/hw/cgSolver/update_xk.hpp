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
#include "xf_blas.hpp"
#include "nrm2s.hpp"

#ifndef XF_HPC_CG_UPDATE_XK_HPP
#define XF_HPC_CG_UPDATE_XK_HPP
namespace xf {
namespace hpc {
namespace cg {

template <typename t_DataType, int t_ParEntries>
void proc_update_xk(hls::stream<typename xf::blas::WideType<t_DataType, t_ParEntries>::t_TypeInt>& p_xkStrIn,
                    hls::stream<typename xf::blas::WideType<t_DataType, t_ParEntries>::t_TypeInt>& p_xkStrOut,
                    hls::stream<typename xf::blas::WideType<t_DataType, t_ParEntries>::t_TypeInt>& p_pkStrIn,
                    t_DataType p_alpha,
                    uint32_t p_size) {
#pragma HLS DATAFLOW
    xf::blas::axpy<t_DataType, t_ParEntries>(p_size, p_alpha, p_pkStrIn, p_xkStrIn, p_xkStrOut);
}

template <typename t_DataType, int t_ParEntries>
void proc_update_xk(typename xf::blas::WideType<t_DataType, t_ParEntries>::t_TypeInt* p_xk_in,
                    typename xf::blas::WideType<t_DataType, t_ParEntries>::t_TypeInt* p_xk_out,
                    typename xf::blas::WideType<t_DataType, t_ParEntries>::t_TypeInt* p_pk,
                    t_DataType p_alpha,
                    uint32_t p_size) {
    typedef typename xf::blas::WideType<t_DataType, t_ParEntries>::t_TypeInt t_TypeInt;
#pragma HLS DATAFLOW
    hls::stream<t_TypeInt> l_xkStrIn, l_pkStrIn;
    hls::stream<t_TypeInt> l_xkStrOut;
    xf::blas::mem2stream(p_size / t_ParEntries, p_xk_in, l_xkStrIn);
    xf::blas::mem2stream(p_size / t_ParEntries, p_pk, l_pkStrIn);
    proc_update_xk<t_DataType, t_ParEntries>(l_xkStrIn, l_xkStrOut, l_pkStrIn, p_alpha, p_size);
    xf::blas::stream2mem<t_TypeInt>(p_size / t_ParEntries, l_xkStrOut, p_xk_out);
}

template <typename t_DataType, int t_ParEntries, int t_TkWidth = 8>
void update_xk(typename xf::blas::WideType<t_DataType, t_ParEntries>::t_TypeInt* p_xk_in,
               typename xf::blas::WideType<t_DataType, t_ParEntries>::t_TypeInt* p_xk_out,
               typename xf::blas::WideType<t_DataType, t_ParEntries>::t_TypeInt* p_pk,
               hls::stream<ap_uint<t_TkWidth> >& p_tokenIn) {
    Token<t_DataType> l_token;
    StreamInstr<sizeof(l_token)> l_cs;
    l_token.read_decode(p_tokenIn, l_cs);

    while (!l_token.getExit()) {
        t_DataType l_alpha = l_token.getAlpha();
        uint32_t l_size = l_token.getVecSize();
        proc_update_xk<t_DataType, t_ParEntries>(p_xk_in, p_xk_out, p_pk, l_alpha, l_size);
        l_token.read_decode(p_tokenIn, l_cs);
    }
}
}
}
}
#endif
