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
#include "signal.hpp"

#ifndef XF_HPC_CG_UPDATE_PK_HPP
#define XF_HPC_CG_UPDATE_PK_HPP
namespace xf {
namespace hpc {
namespace cg {

template <typename t_DataType, int t_ParEntries>
void proc_update_pk(hls::stream<typename xf::blas::WideType<t_DataType, t_ParEntries>::t_TypeInt>& p_pkStrIn,
                    hls::stream<typename xf::blas::WideType<t_DataType, t_ParEntries>::t_TypeInt>& p_pkStrOut,
                    hls::stream<typename xf::blas::WideType<t_DataType, t_ParEntries>::t_TypeInt>& p_zkStrIn,
                    t_DataType p_beta,
                    uint32_t p_size) {
#pragma HLS DATAFLOW
    xf::blas::axpy<t_DataType, t_ParEntries>(p_size, p_beta, p_pkStrIn, p_zkStrIn, p_pkStrOut);
}

template <typename t_DataType, int t_ParEntries>
void proc_update_pk(typename xf::blas::WideType<t_DataType, t_ParEntries>::t_TypeInt* p_rk,
                    typename xf::blas::WideType<t_DataType, t_ParEntries>::t_TypeInt* p_pk_in,
                    typename xf::blas::WideType<t_DataType, t_ParEntries>::t_TypeInt* p_pk_out,
                    t_DataType p_beta,
                    uint32_t p_size) {
    typedef typename xf::blas::WideType<t_DataType, t_ParEntries>::t_TypeInt t_TypeInt;
#pragma HLS DATAFLOW
    hls::stream<t_TypeInt> l_rkStrIn, l_pkStrIn;
    hls::stream<t_TypeInt> l_pkStrOut;
    xf::blas::mem2stream(p_size / t_ParEntries, p_rk, l_rkStrIn);
    xf::blas::mem2stream(p_size / t_ParEntries, p_pk_in, l_pkStrIn);
    proc_update_pk<t_DataType, t_ParEntries>(l_pkStrIn, l_pkStrOut, l_rkStrIn, p_beta, p_size);
    xf::blas::stream2mem<t_TypeInt>(p_size / t_ParEntries, l_pkStrOut, p_pk_out);
}

template <typename t_DataType, int t_ParEntries, int t_TkWidth = 8>
void update_pk(typename xf::blas::WideType<t_DataType, t_ParEntries>::t_TypeInt* p_rk,
               typename xf::blas::WideType<t_DataType, t_ParEntries>::t_TypeInt* p_pk_in,
               typename xf::blas::WideType<t_DataType, t_ParEntries>::t_TypeInt* p_pk_out,
               hls::stream<ap_uint<t_TkWidth> >& p_tokenIn,
               hls::stream<ap_uint<t_TkWidth> >& p_tokenOut) {
    Token<t_DataType> l_token;
    StreamInstr<sizeof(l_token)> l_cs;

    l_token.read_decode(p_tokenIn, l_cs);

    while (!l_token.getExit()) {
        t_DataType l_beta = l_token.getBeta();
        uint32_t l_size = l_token.getVecSize();

        proc_update_pk<t_DataType, t_ParEntries>(p_rk, p_pk_in, p_pk_out, l_beta, l_size);
        ap_wait();
        l_token.encode_write(p_tokenOut, l_cs);
        l_token.read_decode(p_tokenIn, l_cs);
    }
}
}
}
}
#endif
