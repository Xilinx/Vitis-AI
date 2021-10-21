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

namespace xf {
namespace hpc {
namespace cg {

template <typename t_DataType, int t_ParEntries>
void proc_update_xr(typename xf::blas::WideType<t_DataType, t_ParEntries>::t_TypeInt* p_xk_in,
                    typename xf::blas::WideType<t_DataType, t_ParEntries>::t_TypeInt* p_xk_out,
                    typename xf::blas::WideType<t_DataType, t_ParEntries>::t_TypeInt* p_rk_in,
                    typename xf::blas::WideType<t_DataType, t_ParEntries>::t_TypeInt* p_rk_out,
                    typename xf::blas::WideType<t_DataType, t_ParEntries>::t_TypeInt* p_pk,
                    typename xf::blas::WideType<t_DataType, t_ParEntries>::t_TypeInt* p_Apk,
                    t_DataType p_alpha,
                    uint32_t p_size,
                    t_DataType& p_res) {
    typedef typename xf::blas::WideType<t_DataType, t_ParEntries>::t_TypeInt t_TypeInt;
#pragma HLS DATAFLOW
    hls::stream<t_TypeInt> l_rkStrIn, l_xkStrIn, l_pkStrIn, l_ApkStrIn;
    hls::stream<t_TypeInt> l_xkStrOut, l_rkStrOut, l_rkStrDup[2];
    xf::blas::mem2stream(p_size / t_ParEntries, p_xk_in, l_xkStrIn);
    xf::blas::mem2stream(p_size / t_ParEntries, p_rk_in, l_rkStrIn);
    xf::blas::mem2stream(p_size / t_ParEntries, p_pk, l_pkStrIn);
    xf::blas::mem2stream(p_size / t_ParEntries, p_Apk, l_ApkStrIn);
    xf::blas::axpy<t_DataType, CG_parEntries>(p_size, p_alpha, l_pkStrIn, l_xkStrIn, l_xkStrOut);
    xf::blas::axpy<t_DataType, CG_parEntries>(p_size, -p_alpha, l_ApkStrIn, l_rkStrIn, l_rkStrOut);
    xf::blas::duplicateStream<2>(p_size / t_ParEntries, l_rkStrOut, l_rkStrDup);
    nrm2s<t_DataType, xf::blas::mylog2(CG_parEntries)>(p_size, l_rkStrDup[0], p_res);
    xf::blas::stream2mem<t_TypeInt>(p_size / t_ParEntries, l_xkStrOut, p_xk_out);
    xf::blas::stream2mem<t_TypeInt>(p_size / t_ParEntries, l_rkStrDup[1], p_rk_out);
}

template <typename t_DataType, int t_ParEntries, int t_TkWidth = 8>
void update_xr(typename xf::blas::WideType<t_DataType, t_ParEntries>::t_TypeInt* p_xk_in,
               typename xf::blas::WideType<t_DataType, t_ParEntries>::t_TypeInt* p_xk_out,
               typename xf::blas::WideType<t_DataType, t_ParEntries>::t_TypeInt* p_rk_in,
               typename xf::blas::WideType<t_DataType, t_ParEntries>::t_TypeInt* p_rk_out,
               typename xf::blas::WideType<t_DataType, t_ParEntries>::t_TypeInt* p_pk,
               typename xf::blas::WideType<t_DataType, t_ParEntries>::t_TypeInt* p_Apk,
               hls::stream<ap_uint<t_TkWidth> >& p_tokenIn,
               hls::stream<ap_uint<t_TkWidth> >& p_tokenOut) {
    Token<t_DataType> l_token;
    StreamInstr<sizeof(l_token)> l_cs;
    l_token.read_decode(p_tokenIn, l_cs);

    while (!l_token.getExit()) {
        t_DataType l_alpha = l_token.getAlpha();
        uint32_t l_size = l_token.getVecSize();
        t_DataType l_res = 0;
        proc_update_xr<t_DataType, t_ParEntries>(p_xk_in, p_xk_out, p_rk_in, p_rk_out, p_pk, p_Apk, l_alpha, l_size,
                                                 l_res);
        l_token.setBeta(l_res / l_token.getRes());
        l_token.setRes(l_res);
        l_token.encode_write(p_tokenOut, l_cs);
        l_token.read_decode(p_tokenIn, l_cs);
    }
    l_token.encode_write(p_tokenOut, l_cs);
}
}
}
}
