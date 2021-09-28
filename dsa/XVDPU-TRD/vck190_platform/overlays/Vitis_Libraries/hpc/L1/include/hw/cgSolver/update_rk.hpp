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
#include "streamOps.hpp"
#include "nrm2s.hpp"

#ifndef XF_HPC_CG_UPDATE_RK_HPP
#define XF_HPC_CG_UPDATE_RK_HPP

namespace xf {
namespace hpc {
namespace cg {

template <typename t_DataType, int t_ParEntries>
void proc_update_rk(hls::stream<typename xf::blas::WideType<t_DataType, t_ParEntries>::t_TypeInt>& p_rkStrIn,
                    hls::stream<typename xf::blas::WideType<t_DataType, t_ParEntries>::t_TypeInt>& p_rkStrOut,
                    hls::stream<typename xf::blas::WideType<t_DataType, t_ParEntries>::t_TypeInt>& p_ApkStrIn,
                    hls::stream<typename xf::blas::WideType<t_DataType, 1>::t_TypeInt>& p_res,
                    t_DataType p_alpha,
                    uint32_t p_size) {
#pragma HLS DATAFLOW
    typedef typename xf::blas::WideType<t_DataType, t_ParEntries>::t_TypeInt t_TypeInt;
    hls::stream<t_TypeInt> l_rkStrOut, l_rkStrDup;
    xf::blas::axpy<t_DataType, CG_parEntries>(p_size, p_alpha, p_ApkStrIn, p_rkStrIn, l_rkStrOut);
    duplicate(p_size / t_ParEntries, l_rkStrOut, p_rkStrOut, l_rkStrDup);
    nrm2s<t_DataType, xf::blas::mylog2(CG_parEntries)>(p_size, l_rkStrDup, p_res);
}

template <typename t_DataType, int t_ParEntries>
void proc_update_rk(typename xf::blas::WideType<t_DataType, t_ParEntries>::t_TypeInt* p_rk_in,
                    typename xf::blas::WideType<t_DataType, t_ParEntries>::t_TypeInt* p_rk_out,
                    typename xf::blas::WideType<t_DataType, t_ParEntries>::t_TypeInt* p_Apk,
                    hls::stream<typename xf::blas::WideType<t_DataType, 1>::t_TypeInt>& p_res,
                    t_DataType p_alpha,
                    uint32_t p_size) {
    typedef typename xf::blas::WideType<t_DataType, t_ParEntries>::t_TypeInt t_TypeInt;
#pragma HLS DATAFLOW
    hls::stream<t_TypeInt> l_rkStrIn, l_ApkStrIn, l_rkStrOut;
    xf::blas::mem2stream(p_size / t_ParEntries, p_rk_in, l_rkStrIn);
    xf::blas::mem2stream(p_size / t_ParEntries, p_Apk, l_ApkStrIn);
    proc_update_rk<t_DataType, t_ParEntries>(l_rkStrIn, l_rkStrOut, l_ApkStrIn, p_res, p_alpha, p_size);
    xf::blas::stream2mem<t_TypeInt>(p_size / t_ParEntries, l_rkStrOut, p_rk_out);
}

template <typename t_DataType, int t_ParEntries, int t_TkWidth = 8>
void update_rk(typename xf::blas::WideType<t_DataType, t_ParEntries>::t_TypeInt* p_rk_in,
               typename xf::blas::WideType<t_DataType, t_ParEntries>::t_TypeInt* p_rk_out,
               typename xf::blas::WideType<t_DataType, t_ParEntries>::t_TypeInt* p_Apk,
               hls::stream<ap_uint<t_TkWidth> >& p_tokenIn,
               hls::stream<ap_uint<t_TkWidth> >& p_tokenOut) {
    Token<t_DataType> l_token;
    StreamInstr<sizeof(l_token)> l_cs;
    l_token.read_decode(p_tokenIn, l_cs);

    while (!l_token.getExit()) {
        t_DataType l_alpha = -l_token.getAlpha();
        uint32_t l_size = l_token.getVecSize();
        hls::stream<typename xf::blas::WideType<t_DataType, 1>::t_TypeInt> l_resStr;
        proc_update_rk<t_DataType, t_ParEntries>(p_rk_in, p_rk_out, p_Apk, l_resStr, l_alpha, l_size);
        t_DataType l_res = ((xf::blas::WideType<t_DataType, 1>)l_resStr.read())[0];
        l_token.setBeta(l_res / l_token.getRZ());
        l_token.setRes(l_res);
        l_token.setRZ(l_res);
        l_token.encode_write(p_tokenOut, l_cs);
        l_token.read_decode(p_tokenIn, l_cs);
    }
    l_token.encode_write(p_tokenOut, l_cs);
}
}
}
}
#endif
