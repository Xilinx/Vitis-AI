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
#include "streamOps.hpp"
#include "xf_blas.hpp"
#include "nrm2s.hpp"
#include "timer.hpp"

#ifndef XF_HPC_CG_UPDATE_RK_HPP
#define XF_HPC_CG_UPDATE_RK_HPP

namespace xf {
namespace hpc {
namespace cg {

template <typename t_DataType, int t_ParEntries>
void proc_update_rk(hls::stream<typename xf::blas::WideType<t_DataType, t_ParEntries>::t_TypeInt>& p_rkStrIn,
                    hls::stream<typename xf::blas::WideType<t_DataType, t_ParEntries>::t_TypeInt>& p_rkStrOut,
                    hls::stream<typename xf::blas::WideType<t_DataType, t_ParEntries>::t_TypeInt>& p_ApkStrIn,
                    hls::stream<typename xf::blas::WideType<t_DataType, t_ParEntries>::t_TypeInt>& p_zkStr,
                    hls::stream<typename xf::blas::WideType<t_DataType, t_ParEntries>::t_TypeInt>& p_jkStrIn,
                    hls::stream<typename xf::blas::WideType<t_DataType, 1>::t_TypeInt>& p_res,
                    hls::stream<typename xf::blas::WideType<t_DataType, 1>::t_TypeInt>& p_rz,
                    t_DataType p_alpha,
                    uint32_t p_size) {
    typedef typename xf::blas::WideType<t_DataType, t_ParEntries>::t_TypeInt t_TypeInt;
    constexpr int t_LogParEntries = xf::blas::mylog2(t_ParEntries);
    hls::stream<t_TypeInt> l_zkStrOut, l_rkStrOut, l_zkStrDup, l_rkStrDup[4];
#pragma HLS STREAM variable = l_rkStrDup[2] depth = 32

#pragma HLS DATAFLOW
    xf::blas::axpy<t_DataType, t_ParEntries>(p_size, -p_alpha, p_ApkStrIn, p_rkStrIn, l_rkStrOut);
    duplicate(p_size / t_ParEntries, l_rkStrOut, p_rkStrOut, l_rkStrDup[3]);
    xf::blas::duplicateStream<3>(p_size / t_ParEntries, l_rkStrDup[3], l_rkStrDup);
    xf::blas::mul<t_DataType, t_ParEntries>(p_size, l_rkStrDup[0], p_jkStrIn, l_zkStrOut);
    duplicate(p_size / t_ParEntries, l_zkStrOut, p_zkStr, l_zkStrDup);
    nrm2s<t_DataType, t_LogParEntries>(p_size, l_rkStrDup[1], p_res);
    xf::blas::DotHelper<t_DataType, t_LogParEntries>::dot(p_size, 1, l_rkStrDup[2], l_zkStrDup, p_rz);
}

template <typename t_DataType, int t_ParEntries>
void proc_update_rk(typename xf::blas::WideType<t_DataType, t_ParEntries>::t_TypeInt* p_rk_in,
                    typename xf::blas::WideType<t_DataType, t_ParEntries>::t_TypeInt* p_rk_out,
                    typename xf::blas::WideType<t_DataType, t_ParEntries>::t_TypeInt* p_zk,
                    typename xf::blas::WideType<t_DataType, t_ParEntries>::t_TypeInt* p_jacobi,
                    typename xf::blas::WideType<t_DataType, t_ParEntries>::t_TypeInt* p_Apk,
                    hls::stream<typename xf::blas::WideType<t_DataType, 1>::t_TypeInt>& p_res,
                    hls::stream<typename xf::blas::WideType<t_DataType, 1>::t_TypeInt>& p_rz,
                    t_DataType p_alpha,
                    uint32_t p_size) {
    typedef typename xf::blas::WideType<t_DataType, t_ParEntries>::t_TypeInt t_TypeInt;
#pragma HLS DATAFLOW
    constexpr int t_LogParEntries = xf::blas::mylog2(t_ParEntries);
    const int l_size = p_size / t_ParEntries;

    hls::stream<t_TypeInt> l_rkStrIn, l_jkStrIn, l_ApkStrIn;
    hls::stream<t_TypeInt> l_zkStrOut, l_rkStrOut;

    xf::blas::mem2stream(l_size, p_jacobi, l_jkStrIn);
    xf::blas::mem2stream(l_size, p_rk_in, l_rkStrIn);
    xf::blas::mem2stream(l_size, p_Apk, l_ApkStrIn);

    proc_update_rk<t_DataType, t_ParEntries>(l_rkStrIn, l_rkStrOut, l_ApkStrIn, l_zkStrOut, l_jkStrIn, p_res, p_rz,
                                             p_alpha, p_size);
    xf::blas::stream2mem<t_TypeInt>(l_size, l_zkStrOut, p_zk);
    xf::blas::stream2mem<t_TypeInt>(l_size, l_rkStrOut, p_rk_out);
}

template <typename t_DataType, int t_ParEntries, int t_TkWidth = 8>
void update_rk(typename xf::blas::WideType<t_DataType, t_ParEntries>::t_TypeInt* p_rk_in,
               typename xf::blas::WideType<t_DataType, t_ParEntries>::t_TypeInt* p_rk_out,
               typename xf::blas::WideType<t_DataType, t_ParEntries>::t_TypeInt* p_zk,
               typename xf::blas::WideType<t_DataType, t_ParEntries>::t_TypeInt* p_jacobi,
               typename xf::blas::WideType<t_DataType, t_ParEntries>::t_TypeInt* p_Apk,
               hls::stream<ap_uint<t_TkWidth> >& p_tokenIn,
               hls::stream<ap_uint<t_TkWidth> >& p_tokenOut) {
    Token<t_DataType> l_token;
    StreamInstr<sizeof(l_token)> l_cs;
    l_token.read_decode(p_tokenIn, l_cs);

    while (!l_token.getExit()) {
        t_DataType l_alpha = l_token.getAlpha();
        uint32_t l_size = l_token.getVecSize();
        hls::stream<typename xf::blas::WideType<t_DataType, 1>::t_TypeInt> l_resStr, l_rzStr;
        proc_update_rk<t_DataType, t_ParEntries>(p_rk_in, p_rk_out, p_zk, p_jacobi, p_Apk, l_resStr, l_rzStr, l_alpha,
                                                 l_size);
        xf::blas::WideType<t_DataType, 1> l_res = l_resStr.read();
        xf::blas::WideType<t_DataType, 1> l_rz = l_rzStr.read();

        l_token.setBeta(l_rz[0] / l_token.getRZ());
        l_token.setRes(l_res[0]);
        l_token.setRZ(l_rz[0]);
        l_token.encode_write(p_tokenOut, l_cs);
        l_token.read_decode(p_tokenIn, l_cs);
    }
    l_token.encode_write(p_tokenOut, l_cs);
}
}
}
}
#endif
