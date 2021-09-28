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

/**
 * @file krnl_storeApk.cpp
 * @brief krnl_storeApk definition.
 *
 * This file is part of Vitis HPC Library.
 */

#include "krnl_storeApk.hpp"

void packMultApkStr(uint32_t p_size,
                    hls::stream<typename xf::blas::WideType<CG_dataType, 1>::t_TypeInt>& p_ApkStr,
                    hls::stream<CG_vecInterface>& p_pkcStr,
                    hls::stream<typename xf::blas::WideType<CG_dataType, CG_vecParEntries>::t_TypeInt>& p_ApkStrC,
                    hls::stream<typename xf::blas::WideType<CG_dataType, 1>::t_TypeInt>& p_dotStr) {
#pragma HLS DATAFLOW
    hls::stream<typename xf::blas::WideType<CG_dataType, CG_vecParEntries>::t_TypeInt> l_ApStr, l_ApStr0;
    xf::hpc::stream2wide<sizeof(CG_dataType) * 8, CG_vecParEntries>(p_size / CG_vecParEntries, p_ApkStr, l_ApStr);
    xf::hpc::duplicate(p_size / CG_vecParEntries, l_ApStr, l_ApStr0, p_ApkStrC);
    xf::blas::DotHelper<CG_dataType, xf::blas::mylog2(CG_vecParEntries)>::dot(p_size, 1, p_pkcStr, l_ApStr0, p_dotStr);
}

void proc_storeApk(uint32_t p_size,
                   hls::stream<typename xf::blas::WideType<CG_dataType, 1>::t_TypeInt>& p_ApkStr,
                   CG_vecInterface* p_pkc,
                   CG_vecInterface* p_Apk,
                   hls::stream<typename xf::blas::WideType<CG_dataType, 1>::t_TypeInt>& p_dotStr) {
    hls::stream<CG_vecInterface> l_pkcStr, l_ApkStr;
#pragma HLS DATAFLOW
    xf::blas::mem2stream(p_size / CG_vecParEntries, p_pkc, l_pkcStr);
    packMultApkStr(p_size, p_ApkStr, l_pkcStr, l_ApkStr, p_dotStr);
    xf::blas::stream2mem<CG_vecInterface>(p_size / CG_vecParEntries, l_ApkStr, p_Apk);
}

extern "C" void krnl_storeApk(hls::stream<typename xf::blas::WideType<CG_dataType, 1>::t_TypeInt>& p_ApkStr,
                              CG_interface* p_pkc,
                              CG_interface* p_Apk,
                              CG_tkStrType& p_tkInStr,
                              CG_tkStrType& p_tkOutStr) {
    POINTER(p_pkc, p_pkc)
    POINTER(p_Apk, p_Apk)
    AXIS(p_ApkStr)
    AXIS(p_tkInStr)
    AXIS(p_tkOutStr)
    SCALAR(return )

    xf::hpc::cg::Token<CG_dataType> l_token;
    xf::hpc::StreamInstr<sizeof(l_token)> l_cs;
    l_token.read_decode(p_tkInStr, l_cs);
    while (!l_token.getExit()) {
        uint32_t l_size = l_token.getVecSize();
        hls::stream<typename xf::blas::WideType<CG_dataType, 1>::t_TypeInt> l_dotStr;
        proc_storeApk(l_size, p_ApkStr, p_pkc, p_Apk, l_dotStr);
        xf::blas::WideType<CG_dataType, 1> l_r = l_dotStr.read();
        CG_dataType l_dot = l_r[0];
        l_token.setAlpha(l_token.getRZ() / l_dot);
        l_token.encode_write(p_tkOutStr, l_cs);
        l_token.read_decode(p_tkInStr, l_cs);
    }
    l_token.encode_write(p_tkOutStr, l_cs);
}
