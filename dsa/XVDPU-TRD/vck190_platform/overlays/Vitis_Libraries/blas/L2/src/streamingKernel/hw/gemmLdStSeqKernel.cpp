/**********
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
 * **********/

/**
 * @file gemmLdStSeqKernel.cpp
 * @brief gemm sequential load store kernel
 *
 * This file is part of Vitis BLAS Library
 */

#include "blasKernels.hpp"

extern "C" void gemmLdStSeqKernel(ap_uint<BLAS_ddrMemBits>* p_rdPtr, ap_uint<BLAS_ddrMemBits>* p_wrPtr) {
#pragma HLS INTERFACE m_axi port = p_rdPtr offset = slave bundle = gmem latency = 125
#pragma HLS INTERFACE m_axi port = p_wrPtr offset = slave bundle = gmem latency = 125
#pragma HLS INTERFACE s_axilite port = p_rdPtr bundle = control
#pragma HLS INTERFACE s_axilite port = p_wrPtr bundle = control

#pragma HLS INTERFACE s_axilite port = return bundle = control

#pragma HLS DATAFLOW

    hls::stream<ap_uint<BLAS_ddrMemBits> > l_aStr;
#pragma HLS stream variable = l_aStr depth = 4096
    hls::stream<ap_uint<BLAS_ddrMemBits> > l_bStr;
#pragma HLS stream variable = l_bStr depth = 4096
    hls::stream<ap_uint<BLAS_ddrMemBits> > l_xStr;
#pragma HLS stream variable = l_xStr depth = 4096
    hls::stream<ap_uint<16> > l_opCodeStr;
    hls::stream<ap_uint<32> > l_resStr;

    xf::blas::gemmLoad<BLAS_maxNumInstrs, BLAS_memWordsPerInstr, BLAS_parEntries, BLAS_mParWords, BLAS_kParWords,
                       BLAS_nParWords, BLAS_ddrMemBits>(p_rdPtr, l_aStr, l_bStr, l_xStr, l_opCodeStr);

    xf::blas::gemmLdStTimer(l_opCodeStr, l_resStr);

    xf::blas::gemmStoreABX<BLAS_resOffsetBytes, BLAS_memWordsPerInstr, BLAS_parEntries, BLAS_mParWords, BLAS_kParWords,
                           BLAS_nParWords, BLAS_ddrMemBits>(l_aStr, l_bStr, l_xStr, l_resStr, p_wrPtr);
}
