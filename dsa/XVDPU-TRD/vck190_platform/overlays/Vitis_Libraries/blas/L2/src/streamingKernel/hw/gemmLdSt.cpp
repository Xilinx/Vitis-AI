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
#include "blasKernels.hpp"

extern "C" void gemmKernel(ap_uint<BLAS_ddrMemBits>* p_rdPtr,
                           ap_uint<BLAS_ddrMemBits>* p_wrPtr,
                           hls::stream<ap_uint<16> >& out,
                           hls::stream<ap_uint<32> >& in) {
#pragma HLS INTERFACE m_axi port = p_rdPtr offset = slave bundle = gmem latency = 125
#pragma HLS INTERFACE s_axilite port = p_rdPtr bundle = control
#pragma HLS INTERFACE m_axi port = p_wrPtr offset = slave bundle = gmem latency = 125
#pragma HLS INTERFACE s_axilite port = p_wrPtr bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

#pragma HLS INTERFACE axis port = in
#pragma HLS INTERFACE axis port = out

#pragma HLS DATAFLOW

    hls::stream<ap_uint<BLAS_ddrMemBits> > l_aStr;
#pragma HLS stream variable = l_aStr depth = 16
    hls::stream<ap_uint<BLAS_ddrMemBits> > l_bStr;
#pragma HLS stream variable = l_bStr depth = 16
    hls::stream<ap_uint<BLAS_ddrMemBits> > l_xStr;
#pragma HLS stream variable = l_xStr depth = 16

    xf::blas::gemmLoad<BLAS_maxNumInstrs, BLAS_memWordsPerInstr, BLAS_parEntries, BLAS_mParWords, BLAS_kParWords,
                       BLAS_nParWords, BLAS_ddrMemBits>(p_rdPtr, l_aStr, l_bStr, l_xStr, out);

    xf::blas::gemmStoreABX<BLAS_resOffsetBytes, BLAS_memWordsPerInstr, BLAS_parEntries, BLAS_mParWords, BLAS_kParWords,
                           BLAS_nParWords, BLAS_ddrMemBits>(l_aStr, l_bStr, l_xStr, in, p_wrPtr);
}
