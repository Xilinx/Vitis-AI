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

extern "C" void gemmLdStKernel(ap_uint<BLAS_ddrMemBits>* p_rdPtr,
                               ap_uint<BLAS_ddrMemBits>* p_wrPtr,
                               hls::stream<ap_uint<BLAS_ddrMemBits> >& out0,
                               hls::stream<ap_uint<BLAS_ddrMemBits> >& out1,
                               hls::stream<ap_uint<BLAS_ddrMemBits> >& out2,
                               hls::stream<ap_uint<16> >& out3,
                               hls::stream<ap_uint<BLAS_ddrMemBits> >& in0,
                               hls::stream<ap_uint<BLAS_ddrMemBits> >& in1,
                               hls::stream<ap_uint<BLAS_ddrMemBits> >& in2,
                               hls::stream<ap_uint<32> >& in3) {
#pragma HLS INTERFACE m_axi port = p_rdPtr offset = slave bundle = gmem latency = 125
#pragma HLS INTERFACE m_axi port = p_wrPtr offset = slave bundle = gmem latency = 125
#pragma HLS INTERFACE s_axilite port = p_rdPtr bundle = control
#pragma HLS INTERFACE s_axilite port = p_wrPtr bundle = control

#pragma HLS INTERFACE axis port = out0
#pragma HLS INTERFACE axis port = out1
#pragma HLS INTERFACE axis port = out2
#pragma HLS INTERFACE axis port = out3
#pragma HLS INTERFACE axis port = in0
#pragma HLS INTERFACE axis port = in1
#pragma HLS INTERFACE axis port = in2
#pragma HLS INTERFACE axis port = in3

#pragma HLS INTERFACE s_axilite port = return bundle = control

#pragma HLS DATAFLOW

    xf::blas::gemmLoad<BLAS_maxNumInstrs, BLAS_memWordsPerInstr, BLAS_parEntries, BLAS_mParWords, BLAS_kParWords,
                       BLAS_nParWords, BLAS_ddrMemBits>(p_rdPtr, out0, out1, out2, out3);

    xf::blas::gemmStoreABX<BLAS_resOffsetBytes, BLAS_memWordsPerInstr, BLAS_parEntries, BLAS_mParWords, BLAS_kParWords,
                           BLAS_nParWords, BLAS_ddrMemBits>(in0, in1, in2, in3, p_wrPtr);
}
