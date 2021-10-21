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
#include "gemmLoadKernel.hpp"
#include "gemmStoreKernel.hpp"

extern "C" void gemmLoadStoreKernel(ap_uint<BLAS_ddrMemBits>* p_rdPtr,
                                    ap_uint<BLAS_ddrMemBits>* p_wrPtr,
                                    hls::stream<ap_uint<BLAS_ddrMemBits> >& l_aStr,
                                    hls::stream<ap_uint<BLAS_ddrMemBits> >& l_bStr,
                                    hls::stream<ap_uint<BLAS_ddrMemBits> >& l_xStr,
                                    hls::stream<ap_uint<16> >& l_opCodeStr,
                                    hls::stream<ap_uint<BLAS_ddrMemBits> >& l_cStr,
                                    hls::stream<ap_uint<32> >& l_resStr) {
#pragma HLS INTERFACE m_axi port = p_rdPtr offset = slave bundle = gmem latency = 125 num_read_outstanding = 32
#pragma HLS INTERFACE m_axi port = p_wrPtr offset = slave bundle = gmem latency = 125 num_write_outstanding = 32
#pragma HLS INTERFACE s_axilite port = p_rdPtr bundle = control
#pragma HLS INTERFACE s_axilite port = p_wrPtr bundle = control

#pragma HLS INTERFACE axis port = l_aStr
#pragma HLS INTERFACE axis port = l_bStr
#pragma HLS INTERFACE axis port = l_xStr
#pragma HLS INTERFACE axis port = l_opCodeStr
#pragma HLS INTERFACE axis port = l_cStr
#pragma HLS INTERFACE axis port = l_resStr

#pragma HLS INTERFACE s_axilite port = return bundle = control

#pragma HLS DATAFLOW

    xf::blas::gemmLoad<BLAS_maxNumInstrs, BLAS_memWordsPerInstr, BLAS_parEntries, BLAS_mParWords, BLAS_kParWords,
                       BLAS_nParWords, BLAS_ddrMemBits>(p_rdPtr, l_aStr, l_bStr, l_xStr, l_opCodeStr);
    xf::blas::gemmStore<BLAS_resOffsetBytes, BLAS_memWordsPerInstr, BLAS_parEntries, BLAS_mParWords, BLAS_kParWords,
                        BLAS_nParWords, BLAS_ddrMemBits>(l_cStr, l_resStr, p_wrPtr);
}
