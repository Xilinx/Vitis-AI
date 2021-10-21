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
#include "gemmStoreKernel.hpp"

extern "C" void gemmStoreKernel(ap_uint<BLAS_ddrMemBits>* p_wrPtr,
                                hls::stream<ap_uint<BLAS_ddrMemBits> >& l_cStr,
                                hls::stream<ap_uint<32> >& l_resStr) {
#pragma HLS INTERFACE m_axi port = p_wrPtr offset = slave bundle = gmem latency = 125 num_write_outstanding = 32
#pragma HLS INTERFACE s_axilite port = p_wrPtr bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

#pragma HLS INTERFACE axis port = l_cStr bundle = l_cStr
#pragma HLS INTERFACE axis port = l_resStr bundle = l_resStr

    xf::blas::gemmStore<BLAS_resOffsetBytes, BLAS_memWordsPerInstr, BLAS_parEntries, BLAS_mParWords, BLAS_kParWords,
                        BLAS_nParWords, BLAS_ddrMemBits>(l_cStr, l_resStr, p_wrPtr);
}
