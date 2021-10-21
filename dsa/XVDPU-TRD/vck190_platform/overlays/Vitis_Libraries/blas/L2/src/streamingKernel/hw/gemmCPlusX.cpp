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
#include "blasInstr.hpp"
#include "gemmCPlusX.hpp"

void gemmCPlusXKernel(hls::stream<ap_uint<BLAS_ddrMemBits> >& l_xStr,
                      hls::stream<ap_uint<BLAS_ddrMemBits> >& l_sStr,
                      hls::stream<ap_uint<BLAS_ddrMemBits> >& l_cStr) {
#pragma HLS INTERFACE axis port = l_cStr bundle = l_cStr
#pragma HLS INTERFACE axis port = l_xStr bundle = l_xStr
#pragma HLS INTERFACE axis port = l_sStr bundle = l_sStr
#pragma HLS INTERFACE ap_ctrl_none port = return

    xf::blas::gemmCPlusX<BLAS_dataType, BLAS_memWordsPerInstr, BLAS_parEntries, BLAS_mParWords, BLAS_kParWords,
                         BLAS_nParWords, BLAS_ddrMemBits>(l_xStr, l_sStr, l_cStr);
}
