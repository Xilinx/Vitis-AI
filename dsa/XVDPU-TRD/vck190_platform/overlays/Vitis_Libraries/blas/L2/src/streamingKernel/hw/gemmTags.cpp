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
#include "gemmTags.hpp"

void gemmTagsKernel(hls::stream<ap_uint<BLAS_ddrMemBits> >& p_a,
                    hls::stream<ap_uint<BLAS_ddrMemBits> >& p_b,
                    hls::stream<ap_uint<BLAS_ddrMemBits> >& p_aOut,
                    hls::stream<ap_uint<2> >& p_tagOut,
                    hls::stream<ap_uint<BLAS_ddrMemBits> >& p_bOut) {
#pragma HLS INTERFACE axis port = p_a
#pragma HLS INTERFACE axis port = p_b
#pragma HLS INTERFACE axis port = p_aOut
#pragma HLS INTERFACE axis port = p_tagOut
#pragma HLS INTERFACE axis port = p_bOut
#pragma HLS INTERFACE ap_ctrl_none port = return

    xf::blas::gemmTags<BLAS_dataType, BLAS_memWordsPerInstr, BLAS_parEntries, BLAS_mParWords, BLAS_kParWords,
                       BLAS_nParWords, BLAS_ddrMemBits>(p_a, p_b, p_aOut, p_tagOut, p_bOut);
}
