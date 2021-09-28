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
#include "gemmMulsSink.hpp"
extern "C" void gemmMulsSinkKernel(hls::stream<BLAS_dataType>& p_aIn,
                                   hls::stream<ap_uint<2> >& p_tagIn,
                                   hls::stream<ap_uint<BLAS_ddrMemBits> >& p_bIn,
                                   hls::stream<ap_uint<BLAS_ddrMemBits> >& p_out) {
#pragma HLS INTERFACE ap_ctrl_none port = return

#pragma HLS INTERFACE axis port = p_aIn
#pragma HLS INTERFACE axis port = p_tagIn
#pragma HLS INTERFACE axis port = p_bIn
#pragma HLS INTERFACE axis port = p_out
    xf::blas::gemmMulsSink<BLAS_dataType, BLAS_parEntries, BLAS_kParWords>(p_aIn, p_tagIn, p_bIn, p_out);
}
