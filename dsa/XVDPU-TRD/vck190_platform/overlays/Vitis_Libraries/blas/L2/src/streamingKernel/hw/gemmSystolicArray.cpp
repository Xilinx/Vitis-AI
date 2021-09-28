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
#include "gemmSystolicArray.hpp"
extern "C" void gemmSystolicArrayKernel(hls::stream<ap_uint<BLAS_ddrMemBits> >& p_a,
                                        hls::stream<ap_uint<2> >& p_tag,
                                        hls::stream<ap_uint<BLAS_ddrMemBits> >& p_b,
                                        hls::stream<ap_uint<BLAS_ddrMemBits> >& p_bOut,
                                        hls::stream<BLAS_dataType>& p_aOut_0,
                                        hls::stream<BLAS_dataType>& p_aOut_1,
                                        hls::stream<BLAS_dataType>& p_aOut_2,
                                        hls::stream<BLAS_dataType>& p_aOut_3,
                                        hls::stream<BLAS_dataType>& p_aOut_4,
                                        hls::stream<BLAS_dataType>& p_aOut_5,
                                        hls::stream<BLAS_dataType>& p_aOut_6,
                                        hls::stream<BLAS_dataType>& p_aOut_7,
                                        hls::stream<BLAS_dataType>& p_aOut_8,
                                        hls::stream<BLAS_dataType>& p_aOut_9,
                                        hls::stream<BLAS_dataType>& p_aOut_10,
                                        hls::stream<BLAS_dataType>& p_aOut_11,
                                        hls::stream<BLAS_dataType>& p_aOut_12,
                                        hls::stream<BLAS_dataType>& p_aOut_13,
                                        hls::stream<BLAS_dataType>& p_aOut_14,
                                        hls::stream<BLAS_dataType>& p_aOut_15,
                                        hls::stream<ap_uint<2> >& p_tagOut_0,
                                        hls::stream<ap_uint<2> >& p_tagOut_1,
                                        hls::stream<ap_uint<2> >& p_tagOut_2,
                                        hls::stream<ap_uint<2> >& p_tagOut_3,
                                        hls::stream<ap_uint<2> >& p_tagOut_4,
                                        hls::stream<ap_uint<2> >& p_tagOut_5,
                                        hls::stream<ap_uint<2> >& p_tagOut_6,
                                        hls::stream<ap_uint<2> >& p_tagOut_7,
                                        hls::stream<ap_uint<2> >& p_tagOut_8,
                                        hls::stream<ap_uint<2> >& p_tagOut_9,
                                        hls::stream<ap_uint<2> >& p_tagOut_10,
                                        hls::stream<ap_uint<2> >& p_tagOut_11,
                                        hls::stream<ap_uint<2> >& p_tagOut_12,
                                        hls::stream<ap_uint<2> >& p_tagOut_13,
                                        hls::stream<ap_uint<2> >& p_tagOut_14,
                                        hls::stream<ap_uint<2> >& p_tagOut_15) {
#pragma HLS INTERFACE ap_ctrl_none port = return

#pragma HLS INTERFACE axis port = p_a
#pragma HLS INTERFACE axis port = p_tag
#pragma HLS INTERFACE axis port = p_b
#pragma HLS INTERFACE axis port = p_bOut
#pragma HLS INTERFACE axis port = p_aOut_0
#pragma HLS INTERFACE axis port = p_aOut_1
#pragma HLS INTERFACE axis port = p_aOut_2
#pragma HLS INTERFACE axis port = p_aOut_3
#pragma HLS INTERFACE axis port = p_aOut_4
#pragma HLS INTERFACE axis port = p_aOut_5
#pragma HLS INTERFACE axis port = p_aOut_6
#pragma HLS INTERFACE axis port = p_aOut_7
#pragma HLS INTERFACE axis port = p_aOut_8
#pragma HLS INTERFACE axis port = p_aOut_9
#pragma HLS INTERFACE axis port = p_aOut_10
#pragma HLS INTERFACE axis port = p_aOut_11
#pragma HLS INTERFACE axis port = p_aOut_12
#pragma HLS INTERFACE axis port = p_aOut_13
#pragma HLS INTERFACE axis port = p_aOut_14
#pragma HLS INTERFACE axis port = p_aOut_15
#pragma HLS INTERFACE axis port = p_tagOut_0
#pragma HLS INTERFACE axis port = p_tagOut_1
#pragma HLS INTERFACE axis port = p_tagOut_2
#pragma HLS INTERFACE axis port = p_tagOut_3
#pragma HLS INTERFACE axis port = p_tagOut_4
#pragma HLS INTERFACE axis port = p_tagOut_5
#pragma HLS INTERFACE axis port = p_tagOut_6
#pragma HLS INTERFACE axis port = p_tagOut_7
#pragma HLS INTERFACE axis port = p_tagOut_8
#pragma HLS INTERFACE axis port = p_tagOut_9
#pragma HLS INTERFACE axis port = p_tagOut_10
#pragma HLS INTERFACE axis port = p_tagOut_11
#pragma HLS INTERFACE axis port = p_tagOut_12
#pragma HLS INTERFACE axis port = p_tagOut_13
#pragma HLS INTERFACE axis port = p_tagOut_14
#pragma HLS INTERFACE axis port = p_tagOut_15

    xf::blas::gemmSystolicArray<BLAS_dataType, BLAS_parEntries, BLAS_kParWords, BLAS_ddrMemBits>(
        p_a, p_tag, p_b, p_bOut, p_aOut_0, p_aOut_1, p_aOut_2, p_aOut_3, p_aOut_4, p_aOut_5, p_aOut_6, p_aOut_7,
        p_aOut_8, p_aOut_9, p_aOut_10, p_aOut_11, p_aOut_12, p_aOut_13, p_aOut_14, p_aOut_15, p_tagOut_0, p_tagOut_1,
        p_tagOut_2, p_tagOut_3, p_tagOut_4, p_tagOut_5, p_tagOut_6, p_tagOut_7, p_tagOut_8, p_tagOut_9, p_tagOut_10,
        p_tagOut_11, p_tagOut_12, p_tagOut_13, p_tagOut_14, p_tagOut_15);
}
