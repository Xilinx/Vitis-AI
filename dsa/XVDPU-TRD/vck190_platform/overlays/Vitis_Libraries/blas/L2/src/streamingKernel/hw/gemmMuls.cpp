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
#include "gemmMuls.hpp"
#include "gemmMulsSink.hpp"
extern "C" void gemmMulsKernel(hls::stream<BLAS_dataType>& p_aIn0,
                               hls::stream<BLAS_dataType>& p_aIn1,
                               hls::stream<BLAS_dataType>& p_aIn2,
                               hls::stream<BLAS_dataType>& p_aIn3,
                               hls::stream<BLAS_dataType>& p_aIn4,
                               hls::stream<BLAS_dataType>& p_aIn5,
                               hls::stream<BLAS_dataType>& p_aIn6,
                               hls::stream<BLAS_dataType>& p_aIn7,
                               hls::stream<BLAS_dataType>& p_aIn8,
                               hls::stream<BLAS_dataType>& p_aIn9,
                               hls::stream<BLAS_dataType>& p_aIn10,
                               hls::stream<BLAS_dataType>& p_aIn11,
                               hls::stream<BLAS_dataType>& p_aIn12,
                               hls::stream<BLAS_dataType>& p_aIn13,
                               hls::stream<BLAS_dataType>& p_aIn14,
                               hls::stream<BLAS_dataType>& p_aIn15,
                               hls::stream<ap_uint<2> >& p_tagIn0,
                               hls::stream<ap_uint<2> >& p_tagIn1,
                               hls::stream<ap_uint<2> >& p_tagIn2,
                               hls::stream<ap_uint<2> >& p_tagIn3,
                               hls::stream<ap_uint<2> >& p_tagIn4,
                               hls::stream<ap_uint<2> >& p_tagIn5,
                               hls::stream<ap_uint<2> >& p_tagIn6,
                               hls::stream<ap_uint<2> >& p_tagIn7,
                               hls::stream<ap_uint<2> >& p_tagIn8,
                               hls::stream<ap_uint<2> >& p_tagIn9,
                               hls::stream<ap_uint<2> >& p_tagIn10,
                               hls::stream<ap_uint<2> >& p_tagIn11,
                               hls::stream<ap_uint<2> >& p_tagIn12,
                               hls::stream<ap_uint<2> >& p_tagIn13,
                               hls::stream<ap_uint<2> >& p_tagIn14,
                               hls::stream<ap_uint<2> >& p_tagIn15,
                               hls::stream<ap_uint<BLAS_ddrMemBits> >& p_bIn,
                               hls::stream<ap_uint<BLAS_ddrMemBits> >& p_out0,
                               hls::stream<ap_uint<BLAS_ddrMemBits> >& p_out1,
                               hls::stream<ap_uint<BLAS_ddrMemBits> >& p_out2,
                               hls::stream<ap_uint<BLAS_ddrMemBits> >& p_out3,
                               hls::stream<ap_uint<BLAS_ddrMemBits> >& p_out4,
                               hls::stream<ap_uint<BLAS_ddrMemBits> >& p_out5,
                               hls::stream<ap_uint<BLAS_ddrMemBits> >& p_out6,
                               hls::stream<ap_uint<BLAS_ddrMemBits> >& p_out7,
                               hls::stream<ap_uint<BLAS_ddrMemBits> >& p_out8,
                               hls::stream<ap_uint<BLAS_ddrMemBits> >& p_out9,
                               hls::stream<ap_uint<BLAS_ddrMemBits> >& p_out10,
                               hls::stream<ap_uint<BLAS_ddrMemBits> >& p_out11,
                               hls::stream<ap_uint<BLAS_ddrMemBits> >& p_out12,
                               hls::stream<ap_uint<BLAS_ddrMemBits> >& p_out13,
                               hls::stream<ap_uint<BLAS_ddrMemBits> >& p_out14,
                               hls::stream<ap_uint<BLAS_ddrMemBits> >& p_out15) {
#pragma HLS INTERFACE ap_ctrl_none port = return

#pragma HLS INTERFACE axis port = p_aIn0
#pragma HLS INTERFACE axis port = p_aIn1
#pragma HLS INTERFACE axis port = p_aIn2
#pragma HLS INTERFACE axis port = p_aIn3
#pragma HLS INTERFACE axis port = p_aIn4
#pragma HLS INTERFACE axis port = p_aIn5
#pragma HLS INTERFACE axis port = p_aIn6
#pragma HLS INTERFACE axis port = p_aIn7
#pragma HLS INTERFACE axis port = p_aIn8
#pragma HLS INTERFACE axis port = p_aIn9
#pragma HLS INTERFACE axis port = p_aIn10
#pragma HLS INTERFACE axis port = p_aIn11
#pragma HLS INTERFACE axis port = p_aIn12
#pragma HLS INTERFACE axis port = p_aIn13
#pragma HLS INTERFACE axis port = p_aIn14
#pragma HLS INTERFACE axis port = p_aIn15
#pragma HLS INTERFACE axis port = p_tagIn0
#pragma HLS INTERFACE axis port = p_tagIn1
#pragma HLS INTERFACE axis port = p_tagIn2
#pragma HLS INTERFACE axis port = p_tagIn3
#pragma HLS INTERFACE axis port = p_tagIn4
#pragma HLS INTERFACE axis port = p_tagIn5
#pragma HLS INTERFACE axis port = p_tagIn6
#pragma HLS INTERFACE axis port = p_tagIn7
#pragma HLS INTERFACE axis port = p_tagIn8
#pragma HLS INTERFACE axis port = p_tagIn9
#pragma HLS INTERFACE axis port = p_tagIn10
#pragma HLS INTERFACE axis port = p_tagIn11
#pragma HLS INTERFACE axis port = p_tagIn12
#pragma HLS INTERFACE axis port = p_tagIn13
#pragma HLS INTERFACE axis port = p_tagIn14
#pragma HLS INTERFACE axis port = p_tagIn15
#pragma HLS INTERFACE axis port = p_bIn
#pragma HLS INTERFACE axis port = p_out0
#pragma HLS INTERFACE axis port = p_out1
#pragma HLS INTERFACE axis port = p_out2
#pragma HLS INTERFACE axis port = p_out3
#pragma HLS INTERFACE axis port = p_out4
#pragma HLS INTERFACE axis port = p_out5
#pragma HLS INTERFACE axis port = p_out6
#pragma HLS INTERFACE axis port = p_out7
#pragma HLS INTERFACE axis port = p_out8
#pragma HLS INTERFACE axis port = p_out9
#pragma HLS INTERFACE axis port = p_out10
#pragma HLS INTERFACE axis port = p_out11
#pragma HLS INTERFACE axis port = p_out12
#pragma HLS INTERFACE axis port = p_out13
#pragma HLS INTERFACE axis port = p_out14
#pragma HLS INTERFACE axis port = p_out15

#pragma HLS DATAFLOW
    hls::stream<ap_uint<BLAS_ddrMemBits> > l_bOut[BLAS_parEntries];
    xf::blas::gemmMuls<BLAS_dataType, BLAS_parEntries, BLAS_kParWords>(p_aIn0, p_tagIn0, p_bIn, l_bOut[0], p_out0);
    xf::blas::gemmMuls<BLAS_dataType, BLAS_parEntries, BLAS_kParWords>(p_aIn1, p_tagIn1, l_bOut[0], l_bOut[1], p_out1);
    xf::blas::gemmMuls<BLAS_dataType, BLAS_parEntries, BLAS_kParWords>(p_aIn2, p_tagIn2, l_bOut[1], l_bOut[2], p_out2);
    xf::blas::gemmMuls<BLAS_dataType, BLAS_parEntries, BLAS_kParWords>(p_aIn3, p_tagIn3, l_bOut[2], l_bOut[3], p_out3);
    xf::blas::gemmMuls<BLAS_dataType, BLAS_parEntries, BLAS_kParWords>(p_aIn4, p_tagIn4, l_bOut[3], l_bOut[4], p_out4);
    xf::blas::gemmMuls<BLAS_dataType, BLAS_parEntries, BLAS_kParWords>(p_aIn5, p_tagIn5, l_bOut[4], l_bOut[5], p_out5);
    xf::blas::gemmMuls<BLAS_dataType, BLAS_parEntries, BLAS_kParWords>(p_aIn6, p_tagIn6, l_bOut[5], l_bOut[6], p_out6);
    xf::blas::gemmMuls<BLAS_dataType, BLAS_parEntries, BLAS_kParWords>(p_aIn7, p_tagIn7, l_bOut[6], l_bOut[7], p_out7);
    xf::blas::gemmMuls<BLAS_dataType, BLAS_parEntries, BLAS_kParWords>(p_aIn8, p_tagIn8, l_bOut[7], l_bOut[8], p_out8);
    xf::blas::gemmMuls<BLAS_dataType, BLAS_parEntries, BLAS_kParWords>(p_aIn9, p_tagIn9, l_bOut[8], l_bOut[9], p_out9);
    xf::blas::gemmMuls<BLAS_dataType, BLAS_parEntries, BLAS_kParWords>(p_aIn10, p_tagIn10, l_bOut[9], l_bOut[10],
                                                                       p_out10);
    xf::blas::gemmMuls<BLAS_dataType, BLAS_parEntries, BLAS_kParWords>(p_aIn11, p_tagIn11, l_bOut[10], l_bOut[11],
                                                                       p_out11);
    xf::blas::gemmMuls<BLAS_dataType, BLAS_parEntries, BLAS_kParWords>(p_aIn12, p_tagIn12, l_bOut[11], l_bOut[12],
                                                                       p_out12);
    xf::blas::gemmMuls<BLAS_dataType, BLAS_parEntries, BLAS_kParWords>(p_aIn13, p_tagIn13, l_bOut[12], l_bOut[13],
                                                                       p_out13);
    xf::blas::gemmMuls<BLAS_dataType, BLAS_parEntries, BLAS_kParWords>(p_aIn14, p_tagIn14, l_bOut[13], l_bOut[14],
                                                                       p_out14);
    xf::blas::gemmMulsSink<BLAS_dataType, BLAS_parEntries, BLAS_kParWords>(p_aIn15, p_tagIn15, l_bOut[14], p_out15);
}
