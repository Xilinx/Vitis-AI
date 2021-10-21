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
#include "ap_int.h"
#include "hls_stream.h"
extern "C" void gemmMergeKernel(hls::stream<ap_uint<BLAS_ddrMemBits> >& p_sum_0,
                                hls::stream<ap_uint<BLAS_ddrMemBits> >& p_sum_1,
                                hls::stream<ap_uint<BLAS_ddrMemBits> >& p_sum_2,
                                hls::stream<ap_uint<BLAS_ddrMemBits> >& p_sum_3,
                                hls::stream<ap_uint<BLAS_ddrMemBits> >& p_sum_4,
                                hls::stream<ap_uint<BLAS_ddrMemBits> >& p_sum_5,
                                hls::stream<ap_uint<BLAS_ddrMemBits> >& p_sum_6,
                                hls::stream<ap_uint<BLAS_ddrMemBits> >& p_sum_7,
                                hls::stream<ap_uint<BLAS_ddrMemBits> >& p_sum_8,
                                hls::stream<ap_uint<BLAS_ddrMemBits> >& p_sum_9,
                                hls::stream<ap_uint<BLAS_ddrMemBits> >& p_sum_10,
                                hls::stream<ap_uint<BLAS_ddrMemBits> >& p_sum_11,
                                hls::stream<ap_uint<BLAS_ddrMemBits> >& p_sum_12,
                                hls::stream<ap_uint<BLAS_ddrMemBits> >& p_sum_13,
                                hls::stream<ap_uint<BLAS_ddrMemBits> >& p_sum_14,
                                hls::stream<ap_uint<BLAS_ddrMemBits> >& p_sum_15,
                                hls::stream<ap_uint<BLAS_ddrMemBits> >& p_sum) {
#pragma HLS INTERFACE ap_ctrl_none port = return

#pragma HLS INTERFACE axis port = p_sum_0 bundle = p_sum_0
#pragma HLS INTERFACE axis port = p_sum_1 bundle = p_sum_1
#pragma HLS INTERFACE axis port = p_sum_2 bundle = p_sum_2
#pragma HLS INTERFACE axis port = p_sum_3 bundle = p_sum_3
#pragma HLS INTERFACE axis port = p_sum_4 bundle = p_sum_4
#pragma HLS INTERFACE axis port = p_sum_5 bundle = p_sum_5
#pragma HLS INTERFACE axis port = p_sum_6 bundle = p_sum_6
#pragma HLS INTERFACE axis port = p_sum_7 bundle = p_sum_7
#pragma HLS INTERFACE axis port = p_sum_8 bundle = p_sum_8
#pragma HLS INTERFACE axis port = p_sum_9 bundle = p_sum_9
#pragma HLS INTERFACE axis port = p_sum_10 bundle = p_sum_10
#pragma HLS INTERFACE axis port = p_sum_11 bundle = p_sum_11
#pragma HLS INTERFACE axis port = p_sum_12 bundle = p_sum_12
#pragma HLS INTERFACE axis port = p_sum_13 bundle = p_sum_13
#pragma HLS INTERFACE axis port = p_sum_14 bundle = p_sum_14
#pragma HLS INTERFACE axis port = p_sum_15 bundle = p_sum_15
#pragma HLS INTERFACE axis port = p_sum bundle = p_sum
    p_sum.write(p_sum_0.read());
    p_sum.write(p_sum_1.read());
    p_sum.write(p_sum_2.read());
    p_sum.write(p_sum_3.read());
    p_sum.write(p_sum_4.read());
    p_sum.write(p_sum_5.read());
    p_sum.write(p_sum_6.read());
    p_sum.write(p_sum_7.read());
    p_sum.write(p_sum_8.read());
    p_sum.write(p_sum_9.read());
    p_sum.write(p_sum_10.read());
    p_sum.write(p_sum_11.read());
    p_sum.write(p_sum_12.read());
    p_sum.write(p_sum_13.read());
    p_sum.write(p_sum_14.read());
    p_sum.write(p_sum_15.read());
}
