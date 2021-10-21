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
#include "gemmAdds.hpp"

extern "C" void gemmAddsKernel(hls::stream<ap_uint<BLAS_ddrMemBits> >& p_in0,
                               hls::stream<ap_uint<BLAS_ddrMemBits> >& p_in1,
                               hls::stream<ap_uint<BLAS_ddrMemBits> >& p_in2,
                               hls::stream<ap_uint<BLAS_ddrMemBits> >& p_in3,
                               hls::stream<ap_uint<BLAS_ddrMemBits> >& p_in4,
                               hls::stream<ap_uint<BLAS_ddrMemBits> >& p_in5,
                               hls::stream<ap_uint<BLAS_ddrMemBits> >& p_in6,
                               hls::stream<ap_uint<BLAS_ddrMemBits> >& p_in7,
                               hls::stream<ap_uint<BLAS_ddrMemBits> >& p_in8,
                               hls::stream<ap_uint<BLAS_ddrMemBits> >& p_in9,
                               hls::stream<ap_uint<BLAS_ddrMemBits> >& p_in10,
                               hls::stream<ap_uint<BLAS_ddrMemBits> >& p_in11,
                               hls::stream<ap_uint<BLAS_ddrMemBits> >& p_in12,
                               hls::stream<ap_uint<BLAS_ddrMemBits> >& p_in13,
                               hls::stream<ap_uint<BLAS_ddrMemBits> >& p_in14,
                               hls::stream<ap_uint<BLAS_ddrMemBits> >& p_in15,
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

#pragma HLS INTERFACE axis port = p_in0
#pragma HLS INTERFACE axis port = p_in1
#pragma HLS INTERFACE axis port = p_in2
#pragma HLS INTERFACE axis port = p_in3
#pragma HLS INTERFACE axis port = p_in4
#pragma HLS INTERFACE axis port = p_in5
#pragma HLS INTERFACE axis port = p_in6
#pragma HLS INTERFACE axis port = p_in7
#pragma HLS INTERFACE axis port = p_in8
#pragma HLS INTERFACE axis port = p_in9
#pragma HLS INTERFACE axis port = p_in10
#pragma HLS INTERFACE axis port = p_in11
#pragma HLS INTERFACE axis port = p_in12
#pragma HLS INTERFACE axis port = p_in13
#pragma HLS INTERFACE axis port = p_in14
#pragma HLS INTERFACE axis port = p_in15
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
    xf::blas::gemmAdds<BLAS_dataType, BLAS_parEntries, BLAS_kParWords, 4>(p_in0, p_out0);
    xf::blas::gemmAdds<BLAS_dataType, BLAS_parEntries, BLAS_kParWords, 4>(p_in1, p_out1);
    xf::blas::gemmAdds<BLAS_dataType, BLAS_parEntries, BLAS_kParWords, 4>(p_in2, p_out2);
    xf::blas::gemmAdds<BLAS_dataType, BLAS_parEntries, BLAS_kParWords, 4>(p_in3, p_out3);
    xf::blas::gemmAdds<BLAS_dataType, BLAS_parEntries, BLAS_kParWords, 4>(p_in4, p_out4);
    xf::blas::gemmAdds<BLAS_dataType, BLAS_parEntries, BLAS_kParWords, 4>(p_in5, p_out5);
    xf::blas::gemmAdds<BLAS_dataType, BLAS_parEntries, BLAS_kParWords, 4>(p_in6, p_out6);
    xf::blas::gemmAdds<BLAS_dataType, BLAS_parEntries, BLAS_kParWords, 4>(p_in7, p_out7);
    xf::blas::gemmAdds<BLAS_dataType, BLAS_parEntries, BLAS_kParWords, 4>(p_in8, p_out8);
    xf::blas::gemmAdds<BLAS_dataType, BLAS_parEntries, BLAS_kParWords, 4>(p_in9, p_out9);
    xf::blas::gemmAdds<BLAS_dataType, BLAS_parEntries, BLAS_kParWords, 4>(p_in10, p_out10);
    xf::blas::gemmAdds<BLAS_dataType, BLAS_parEntries, BLAS_kParWords, 4>(p_in11, p_out11);
    xf::blas::gemmAdds<BLAS_dataType, BLAS_parEntries, BLAS_kParWords, 4>(p_in12, p_out12);
    xf::blas::gemmAdds<BLAS_dataType, BLAS_parEntries, BLAS_kParWords, 4>(p_in13, p_out13);
    xf::blas::gemmAdds<BLAS_dataType, BLAS_parEntries, BLAS_kParWords, 4>(p_in14, p_out14);
    xf::blas::gemmAdds<BLAS_dataType, BLAS_parEntries, BLAS_kParWords, 4>(p_in15, p_out15);
}
