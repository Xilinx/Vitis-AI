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
#include "kernel.hpp"

#include "xf_database/scan_col.hpp"

void aggr(hls::stream<ap_uint<64> >& c0_strm,
          hls::stream<ap_uint<64> >& c1_strm,
          hls::stream<bool>& e_strm,
          ap_uint<64>* bufo) {
    bool e = e_strm.read();
    ap_uint<64> s = 0;
    while (!e) {
#pragma HLS pipeline II = 1
        ap_uint<64> a = c0_strm.read();
        ap_uint<64> b = c1_strm.read();
        e = e_strm.read();
        s += (b - a);
    }
    *bufo = s;
}

extern "C" {
void Test(ap_uint<64 * VEC_LEN> buf0[BUF_DEPTH], ap_uint<64 * VEC_LEN> buf1[BUF_DEPTH], int nrow, ap_uint<64>* bufo) {
#pragma HLS INTERFACE m_axi port = buf0 bundle = gmem0_0 num_read_outstanding = 4 max_read_burst_length = \
    64 num_write_outstanding = 4 max_write_burst_length = 64 latency = 125 offset = slave

#pragma HLS INTERFACE m_axi port = buf1 bundle = gmem0_1 num_read_outstanding = 4 max_read_burst_length = \
    64 num_write_outstanding = 4 max_write_burst_length = 64 latency = 125 offset = slave

#pragma HLS INTERFACE m_axi port = bufo bundle = gmem2_0 num_read_outstanding = 4 max_read_burst_length = \
    64 num_write_outstanding = 4 max_write_burst_length = 64 latency = 125 offset = slave

#pragma HLS INTERFACE s_axilite port = buf0 bundle = control
#pragma HLS INTERFACE s_axilite port = buf1 bundle = control
#pragma HLS INTERFACE s_axilite port = nrow bundle = control
#pragma HLS INTERFACE s_axilite port = bufo bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

#pragma HLS dataflow

    hls::stream<ap_uint<64> > c0_strm("c0_strm");
#pragma HLS stream variable = c0_strm depth = 8
    hls::stream<ap_uint<64> > c1_strm("c1_strm");
#pragma HLS stream variable = c1_strm depth = 8
    hls::stream<bool> e_strm("e_strm");
#pragma HLS stream variable = e_strm depth = 8

    xf::database::scanCol<BURST_LEN, VEC_LEN, 8, 8>(buf0, buf1, nrow, c0_strm, c1_strm, e_strm);

    aggr(c0_strm, c1_strm, e_strm, bufo);
}
}
