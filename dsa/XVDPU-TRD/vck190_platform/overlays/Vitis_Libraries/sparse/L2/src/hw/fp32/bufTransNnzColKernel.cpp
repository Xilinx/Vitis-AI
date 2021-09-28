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

/**
 * @file bufTransNnzColKernel.cpp
 * @brief bufTransNnzColKernel definition.
 *
 * This file is part of Vitis SPARSE Library.
 */

#include "bufTransNnzColKernel.hpp"
extern "C" void bufTransNnzColKernel(hls::stream<ap_uint<SPARSE_ddrMemBits> >& in0,
                                     hls::stream<ap_uint<SPARSE_dataBits * SPARSE_parEntries> >& out0,
                                     hls::stream<ap_uint<SPARSE_dataBits * SPARSE_parEntries> >& out1,
                                     hls::stream<ap_uint<SPARSE_dataBits * SPARSE_parEntries> >& out2,
                                     hls::stream<ap_uint<SPARSE_dataBits * SPARSE_parEntries> >& out3,
                                     hls::stream<ap_uint<SPARSE_dataBits * SPARSE_parEntries> >& out4,
                                     hls::stream<ap_uint<SPARSE_dataBits * SPARSE_parEntries> >& out5,
                                     hls::stream<ap_uint<SPARSE_dataBits * SPARSE_parEntries> >& out6,
                                     hls::stream<ap_uint<SPARSE_dataBits * SPARSE_parEntries> >& out7,
                                     hls::stream<ap_uint<SPARSE_dataBits * SPARSE_parEntries> >& out8,
                                     hls::stream<ap_uint<SPARSE_dataBits * SPARSE_parEntries> >& out9,
                                     hls::stream<ap_uint<SPARSE_dataBits * SPARSE_parEntries> >& out10,
                                     hls::stream<ap_uint<SPARSE_dataBits * SPARSE_parEntries> >& out11,
                                     hls::stream<ap_uint<SPARSE_dataBits * SPARSE_parEntries> >& out12,
                                     hls::stream<ap_uint<SPARSE_dataBits * SPARSE_parEntries> >& out13,
                                     hls::stream<ap_uint<SPARSE_dataBits * SPARSE_parEntries> >& out14,
                                     hls::stream<ap_uint<SPARSE_dataBits * SPARSE_parEntries> >& out15) {
#pragma HLS INTERFACE axis port = in0
#pragma HLS INTERFACE axis port = out0
#pragma HLS INTERFACE axis port = out1
#pragma HLS INTERFACE axis port = out2
#pragma HLS INTERFACE axis port = out3
#pragma HLS INTERFACE axis port = out4
#pragma HLS INTERFACE axis port = out5
#pragma HLS INTERFACE axis port = out6
#pragma HLS INTERFACE axis port = out7
#pragma HLS INTERFACE axis port = out8
#pragma HLS INTERFACE axis port = out9
#pragma HLS INTERFACE axis port = out10
#pragma HLS INTERFACE axis port = out11
#pragma HLS INTERFACE axis port = out12
#pragma HLS INTERFACE axis port = out13
#pragma HLS INTERFACE axis port = out14
#pragma HLS INTERFACE axis port = out15
#pragma HLS INTERFACE ap_ctrl_none port = return

    const static unsigned int t_DataBits = SPARSE_dataBits * SPARSE_parEntries;

#pragma HLS DATAFLOW
    hls::stream<ap_uint<t_DataBits> > l_datStr[SPARSE_hbmChannels];
    xf::sparse::bufTransNnzCol<SPARSE_maxColMemBlocks, SPARSE_hbmChannels, SPARSE_parEntries, SPARSE_ddrMemBits,
                               SPARSE_dataBits>(in0, l_datStr);
    xf::sparse::fwdDatStr<t_DataBits>(l_datStr[0], out0);
    xf::sparse::fwdDatStr<t_DataBits>(l_datStr[1], out1);
    xf::sparse::fwdDatStr<t_DataBits>(l_datStr[2], out2);
    xf::sparse::fwdDatStr<t_DataBits>(l_datStr[3], out3);
    xf::sparse::fwdDatStr<t_DataBits>(l_datStr[4], out4);
    xf::sparse::fwdDatStr<t_DataBits>(l_datStr[5], out5);
    xf::sparse::fwdDatStr<t_DataBits>(l_datStr[6], out6);
    xf::sparse::fwdDatStr<t_DataBits>(l_datStr[7], out7);
    xf::sparse::fwdDatStr<t_DataBits>(l_datStr[8], out8);
    xf::sparse::fwdDatStr<t_DataBits>(l_datStr[9], out9);
    xf::sparse::fwdDatStr<t_DataBits>(l_datStr[10], out10);
    xf::sparse::fwdDatStr<t_DataBits>(l_datStr[11], out11);
    xf::sparse::fwdDatStr<t_DataBits>(l_datStr[12], out12);
    xf::sparse::fwdDatStr<t_DataBits>(l_datStr[13], out13);
    xf::sparse::fwdDatStr<t_DataBits>(l_datStr[14], out14);
    xf::sparse::fwdDatStr<t_DataBits>(l_datStr[15], out15);
}
