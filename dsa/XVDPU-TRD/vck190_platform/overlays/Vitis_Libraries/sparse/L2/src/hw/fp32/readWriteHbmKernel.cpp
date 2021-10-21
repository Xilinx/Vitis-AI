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
 * @file readWriteHbmKernel.cpp
 * @brief storeDatKernel definition.
 *
 * This file is part of Vitis SPARSE Library.
 */

#include "readWriteHbmKernel.hpp"

extern "C" void readWriteHbmKernel(ap_uint<SPARSE_hbmMemBits>* p_rd0,
                                   ap_uint<SPARSE_hbmMemBits>* p_wr0,
                                   ap_uint<SPARSE_hbmMemBits>* p_rd1,
                                   ap_uint<SPARSE_hbmMemBits>* p_wr1,
                                   ap_uint<SPARSE_hbmMemBits>* p_rd2,
                                   ap_uint<SPARSE_hbmMemBits>* p_wr2,
                                   ap_uint<SPARSE_hbmMemBits>* p_rd3,
                                   ap_uint<SPARSE_hbmMemBits>* p_wr3,
                                   ap_uint<SPARSE_hbmMemBits>* p_rd4,
                                   ap_uint<SPARSE_hbmMemBits>* p_wr4,
                                   ap_uint<SPARSE_hbmMemBits>* p_rd5,
                                   ap_uint<SPARSE_hbmMemBits>* p_wr5,
                                   ap_uint<SPARSE_hbmMemBits>* p_rd6,
                                   ap_uint<SPARSE_hbmMemBits>* p_wr6,
                                   ap_uint<SPARSE_hbmMemBits>* p_rd7,
                                   ap_uint<SPARSE_hbmMemBits>* p_wr7,
                                   hls::stream<ap_uint<SPARSE_hbmMemBits> >& out0,
                                   hls::stream<ap_uint<SPARSE_dataBits * SPARSE_parEntries> >& in0,
                                   hls::stream<ap_uint<SPARSE_hbmMemBits> >& out1,
                                   hls::stream<ap_uint<SPARSE_dataBits * SPARSE_parEntries> >& in1,
                                   hls::stream<ap_uint<SPARSE_hbmMemBits> >& out2,
                                   hls::stream<ap_uint<SPARSE_dataBits * SPARSE_parEntries> >& in2,
                                   hls::stream<ap_uint<SPARSE_hbmMemBits> >& out3,
                                   hls::stream<ap_uint<SPARSE_dataBits * SPARSE_parEntries> >& in3,
                                   hls::stream<ap_uint<SPARSE_hbmMemBits> >& out4,
                                   hls::stream<ap_uint<SPARSE_dataBits * SPARSE_parEntries> >& in4,
                                   hls::stream<ap_uint<SPARSE_hbmMemBits> >& out5,
                                   hls::stream<ap_uint<SPARSE_dataBits * SPARSE_parEntries> >& in5,
                                   hls::stream<ap_uint<SPARSE_hbmMemBits> >& out6,
                                   hls::stream<ap_uint<SPARSE_dataBits * SPARSE_parEntries> >& in6,
                                   hls::stream<ap_uint<SPARSE_hbmMemBits> >& out7,
                                   hls::stream<ap_uint<SPARSE_dataBits * SPARSE_parEntries> >& in7) {
#pragma HLS INTERFACE m_axi port = p_rd0 offset = slave bundle = gmem0
#pragma HLS INTERFACE m_axi port = p_wr0 offset = slave bundle = gmem0
#pragma HLS INTERFACE m_axi port = p_rd1 offset = slave bundle = gmem1
#pragma HLS INTERFACE m_axi port = p_wr1 offset = slave bundle = gmem1
#pragma HLS INTERFACE m_axi port = p_rd2 offset = slave bundle = gmem2
#pragma HLS INTERFACE m_axi port = p_wr2 offset = slave bundle = gmem2
#pragma HLS INTERFACE m_axi port = p_rd3 offset = slave bundle = gmem3
#pragma HLS INTERFACE m_axi port = p_wr3 offset = slave bundle = gmem3
#pragma HLS INTERFACE m_axi port = p_rd4 offset = slave bundle = gmem4
#pragma HLS INTERFACE m_axi port = p_wr4 offset = slave bundle = gmem4
#pragma HLS INTERFACE m_axi port = p_rd5 offset = slave bundle = gmem5
#pragma HLS INTERFACE m_axi port = p_wr5 offset = slave bundle = gmem5
#pragma HLS INTERFACE m_axi port = p_rd6 offset = slave bundle = gmem6
#pragma HLS INTERFACE m_axi port = p_wr6 offset = slave bundle = gmem6
#pragma HLS INTERFACE m_axi port = p_rd7 offset = slave bundle = gmem7
#pragma HLS INTERFACE m_axi port = p_wr7 offset = slave bundle = gmem7
#pragma HLS INTERFACE axis port = out0
#pragma HLS INTERFACE axis port = in0
#pragma HLS INTERFACE axis port = out1
#pragma HLS INTERFACE axis port = in1
#pragma HLS INTERFACE axis port = out2
#pragma HLS INTERFACE axis port = in2
#pragma HLS INTERFACE axis port = out3
#pragma HLS INTERFACE axis port = in3
#pragma HLS INTERFACE axis port = out4
#pragma HLS INTERFACE axis port = in4
#pragma HLS INTERFACE axis port = out5
#pragma HLS INTERFACE axis port = in5
#pragma HLS INTERFACE axis port = out6
#pragma HLS INTERFACE axis port = in6
#pragma HLS INTERFACE axis port = out7
#pragma HLS INTERFACE axis port = in7
#pragma HLS INTERFACE s_axilite port = p_rd0 bundle = control
#pragma HLS INTERFACE s_axilite port = p_wr0 bundle = control
#pragma HLS INTERFACE s_axilite port = p_rd1 bundle = control
#pragma HLS INTERFACE s_axilite port = p_wr1 bundle = control
#pragma HLS INTERFACE s_axilite port = p_rd2 bundle = control
#pragma HLS INTERFACE s_axilite port = p_wr2 bundle = control
#pragma HLS INTERFACE s_axilite port = p_rd3 bundle = control
#pragma HLS INTERFACE s_axilite port = p_wr3 bundle = control
#pragma HLS INTERFACE s_axilite port = p_rd4 bundle = control
#pragma HLS INTERFACE s_axilite port = p_wr4 bundle = control
#pragma HLS INTERFACE s_axilite port = p_rd5 bundle = control
#pragma HLS INTERFACE s_axilite port = p_wr5 bundle = control
#pragma HLS INTERFACE s_axilite port = p_rd6 bundle = control
#pragma HLS INTERFACE s_axilite port = p_wr6 bundle = control
#pragma HLS INTERFACE s_axilite port = p_rd7 bundle = control
#pragma HLS INTERFACE s_axilite port = p_wr7 bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

#pragma HLS DATAFLOW
    xf::sparse::rdWrHbmChannel<SPARSE_maxParamHbmBlocks, SPARSE_maxRowBlocks, SPARSE_parEntries, SPARSE_parGroups,
                               SPARSE_dataBits, SPARSE_hbmMemBits, SPARSE_paramOffset>(p_rd0, in0, p_wr0, out0);

    xf::sparse::rdWrHbmChannel<SPARSE_maxParamHbmBlocks, SPARSE_maxRowBlocks, SPARSE_parEntries, SPARSE_parGroups,
                               SPARSE_dataBits, SPARSE_hbmMemBits, SPARSE_paramOffset>(p_rd1, in1, p_wr1, out1);

    xf::sparse::rdWrHbmChannel<SPARSE_maxParamHbmBlocks, SPARSE_maxRowBlocks, SPARSE_parEntries, SPARSE_parGroups,
                               SPARSE_dataBits, SPARSE_hbmMemBits, SPARSE_paramOffset>(p_rd2, in2, p_wr2, out2);

    xf::sparse::rdWrHbmChannel<SPARSE_maxParamHbmBlocks, SPARSE_maxRowBlocks, SPARSE_parEntries, SPARSE_parGroups,
                               SPARSE_dataBits, SPARSE_hbmMemBits, SPARSE_paramOffset>(p_rd3, in3, p_wr3, out3);

    xf::sparse::rdWrHbmChannel<SPARSE_maxParamHbmBlocks, SPARSE_maxRowBlocks, SPARSE_parEntries, SPARSE_parGroups,
                               SPARSE_dataBits, SPARSE_hbmMemBits, SPARSE_paramOffset>(p_rd4, in4, p_wr4, out4);

    xf::sparse::rdWrHbmChannel<SPARSE_maxParamHbmBlocks, SPARSE_maxRowBlocks, SPARSE_parEntries, SPARSE_parGroups,
                               SPARSE_dataBits, SPARSE_hbmMemBits, SPARSE_paramOffset>(p_rd5, in5, p_wr5, out5);

    xf::sparse::rdWrHbmChannel<SPARSE_maxParamHbmBlocks, SPARSE_maxRowBlocks, SPARSE_parEntries, SPARSE_parGroups,
                               SPARSE_dataBits, SPARSE_hbmMemBits, SPARSE_paramOffset>(p_rd6, in6, p_wr6, out6);

    xf::sparse::rdWrHbmChannel<SPARSE_maxParamHbmBlocks, SPARSE_maxRowBlocks, SPARSE_parEntries, SPARSE_parGroups,
                               SPARSE_dataBits, SPARSE_hbmMemBits, SPARSE_paramOffset>(p_rd7, in7, p_wr7, out7);
}
