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
#ifndef __SYNTHESIS__
#include <stdio.h>
#include <iostream>
#endif

#include "xf_database/gqe_kernel_aggr_v2.hpp"

#include "xf_database/gqe_blocks_v2/load_config_aggr.hpp"
#include "xf_database/gqe_blocks_v2/scan_cols_aggr.hpp"
#include "xf_database/gqe_blocks_v2/write_out_part_aggr.hpp"
#include "xf_database/gqe_blocks/stream_helper.hpp"
#include "xf_database/gqe_blocks/eval_part.hpp"
#include "xf_database/gqe_blocks/filter_part.hpp"
#include "xf_database/gqe_blocks/group_aggregate_part.hpp"
#include "xf_database/gqe_blocks/aggr_part.hpp"
#include "xf_database/gqe_blocks/write_info.hpp"

#include <ap_int.h>
#include <hls_stream.h>

extern "C" void gqeAggr(                                         //
    ap_uint<8 * TPCH_INT_SZ * VEC_SCAN> buf_in0[TEST_BUF_DEPTH], // input data
    ap_uint<8 * TPCH_INT_SZ * VEC_SCAN> buf_in1[TEST_BUF_DEPTH], // input data
    ap_uint<8 * TPCH_INT_SZ * VEC_SCAN> buf_in2[TEST_BUF_DEPTH], // input data
    ap_uint<8 * TPCH_INT_SZ * VEC_SCAN> buf_in3[TEST_BUF_DEPTH], // input data
    ap_uint<8 * TPCH_INT_SZ * VEC_SCAN> buf_in4[TEST_BUF_DEPTH], // input data
    ap_uint<8 * TPCH_INT_SZ * VEC_SCAN> buf_in5[TEST_BUF_DEPTH], // input data
    ap_uint<8 * TPCH_INT_SZ * VEC_SCAN> buf_in6[TEST_BUF_DEPTH], // input data
    ap_uint<8 * TPCH_INT_SZ * VEC_SCAN> buf_in7[TEST_BUF_DEPTH], // input data
    ap_uint<512> buf_metain[24],
    ap_uint<512> buf_metaout[24],
    ap_uint<8 * TPCH_INT_SZ * VEC_LEN> buf_out0[TEST_BUF_DEPTH],  // output data
    ap_uint<8 * TPCH_INT_SZ * VEC_LEN> buf_out1[TEST_BUF_DEPTH],  // output data
    ap_uint<8 * TPCH_INT_SZ * VEC_LEN> buf_out2[TEST_BUF_DEPTH],  // output data
    ap_uint<8 * TPCH_INT_SZ * VEC_LEN> buf_out3[TEST_BUF_DEPTH],  // output data
    ap_uint<8 * TPCH_INT_SZ * VEC_LEN> buf_out4[TEST_BUF_DEPTH],  // output data
    ap_uint<8 * TPCH_INT_SZ * VEC_LEN> buf_out5[TEST_BUF_DEPTH],  // output data
    ap_uint<8 * TPCH_INT_SZ * VEC_LEN> buf_out6[TEST_BUF_DEPTH],  // output data
    ap_uint<8 * TPCH_INT_SZ * VEC_LEN> buf_out7[TEST_BUF_DEPTH],  // output data
    ap_uint<8 * TPCH_INT_SZ * VEC_LEN> buf_out8[TEST_BUF_DEPTH],  // output data
    ap_uint<8 * TPCH_INT_SZ * VEC_LEN> buf_out9[TEST_BUF_DEPTH],  // output data
    ap_uint<8 * TPCH_INT_SZ * VEC_LEN> buf_out10[TEST_BUF_DEPTH], // output data
    ap_uint<8 * TPCH_INT_SZ * VEC_LEN> buf_out11[TEST_BUF_DEPTH], // output data
    ap_uint<8 * TPCH_INT_SZ * VEC_LEN> buf_out12[TEST_BUF_DEPTH], // output data
    ap_uint<8 * TPCH_INT_SZ * VEC_LEN> buf_out13[TEST_BUF_DEPTH], // output data
    ap_uint<8 * TPCH_INT_SZ * VEC_LEN> buf_out14[TEST_BUF_DEPTH], // output data
    ap_uint<8 * TPCH_INT_SZ * VEC_LEN> buf_out15[TEST_BUF_DEPTH], // output data
    ap_uint<8 * TPCH_INT_SZ> buf_cfg[TEST_BUF_DEPTH],             // input config
    ap_uint<8 * TPCH_INT_SZ> buf_result_info[TEST_BUF_DEPTH],     // output result info

    ap_uint<8 * TPCH_INT_SZ * VEC_LEN> ping_buf0[TEST_BUF_DEPTH],
    ap_uint<8 * TPCH_INT_SZ * VEC_LEN> ping_buf1[TEST_BUF_DEPTH],
    ap_uint<8 * TPCH_INT_SZ * VEC_LEN> ping_buf2[TEST_BUF_DEPTH],
    ap_uint<8 * TPCH_INT_SZ * VEC_LEN> ping_buf3[TEST_BUF_DEPTH],

    ap_uint<8 * TPCH_INT_SZ * VEC_LEN> pong_buf0[TEST_BUF_DEPTH],
    ap_uint<8 * TPCH_INT_SZ * VEC_LEN> pong_buf1[TEST_BUF_DEPTH],
    ap_uint<8 * TPCH_INT_SZ * VEC_LEN> pong_buf2[TEST_BUF_DEPTH],
    ap_uint<8 * TPCH_INT_SZ * VEC_LEN> pong_buf3[TEST_BUF_DEPTH]) {
#pragma HLS INTERFACE m_axi offset = slave latency = 64 num_write_outstanding = 1 num_read_outstanding = \
    16 max_write_burst_length = 32 max_read_burst_length = 32 bundle = gmem0_0 port = buf_in0
#pragma HLS INTERFACE s_axilite port = buf_in0 bundle = control

#pragma HLS INTERFACE m_axi offset = slave latency = 64 num_write_outstanding = 1 num_read_outstanding = \
    16 max_write_burst_length = 32 max_read_burst_length = 32 bundle = gmem0_4 port = buf_in1
#pragma HLS INTERFACE s_axilite port = buf_in1 bundle = control

#pragma HLS INTERFACE m_axi offset = slave latency = 64 num_write_outstanding = 1 num_read_outstanding = \
    16 max_write_burst_length = 32 max_read_burst_length = 32 bundle = gmem0_5 port = buf_in2
#pragma HLS INTERFACE s_axilite port = buf_in2 bundle = control

#pragma HLS INTERFACE m_axi offset = slave latency = 64 num_write_outstanding = 1 num_read_outstanding = \
    16 max_write_burst_length = 32 max_read_burst_length = 32 bundle = gmem0_6 port = buf_in3
#pragma HLS INTERFACE s_axilite port = buf_in3 bundle = control

#pragma HLS INTERFACE m_axi offset = slave latency = 64 num_write_outstanding = 1 num_read_outstanding = \
    16 max_write_burst_length = 32 max_read_burst_length = 32 bundle = gmem0_0 port = buf_in4
#pragma HLS INTERFACE s_axilite port = buf_in4 bundle = control

#pragma HLS INTERFACE m_axi offset = slave latency = 64 num_write_outstanding = 1 num_read_outstanding = \
    16 max_write_burst_length = 32 max_read_burst_length = 32 bundle = gmem0_4 port = buf_in5
#pragma HLS INTERFACE s_axilite port = buf_in5 bundle = control

#pragma HLS INTERFACE m_axi offset = slave latency = 64 num_write_outstanding = 1 num_read_outstanding = \
    16 max_write_burst_length = 32 max_read_burst_length = 32 bundle = gmem0_5 port = buf_in6
#pragma HLS INTERFACE s_axilite port = buf_in6 bundle = control

#pragma HLS INTERFACE m_axi offset = slave latency = 64 num_write_outstanding = 1 num_read_outstanding = \
    16 max_write_burst_length = 32 max_read_burst_length = 32 bundle = gmem0_6 port = buf_in7
#pragma HLS INTERFACE s_axilite port = buf_in7 bundle = control

#pragma HLS INTERFACE m_axi offset = slave latency = 64 num_write_outstanding = 1 num_read_outstanding = \
    1 max_write_burst_length = 32 max_read_burst_length = 32 bundle = gmem0_2 port = buf_metain
#pragma HLS INTERFACE s_axilite port = buf_metain bundle = control

#pragma HLS INTERFACE m_axi offset = slave latency = 64 num_write_outstanding = 16 num_read_outstanding = \
    1 max_write_burst_length = 32 max_read_burst_length = 32 bundle = gmem0_1 port = buf_metaout
#pragma HLS INTERFACE s_axilite port = buf_metaout bundle = control

#pragma HLS INTERFACE m_axi offset = slave latency = 64 num_write_outstanding = 16 num_read_outstanding = \
    1 max_write_burst_length = 32 max_read_burst_length = 32 bundle = gmem0_1 port = buf_out0
#pragma HLS INTERFACE s_axilite port = buf_out0 bundle = control

#pragma HLS INTERFACE m_axi offset = slave latency = 64 num_write_outstanding = 16 num_read_outstanding = \
    1 max_write_burst_length = 32 max_read_burst_length = 32 bundle = gmem0_1 port = buf_out1
#pragma HLS INTERFACE s_axilite port = buf_out1 bundle = control

#pragma HLS INTERFACE m_axi offset = slave latency = 64 num_write_outstanding = 16 num_read_outstanding = \
    1 max_write_burst_length = 32 max_read_burst_length = 32 bundle = gmem0_1 port = buf_out2
#pragma HLS INTERFACE s_axilite port = buf_out2 bundle = control

#pragma HLS INTERFACE m_axi offset = slave latency = 64 num_write_outstanding = 16 num_read_outstanding = \
    1 max_write_burst_length = 32 max_read_burst_length = 32 bundle = gmem0_1 port = buf_out3
#pragma HLS INTERFACE s_axilite port = buf_out3 bundle = control

#pragma HLS INTERFACE m_axi offset = slave latency = 64 num_write_outstanding = 16 num_read_outstanding = \
    1 max_write_burst_length = 32 max_read_burst_length = 32 bundle = gmem0_1 port = buf_out4
#pragma HLS INTERFACE s_axilite port = buf_out4 bundle = control

#pragma HLS INTERFACE m_axi offset = slave latency = 64 num_write_outstanding = 16 num_read_outstanding = \
    1 max_write_burst_length = 32 max_read_burst_length = 32 bundle = gmem0_1 port = buf_out5
#pragma HLS INTERFACE s_axilite port = buf_out5 bundle = control

#pragma HLS INTERFACE m_axi offset = slave latency = 64 num_write_outstanding = 16 num_read_outstanding = \
    1 max_write_burst_length = 32 max_read_burst_length = 32 bundle = gmem0_1 port = buf_out6
#pragma HLS INTERFACE s_axilite port = buf_out6 bundle = control

#pragma HLS INTERFACE m_axi offset = slave latency = 64 num_write_outstanding = 16 num_read_outstanding = \
    1 max_write_burst_length = 32 max_read_burst_length = 32 bundle = gmem0_1 port = buf_out7
#pragma HLS INTERFACE s_axilite port = buf_out7 bundle = control

#pragma HLS INTERFACE m_axi offset = slave latency = 64 num_write_outstanding = 16 num_read_outstanding = \
    1 max_write_burst_length = 32 max_read_burst_length = 32 bundle = gmem0_1 port = buf_out8
#pragma HLS INTERFACE s_axilite port = buf_out8 bundle = control

#pragma HLS INTERFACE m_axi offset = slave latency = 64 num_write_outstanding = 16 num_read_outstanding = \
    1 max_write_burst_length = 32 max_read_burst_length = 32 bundle = gmem0_1 port = buf_out9
#pragma HLS INTERFACE s_axilite port = buf_out9 bundle = control

#pragma HLS INTERFACE m_axi offset = slave latency = 64 num_write_outstanding = 16 num_read_outstanding = \
    1 max_write_burst_length = 32 max_read_burst_length = 32 bundle = gmem0_1 port = buf_out10
#pragma HLS INTERFACE s_axilite port = buf_out10 bundle = control

#pragma HLS INTERFACE m_axi offset = slave latency = 64 num_write_outstanding = 16 num_read_outstanding = \
    1 max_write_burst_length = 32 max_read_burst_length = 32 bundle = gmem0_1 port = buf_out11
#pragma HLS INTERFACE s_axilite port = buf_out11 bundle = control

#pragma HLS INTERFACE m_axi offset = slave latency = 64 num_write_outstanding = 16 num_read_outstanding = \
    1 max_write_burst_length = 32 max_read_burst_length = 32 bundle = gmem0_1 port = buf_out12
#pragma HLS INTERFACE s_axilite port = buf_out12 bundle = control

#pragma HLS INTERFACE m_axi offset = slave latency = 64 num_write_outstanding = 16 num_read_outstanding = \
    1 max_write_burst_length = 32 max_read_burst_length = 32 bundle = gmem0_1 port = buf_out13
#pragma HLS INTERFACE s_axilite port = buf_out13 bundle = control

#pragma HLS INTERFACE m_axi offset = slave latency = 64 num_write_outstanding = 16 num_read_outstanding = \
    1 max_write_burst_length = 32 max_read_burst_length = 32 bundle = gmem0_1 port = buf_out14
#pragma HLS INTERFACE s_axilite port = buf_out14 bundle = control

#pragma HLS INTERFACE m_axi offset = slave latency = 64 num_write_outstanding = 16 num_read_outstanding = \
    1 max_write_burst_length = 32 max_read_burst_length = 32 bundle = gmem0_1 port = buf_out15
#pragma HLS INTERFACE s_axilite port = buf_out15 bundle = control

#pragma HLS INTERFACE m_axi offset = slave latency = 64 num_write_outstanding = 1 num_read_outstanding = \
    1 max_write_burst_length = 32 max_read_burst_length = 32 bundle = gmem0_2 port = buf_cfg
#pragma HLS INTERFACE s_axilite port = buf_cfg bundle = control

#pragma HLS INTERFACE m_axi offset = slave latency = 64 num_write_outstanding = 1 num_read_outstanding = \
    1 max_write_burst_length = 32 max_read_burst_length = 32 bundle = gmem0_3 port = buf_result_info
#pragma HLS INTERFACE s_axilite port = buf_result_info bundle = control

#pragma HLS INTERFACE m_axi offset = slave latency = 64 num_write_outstanding = 16 num_read_outstanding = \
    16 max_write_burst_length = 32 max_read_burst_length = 32 bundle = gmem1_0 port = ping_buf0
#pragma HLS INTERFACE s_axilite port = ping_buf0 bundle = control

#pragma HLS INTERFACE m_axi offset = slave latency = 64 num_write_outstanding = 16 num_read_outstanding = \
    16 max_write_burst_length = 32 max_read_burst_length = 32 bundle = gmem1_1 port = ping_buf1
#pragma HLS INTERFACE s_axilite port = ping_buf1 bundle = control

#pragma HLS INTERFACE m_axi offset = slave latency = 64 num_write_outstanding = 16 num_read_outstanding = \
    16 max_write_burst_length = 32 max_read_burst_length = 32 bundle = gmem1_2 port = ping_buf2
#pragma HLS INTERFACE s_axilite port = ping_buf2 bundle = control

#pragma HLS INTERFACE m_axi offset = slave latency = 64 num_write_outstanding = 16 num_read_outstanding = \
    16 max_write_burst_length = 32 max_read_burst_length = 32 bundle = gmem1_3 port = ping_buf3
#pragma HLS INTERFACE s_axilite port = ping_buf3 bundle = control

#pragma HLS INTERFACE m_axi offset = slave latency = 64 num_write_outstanding = 16 num_read_outstanding = \
    16 max_write_burst_length = 32 max_read_burst_length = 32 bundle = gmem2_0 port = pong_buf0
#pragma HLS INTERFACE s_axilite port = pong_buf0 bundle = control

#pragma HLS INTERFACE m_axi offset = slave latency = 64 num_write_outstanding = 16 num_read_outstanding = \
    16 max_write_burst_length = 32 max_read_burst_length = 32 bundle = gmem2_1 port = pong_buf1
#pragma HLS INTERFACE s_axilite port = pong_buf1 bundle = control

#pragma HLS INTERFACE m_axi offset = slave latency = 64 num_write_outstanding = 16 num_read_outstanding = \
    16 max_write_burst_length = 32 max_read_burst_length = 32 bundle = gmem2_2 port = pong_buf2
#pragma HLS INTERFACE s_axilite port = pong_buf2 bundle = control

#pragma HLS INTERFACE m_axi offset = slave latency = 64 num_write_outstanding = 16 num_read_outstanding = \
    16 max_write_burst_length = 32 max_read_burst_length = 32 bundle = gmem2_3 port = pong_buf3
#pragma HLS INTERFACE s_axilite port = pong_buf3 bundle = control

#pragma HLS INTERFACE s_axilite port = return bundle = control

// clang-format on
#pragma HLS dataflow

    using namespace xf::database::gqe;

    const int n_channel = 4;
    const int n_column = 8;

#ifndef __SYNTHESIS__
    printf("************************************************************\n");
    printf("             General Query Egnine Kernel\n");
    printf("************************************************************\n");
#endif

    //---------------------------scan--------------------------

    hls::stream<int8_t> cid_strm;
#pragma HLS stream variable = cid_strm depth = 8
#pragma HLS bind_storage variable = cid_strm type = fifo impl = srl

    hls::stream<ap_uint<32> > filter_cfg_strm;
#pragma HLS stream variable = filter_cfg_strm depth = 64
#pragma HLS bind_storage variable = filter_cfg_strm type = fifo impl = srl

    hls::stream<ap_uint<32> > alu0_cfg_strm;
#pragma HLS stream variable = alu0_cfg_strm depth = 16
#pragma HLS bind_storage variable = alu0_cfg_strm type = fifo impl = srl

    hls::stream<ap_uint<32> > alu1_cfg_strm;
#pragma HLS stream variable = alu1_cfg_strm depth = 16
#pragma HLS bind_storage variable = alu1_cfg_strm type = fifo impl = srl

    hls::stream<ap_uint<n_column * n_column> > shuffle1_cfg_strm[n_channel];
#pragma HLS stream variable = shuffle1_cfg_strm depth = 4
#pragma HLS array_partition variable = shuffle1_cfg_strm complete
#pragma HLS bind_storage variable = shuffle1_cfg_strm type = fifo impl = srl

    hls::stream<ap_uint<n_column * n_column> > shuffle2_cfg_strm[n_channel];
#pragma HLS stream variable = shuffle2_cfg_strm depth = 4
#pragma HLS array_partition variable = shuffle2_cfg_strm complete
#pragma HLS bind_storage variable = shuffle2_cfg_strm type = fifo impl = srl

    hls::stream<ap_uint<n_column * n_column> > shuffle3_cfg_strm[n_channel];
#pragma HLS stream variable = shuffle3_cfg_strm depth = 4
#pragma HLS array_partition variable = shuffle3_cfg_strm complete
#pragma HLS bind_storage variable = shuffle3_cfg_strm type = fifo impl = srl

    hls::stream<ap_uint<n_column * n_column> > shuffle4_cfg_strm[n_channel];
#pragma HLS stream variable = shuffle4_cfg_strm depth = 4
#pragma HLS array_partition variable = shuffle4_cfg_strm complete
#pragma HLS bind_storage variable = shuffle4_cfg_strm type = fifo impl = srl

    hls::stream<ap_uint<32> > merge_column_cfg_strm;
#pragma HLS stream variable = merge_column_cfg_strm depth = 4
#pragma HLS bind_storage variable = merge_column_cfg_strm type = fifo impl = srl

    hls::stream<ap_uint<32> > group_aggr_cfg_strm;
#pragma HLS stream variable = group_aggr_cfg_strm depth = 4
#pragma HLS bind_storage variable = group_aggr_cfg_strm type = fifo impl = srl

    hls::stream<bool> direct_aggr_cfg_strm;
#pragma HLS stream variable = direct_aggr_cfg_strm depth = 4
#pragma HLS bind_storage variable = direct_aggr_cfg_strm type = fifo impl = srl

    hls::stream<ap_uint<32> > write_cfg_strm;
#pragma HLS stream variable = write_cfg_strm depth = 4
#pragma HLS bind_storage variable = write_cfg_strm type = fifo impl = srl

    hls::stream<ap_uint<8 * TPCH_INT_SZ> > scan_strms[n_channel][n_column];
#pragma HLS stream variable = scan_strms depth = 8
#pragma HLS array_partition variable = scan_strms complete
#pragma HLS bind_storage variable = scan_strms type = fifo impl = srl

    hls::stream<bool> e_scan_strms[n_channel];
#pragma HLS stream variable = e_scan_strms depth = 8
#pragma HLS array_partition variable = e_scan_strms complete
#pragma HLS bind_storage variable = e_scan_strms type = fifo impl = srl

    hls::stream<int> nrow_strm;
#pragma HLS stream variable = nrow_strm depth = 2

#ifndef __SYNTHESIS__
    printf("******************************\n");
    printf("        Load config\n");
    printf("******************************\n");
#endif

    load_config<n_channel, n_column>(buf_cfg, buf_metain, nrow_strm, cid_strm, alu0_cfg_strm, alu1_cfg_strm,
                                     filter_cfg_strm, shuffle1_cfg_strm, shuffle2_cfg_strm, shuffle3_cfg_strm,
                                     shuffle4_cfg_strm, merge_column_cfg_strm, group_aggr_cfg_strm,
                                     direct_aggr_cfg_strm, write_cfg_strm);

#ifndef __SYNTHESIS__
    printf("******************************\n");
    printf("            Scan\n");
    printf("******************************\n");
#endif

    scan_cols_wrapper<8 * TPCH_INT_SZ, VEC_SCAN, n_channel, n_column>(buf_in0, buf_in1, buf_in2, buf_in3, buf_in4,
                                                                      buf_in5, buf_in6, buf_in7, nrow_strm, cid_strm,
                                                                      scan_strms, e_scan_strms);

#ifndef __SYNTHESIS__
    {
        printf("after Scan\n");
        size_t ss = 0;
        for (int ch = 0; ch < n_channel; ++ch) {
            size_t s = scan_strms[ch][0].size();
            printf("ch:%d nrow=%ld\n", ch, s);
            ss += s;
        }
        printf("total: nrow=%ld\n", ss);
    }
#endif

    //---------------------------eval--------------------------

    hls::stream<ap_uint<8 * TPCH_INT_SZ> > eval0_strms[n_channel][n_column];
#pragma HLS stream variable = eval0_strms depth = 8
#pragma HLS array_partition variable = eval0_strms complete
#pragma HLS bind_storage variable = eval0_strms type = fifo impl = srl

    hls::stream<bool> e_eval0_strms[n_channel];
#pragma HLS stream variable = e_eval0_strms depth = 8
#pragma HLS array_partition variable = e_eval0_strms complete
#pragma HLS bind_storage variable = e_eval0_strms type = fifo impl = srl

    hls::stream<ap_uint<8 * TPCH_INT_SZ> > eval1_strms[n_channel][n_column];
#pragma HLS stream variable = eval1_strms depth = 8
#pragma HLS array_partition variable = eval1_strms complete
#pragma HLS bind_storage variable = eval1_strms type = fifo impl = srl

    hls::stream<bool> e_eval1_strms[n_channel];
#pragma HLS stream variable = e_eval1_strms depth = 8
#pragma HLS array_partition variable = e_eval1_strms complete
#pragma HLS bind_storage variable = e_eval1_strms type = fifo impl = srl

#ifndef __SYNTHESIS__
    printf("******************************\n");
    printf("            Eval0\n");
    printf("******************************\n");
#endif

    // eval0
    multi_dynamic_eval_wrapper<n_channel, n_column, 8 * TPCH_INT_SZ, 8 * TPCH_INT_SZ>(
        alu0_cfg_strm, shuffle1_cfg_strm, scan_strms, e_scan_strms, eval0_strms, e_eval0_strms);

#ifndef __SYNTHESIS__
    printf("******************************\n");
    printf("            Eval1\n");
    printf("******************************\n");
#endif

    // eval1
    multi_dynamic_eval_wrapper<n_channel, n_column, 8 * TPCH_INT_SZ, 8 * TPCH_INT_SZ>(
        alu1_cfg_strm, shuffle2_cfg_strm, eval0_strms, e_eval0_strms, eval1_strms, e_eval1_strms);

#ifndef __SYNTHESIS__
    {
        printf("after Eval\n");
        size_t ss = 0;
        for (int ch = 0; ch < n_channel; ++ch) {
            size_t s = eval1_strms[ch][0].size();
            printf("ch:%d nrow=%ld\n", ch, s);
            ss += s;
        }
        printf("total: nrow=%ld\n", ss);
    }
#endif

    //---------------------------filter--------------------------

    hls::stream<ap_uint<8 * TPCH_INT_SZ> > flt_strms[n_channel][n_column];
#pragma HLS stream variable = flt_strms depth = 512
#pragma HLS array_partition variable = flt_strms complete
#pragma HLS bind_storage variable = flt_strms type = fifo impl = bram
    hls::stream<bool> e_flt_strms[n_channel];
#pragma HLS stream variable = e_flt_strms depth = 128
#pragma HLS array_partition variable = e_flt_strms complete
#pragma HLS bind_storage variable = e_flt_strms type = fifo impl = srl

#ifndef __SYNTHESIS__
    printf("******************************\n");
    printf("           Filter\n");
    printf("******************************\n");
#endif

    // filter
    filter_ongoing<8, 8, 4>(filter_cfg_strm, eval1_strms, e_eval1_strms, flt_strms, e_flt_strms);

#ifndef __SYNTHESIS__
    {
        printf("after Filter\n");
        size_t ss = 0;
        for (int ch = 0; ch < n_channel; ++ch) {
            size_t s = e_flt_strms[ch].size() - 1;
            printf("ch:%d nrow=%ld\n", ch, s);
            ss += s;
        }
        printf("total: nrow=%ld\n", ss);
    }

    for (int ch = 0; ch < n_channel; ++ch) {
        size_t s = flt_strms[ch][0].size();
        for (int c = 1; c < 4; ++c) {
            size_t ss = flt_strms[ch][c].size();
            if (s != ss) {
                printf(
                    "##### flt_strms[%d][0] has %ld row, flt_strms[%d][%d] has %ld "
                    "row.\n",
                    ch, s, ch, c, ss);
            }
        }
    }

    for (int ch = 0; ch < n_channel; ++ch) {
        for (int c = 0; c < n_column; ++c) {
            if (scan_strms[ch][c].size() != 0) {
                printf("##### ch_strms[%d][%d] has data left after filter.\n", ch, c);
            }
        }
    }

    for (int ch = 0; ch < n_channel; ++ch) {
        printf("end flag nubmer %d and data number %d\n", e_flt_strms[ch].size(), flt_strms[ch][0].size());
    }
#endif

    //------------------------group aggr--------------------------

    hls::stream<ap_uint<32> > result_info_strm;
#pragma HLS STREAM variable = result_info_strm depth = 8
#pragma HLS bind_storage variable = result_info_strm type = fifo impl = srl
    hls::stream<ap_uint<8 * TPCH_INT_SZ> > aggr_strms[2 * n_column];
#pragma HLS stream variable = aggr_strms depth = 8
#pragma HLS array_partition variable = aggr_strms complete
#pragma HLS bind_storage variable = aggr_strms type = fifo impl = srl
    hls::stream<bool> e_aggr_strms;
#pragma HLS stream variable = e_aggr_strms depth = 8
#pragma HLS bind_storage variable = e_aggr_strms type = fifo impl = srl

#ifndef __SYNTHESIS__
    printf("******************************\n");
    printf("        Group Aggregate\n");
    printf("******************************\n");
#endif

    group_aggregate_wrapper<8 * TPCH_INT_SZ, n_column, 1, 2, 17, n_channel, n_column, 8 * TPCH_INT_SZ * VEC_LEN, 32,
                            32>(flt_strms, e_flt_strms, shuffle3_cfg_strm, shuffle4_cfg_strm, merge_column_cfg_strm,
                                group_aggr_cfg_strm, result_info_strm, ping_buf0, ping_buf1, ping_buf2, ping_buf3,
                                pong_buf0, pong_buf1, pong_buf2, pong_buf3, aggr_strms, e_aggr_strms);

    write_info<8 * TPCH_INT_SZ, 4>(result_info_strm, buf_result_info);

#ifndef __SYNTHESIS__
    {
        printf("after group aggr\n");
        printf("nrow=%ld\n", aggr_strms[0].size());
    }

    for (int ch = 0; ch < n_channel; ++ch) {
        for (int c = 0; c < 4; ++c) {
            size_t s = flt_strms[ch][c].size();
            if (s != 0) {
                printf("##### flt_strms[%d][%d] has %ld data left after group aggr.\n", ch, c, s);
            }
        }
    }
#endif

    //------------------------direct aggr--------------------------

    hls::stream<ap_uint<32> > direct_aggr_strm[2 * n_column];
#pragma HLS stream variable = direct_aggr_strm depth = 8
#pragma HLS bind_storage variable = direct_aggr_strm type = fifo impl = srl
    hls::stream<bool> e_direct_aggr_strm;
#pragma HLS stream variable = e_direct_aggr_strm depth = 8

#ifndef __SYNTHESIS__
    printf("******************************\n");
    printf("        Direct aggregate\n");
    printf("******************************\n");
#endif

    agg_wrapper<2 * n_column>(direct_aggr_cfg_strm, aggr_strms, e_aggr_strms, direct_aggr_strm, e_direct_aggr_strm);

#ifndef __SYNTHESIS__
    {
        printf("after direct aggr\n");
        printf("nrow=%ld\n", direct_aggr_strm[0].size());
    }

    {
        size_t s = aggr_strms[0].size();
        for (int c = 0; c < 4; ++c) {
            size_t ss = aggr_strms[c].size();
            if (s != ss) {
                printf("##### aggr_strm[0] has %ld row, agg_strm[%d] has %ld row.\n", s, c, ss);
            }
        }
    }
#endif

//------------------------write out--------------------------

#ifndef __SYNTHESIS__
    printf("******************************\n");
    printf("          Write out\n");
    printf("******************************\n");
#endif

    writeTableAggr<BURST_LEN, 8 * TPCH_INT_SZ, VEC_LEN, 2 * n_column>(
        direct_aggr_strm, e_direct_aggr_strm, buf_out0, buf_out1, buf_out2, buf_out3, buf_out4, buf_out5, buf_out6,
        buf_out7, buf_out8, buf_out9, buf_out10, buf_out11, buf_out12, buf_out13, buf_out14, buf_out15, buf_metaout,
        write_cfg_strm);

#ifndef __SYNTHESIS__

#ifdef XDEBUG
    int size512 = buf_out[0].range(63, 32);
    int rowNum = buf_out[0].range(31, 0);
    ap_uint<32>* col0 = (ap_uint<32>*)(buf_out + size512 * 0);
    ap_uint<32>* col1 = (ap_uint<32>*)(buf_out + size512 * 1);
    ap_uint<32>* col2 = (ap_uint<32>*)(buf_out + size512 * 2);
    ap_uint<32>* col3 = (ap_uint<32>*)(buf_out + size512 * 3);
    ap_uint<32>* col4 = (ap_uint<32>*)(buf_out + size512 * 4);
    ap_uint<32>* col5 = (ap_uint<32>*)(buf_out + size512 * 5);
    ap_uint<32>* col6 = (ap_uint<32>*)(buf_out + size512 * 6);
    ap_uint<32>* col7 = (ap_uint<32>*)(buf_out + size512 * 7);
    ap_uint<32>* col8 = (ap_uint<32>*)(buf_out + size512 * 8);
    ap_uint<32>* col9 = (ap_uint<32>*)(buf_out + size512 * 9);
    ap_uint<32>* col10 = (ap_uint<32>*)(buf_out + size512 * 10);
    ap_uint<32>* col11 = (ap_uint<32>*)(buf_out + size512 * 11);
    ap_uint<32>* col12 = (ap_uint<32>*)(buf_out + size512 * 12);
    ap_uint<32>* col13 = (ap_uint<32>*)(buf_out + size512 * 13);
    ap_uint<32>* col14 = (ap_uint<32>*)(buf_out + size512 * 14);
    ap_uint<32>* col15 = (ap_uint<32>*)(buf_out + size512 * 15);

    std::cout << "result size512: " << size512 << std::endl;
    std::cout << "result rowNum:  " << rowNum << std::endl;
    std::cout << "****" << std::endl;

    for (int i = 16; i < 26; i++) {
        std::cout << "col1: " << col0[i] << " ";
        std::cout << "col2: " << col1[i] << " ";
        std::cout << "col3: " << col2[i] << " ";
        std::cout << "col4: " << col3[i] << " ";
        std::cout << "col5: " << col4[i] << " ";
        std::cout << "col6: " << col5[i] << " ";
        std::cout << "col7: " << col6[i] << " ";
        std::cout << "col8: " << col7[i] << std::endl;
    }
    for (int i = 16; i < 26; i++) {
        std::cout << "col9: " << col8[i] << " ";
        std::cout << "col10: " << col9[i] << " ";
        std::cout << "col11: " << col10[i] << " ";
        std::cout << "col12: " << col11[i] << " ";
        std::cout << "col13: " << col12[i] << " ";
        std::cout << "col14: " << col13[i] << " ";
        std::cout << "col15: " << col14[i] << " ";
        std::cout << "col16: " << col15[i] << std::endl;
    }

    std::cout << "buf_in:" << buf_in[0] << std::endl;
    std::cout << "buf_out:" << buf_out[0] << std::endl;
    std::cout << "cfg_in:" << buf_in[0] << std::endl;
    std::cout << "cfg_out:" << buf_in[0] << std::endl;

    std::cout << "buf_ping0:" << ping_buf0[0] << std::endl;
    std::cout << "buf_pong0:" << pong_buf0[0] << std::endl;
    std::cout << "buf_ping1:" << ping_buf0[1] << std::endl;
    std::cout << "buf_pong1:" << pong_buf0[1] << std::endl;
    std::cout << "buf_ping2:" << ping_buf0[2] << std::endl;
    std::cout << "buf_pong2:" << pong_buf0[2] << std::endl;
    std::cout << "buf_ping3:" << ping_buf0[3] << std::endl;
    std::cout << "buf_pong3:" << pong_buf0[3] << std::endl;
#endif

    for (int c = 0; c < 4; ++c) {
        size_t s = aggr_strms[c].size();
        if (s != 0) {
            printf("##### agg_strm[%d] has %ld data left after write-out.\n", c, s);
        }
    }

    std::cout << "kernel end..." << std::endl;
#endif
}
