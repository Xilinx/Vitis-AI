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

#include "xf_database/gqe_blocks_v3/gqe_types.hpp"

#include "xf_database/gqe_blocks/stream_helper.hpp"

#include "xf_database/gqe_blocks_v3/load_config.hpp"
#include "xf_database/gqe_blocks_v3/scan_cols.hpp"

#include "xf_database/gqe_blocks_v3/gqe_filter.hpp"
#include "xf_database/gqe_blocks_v3/hash_join.hpp"

#include "xf_database/gqe_blocks_v3/write_out.hpp"

#include <ap_int.h>
#include <hls_stream.h>

namespace xf {
namespace database {
namespace gqe {

// load kernel config and scan cols in
template <int CH_NM, int COL_NM, int BLEN>
void load_cfg_and_scan(bool build_probe_flag,
                       ap_uint<512> din_krn_cfg[14],
                       ap_uint<512> din_meta[24],
                       hls::burst_maxi<ap_uint<8 * TPCH_INT_SZ * VEC_SCAN> > din_col0,
                       hls::burst_maxi<ap_uint<8 * TPCH_INT_SZ * VEC_SCAN> > din_col1,
                       hls::burst_maxi<ap_uint<8 * TPCH_INT_SZ * VEC_SCAN> > din_col2,
                       hls::burst_maxi<ap_uint<64> > din_val,
                       hls::stream<ap_uint<6> >& join_cfg_strm,
                       hls::stream<ap_uint<36> >& bf_cfg_strm,
                       hls::stream<ap_uint<32> >& filter_cfg_strm,
                       hls::stream<ap_uint<8 * TPCH_INT_SZ> > ch_strms[CH_NM][COL_NM],
                       hls::stream<bool> e_ch_strms[CH_NM],
                       hls::stream<ap_uint<8> >& write_out_cfg_strm) {
    int64_t nrow;
    int secID;
    ap_uint<3> din_col_en;
    ap_uint<2> rowID_flags;

    load_config(build_probe_flag, din_krn_cfg, din_meta, nrow, secID, join_cfg_strm, bf_cfg_strm, filter_cfg_strm,
                din_col_en, rowID_flags, write_out_cfg_strm);

    scan_cols<CH_NM, COL_NM, BLEN>(rowID_flags, nrow, secID, din_col_en, din_col0, din_col1, din_col2, din_val,
                                   ch_strms, e_ch_strms);
}

} // namespace gqe
} // namespace database
} // namespace xf

extern "C" void gqeJoin(size_t _build_probe_flag, // build/probe flag

                        // input data columns
                        hls::burst_maxi<ap_uint<8 * TPCH_INT_SZ * VEC_SCAN> > din_col0,
                        hls::burst_maxi<ap_uint<8 * TPCH_INT_SZ * VEC_SCAN> > din_col1,
                        hls::burst_maxi<ap_uint<8 * TPCH_INT_SZ * VEC_SCAN> > din_col2,

                        // validation buffer
                        hls::burst_maxi<ap_uint<64> > din_val,

                        // kernel config
                        ap_uint<512> din_krn_cfg[14],

                        // meta input buffer
                        ap_uint<512> din_meta[24],
                        // meta output buffer
                        ap_uint<512> dout_meta[24],

                        //  output data columns
                        ap_uint<8 * TPCH_INT_SZ * VEC_LEN>* dout_col0,
                        ap_uint<8 * TPCH_INT_SZ * VEC_LEN>* dout_col1,
                        ap_uint<8 * TPCH_INT_SZ * VEC_LEN>* dout_col2,
                        ap_uint<8 * TPCH_INT_SZ * VEC_LEN>* dout_col3,

                        // hbm buffers used to save build table key/payload
                        ap_uint<256>* htb_buf0,
                        ap_uint<256>* htb_buf1,
                        ap_uint<256>* htb_buf2,
                        ap_uint<256>* htb_buf3,
                        ap_uint<256>* htb_buf4,
                        ap_uint<256>* htb_buf5,
                        ap_uint<256>* htb_buf6,
                        ap_uint<256>* htb_buf7,

                        // overflow buffers
                        ap_uint<256>* stb_buf0,
                        ap_uint<256>* stb_buf1,
                        ap_uint<256>* stb_buf2,
                        ap_uint<256>* stb_buf3,
                        ap_uint<256>* stb_buf4,
                        ap_uint<256>* stb_buf5,
                        ap_uint<256>* stb_buf6,
                        ap_uint<256>* stb_buf7) {
#pragma HLS INTERFACE m_axi offset = slave latency = 64 num_write_outstanding = 1 num_read_outstanding = \
    8 max_write_burst_length = 2 max_read_burst_length = 64 bundle = gmem0_0 port = din_col0

#pragma HLS INTERFACE m_axi offset = slave latency = 64 num_write_outstanding = 1 num_read_outstanding = \
    8 max_write_burst_length = 2 max_read_burst_length = 64 bundle = gmem0_1 port = din_col1

#pragma HLS INTERFACE m_axi offset = slave latency = 64 num_write_outstanding = 1 num_read_outstanding = \
    8 max_write_burst_length = 2 max_read_burst_length = 64 bundle = gmem0_2 port = din_col2

#pragma HLS INTERFACE m_axi offset = slave latency = 64 num_write_outstanding = 1 num_read_outstanding = \
    8 max_write_burst_length = 2 max_read_burst_length = 64 bundle = gmem0_3 port = din_krn_cfg

#pragma HLS INTERFACE m_axi offset = slave latency = 64 num_write_outstanding = 1 num_read_outstanding = \
    8 max_write_burst_length = 2 max_read_burst_length = 64 bundle = gmem0_3 port = din_meta

#pragma HLS INTERFACE m_axi offset = slave latency = 64 num_write_outstanding = 8 num_read_outstanding = \
    1 max_write_burst_length = 64 max_read_burst_length = 2 bundle = gmem1_0 port = dout_meta

#pragma HLS INTERFACE m_axi offset = slave latency = 64 num_write_outstanding = 8 num_read_outstanding = \
    1 max_write_burst_length = 64 max_read_burst_length = 2 bundle = gmem1_0 port = dout_col0

#pragma HLS INTERFACE m_axi offset = slave latency = 64 num_write_outstanding = 8 num_read_outstanding = \
    1 max_write_burst_length = 64 max_read_burst_length = 2 bundle = gmem1_0 port = dout_col1

#pragma HLS INTERFACE m_axi offset = slave latency = 64 num_write_outstanding = 8 num_read_outstanding = \
    1 max_write_burst_length = 64 max_read_burst_length = 2 bundle = gmem1_0 port = dout_col2

#pragma HLS INTERFACE m_axi offset = slave latency = 64 num_write_outstanding = 8 num_read_outstanding = \
    1 max_write_burst_length = 64 max_read_burst_length = 2 bundle = gmem1_0 port = dout_col3

#pragma HLS INTERFACE m_axi offset = slave latency = 256 num_write_outstanding = 256 num_read_outstanding = \
    256 max_write_burst_length = 2 max_read_burst_length = 2 bundle = gmem2_0 port = htb_buf0

#pragma HLS INTERFACE m_axi offset = slave latency = 256 num_write_outstanding = 256 num_read_outstanding = \
    256 max_write_burst_length = 2 max_read_burst_length = 2 bundle = gmem2_1 port = htb_buf1

#pragma HLS INTERFACE m_axi offset = slave latency = 256 num_write_outstanding = 256 num_read_outstanding = \
    256 max_write_burst_length = 2 max_read_burst_length = 2 bundle = gmem2_2 port = htb_buf2

#pragma HLS INTERFACE m_axi offset = slave latency = 256 num_write_outstanding = 256 num_read_outstanding = \
    256 max_write_burst_length = 2 max_read_burst_length = 2 bundle = gmem2_3 port = htb_buf3

#pragma HLS INTERFACE m_axi offset = slave latency = 256 num_write_outstanding = 256 num_read_outstanding = \
    256 max_write_burst_length = 2 max_read_burst_length = 2 bundle = gmem2_4 port = htb_buf4

#pragma HLS INTERFACE m_axi offset = slave latency = 256 num_write_outstanding = 256 num_read_outstanding = \
    256 max_write_burst_length = 2 max_read_burst_length = 2 bundle = gmem2_5 port = htb_buf5

#pragma HLS INTERFACE m_axi offset = slave latency = 256 num_write_outstanding = 256 num_read_outstanding = \
    256 max_write_burst_length = 2 max_read_burst_length = 2 bundle = gmem2_6 port = htb_buf6

#pragma HLS INTERFACE m_axi offset = slave latency = 256 num_write_outstanding = 256 num_read_outstanding = \
    256 max_write_burst_length = 2 max_read_burst_length = 2 bundle = gmem2_7 port = htb_buf7

#pragma HLS INTERFACE m_axi offset = slave latency = 256 num_write_outstanding = 256 num_read_outstanding = \
    16 max_write_burst_length = 2 max_read_burst_length = 2 bundle = gmem3_0 port = stb_buf0

#pragma HLS INTERFACE m_axi offset = slave latency = 256 num_write_outstanding = 256 num_read_outstanding = \
    16 max_write_burst_length = 2 max_read_burst_length = 2 bundle = gmem3_1 port = stb_buf1

#pragma HLS INTERFACE m_axi offset = slave latency = 256 num_write_outstanding = 256 num_read_outstanding = \
    16 max_write_burst_length = 2 max_read_burst_length = 2 bundle = gmem3_2 port = stb_buf2

#pragma HLS INTERFACE m_axi offset = slave latency = 256 num_write_outstanding = 256 num_read_outstanding = \
    16 max_write_burst_length = 2 max_read_burst_length = 2 bundle = gmem3_3 port = stb_buf3

#pragma HLS INTERFACE m_axi offset = slave latency = 256 num_write_outstanding = 256 num_read_outstanding = \
    16 max_write_burst_length = 2 max_read_burst_length = 2 bundle = gmem3_4 port = stb_buf4

#pragma HLS INTERFACE m_axi offset = slave latency = 256 num_write_outstanding = 256 num_read_outstanding = \
    16 max_write_burst_length = 2 max_read_burst_length = 2 bundle = gmem3_5 port = stb_buf5

#pragma HLS INTERFACE m_axi offset = slave latency = 256 num_write_outstanding = 256 num_read_outstanding = \
    16 max_write_burst_length = 2 max_read_burst_length = 2 bundle = gmem3_6 port = stb_buf6

#pragma HLS INTERFACE m_axi offset = slave latency = 256 num_write_outstanding = 256 num_read_outstanding = \
    16 max_write_burst_length = 2 max_read_burst_length = 2 bundle = gmem3_7 port = stb_buf7

#pragma HLS INTERFACE s_axilite port = _build_probe_flag bundle = control

#pragma HLS INTERFACE s_axilite port = din_col0 bundle = control
#pragma HLS INTERFACE s_axilite port = din_col1 bundle = control
#pragma HLS INTERFACE s_axilite port = din_col2 bundle = control
#pragma HLS INTERFACE s_axilite port = din_val bundle = control

#pragma HLS INTERFACE s_axilite port = din_krn_cfg bundle = control
#pragma HLS INTERFACE s_axilite port = din_meta bundle = control
#pragma HLS INTERFACE s_axilite port = dout_meta bundle = control

#pragma HLS INTERFACE s_axilite port = dout_col0 bundle = control
#pragma HLS INTERFACE s_axilite port = dout_col1 bundle = control
#pragma HLS INTERFACE s_axilite port = dout_col2 bundle = control
#pragma HLS INTERFACE s_axilite port = dout_col3 bundle = control

#pragma HLS INTERFACE s_axilite port = htb_buf0 bundle = control
#pragma HLS INTERFACE s_axilite port = htb_buf1 bundle = control
#pragma HLS INTERFACE s_axilite port = htb_buf2 bundle = control
#pragma HLS INTERFACE s_axilite port = htb_buf3 bundle = control
#pragma HLS INTERFACE s_axilite port = htb_buf4 bundle = control
#pragma HLS INTERFACE s_axilite port = htb_buf5 bundle = control
#pragma HLS INTERFACE s_axilite port = htb_buf6 bundle = control
#pragma HLS INTERFACE s_axilite port = htb_buf7 bundle = control

#pragma HLS INTERFACE s_axilite port = stb_buf0 bundle = control
#pragma HLS INTERFACE s_axilite port = stb_buf1 bundle = control
#pragma HLS INTERFACE s_axilite port = stb_buf2 bundle = control
#pragma HLS INTERFACE s_axilite port = stb_buf3 bundle = control
#pragma HLS INTERFACE s_axilite port = stb_buf4 bundle = control
#pragma HLS INTERFACE s_axilite port = stb_buf5 bundle = control
#pragma HLS INTERFACE s_axilite port = stb_buf6 bundle = control
#pragma HLS INTERFACE s_axilite port = stb_buf7 bundle = control

#pragma HLS INTERFACE s_axilite port = return bundle = control

#ifndef __SYNTHESIS__
    std::cout << "in gqeJOIN kernel ..............." << std::endl;
#endif

    using namespace xf::database::gqe;

#pragma HLS dataflow

    const int nch = 4;
    const int ncol = 3;
    const int ncol_out = 4;
    const int burstlen = 32;

    hls::stream<ap_uint<8 * TPCH_INT_SZ> > ch_strms[nch][ncol];
#pragma HLS stream variable = ch_strms depth = 32
#pragma HLS resource variable = ch_strms core = FIFO_LUTRAM

    hls::stream<bool> e_ch_strms[nch];
#pragma HLS stream variable = e_ch_strms depth = 32

    hls::stream<ap_uint<32> > filter_cfg_strm;
#pragma HLS stream variable = filter_cfg_strm depth = 128
#pragma HLS resource variable = filter_cfg_strm core = FIFO_LUTRAM

    hls::stream<ap_uint<8> > write_out_cfg_strm;
#pragma HLS stream variable = write_out_cfg_strm depth = 2

#ifndef __SYNTHESIS__
    printf("************************************************************\n");
    printf("             General Query Egnine JOIN Kernel\n");
    printf("************************************************************\n");
#endif

#ifndef __SYNTHESIS__
    std::cout << "===================meta buffer ===================" << std::endl;
    std::cout << "din_meta.vnum is " << din_meta[0].range(7, 0) << std::endl;
    std::cout << "din_meta.length is " << din_meta[0].range(71, 8) << std::endl;
    std::cout << std::endl;
#endif

    bool build_probe_flag = (bool)(_build_probe_flag);
#ifndef __SYNTHESIS__
    std::cout << "build_probe_flag= " << build_probe_flag << std::endl;
#endif

    // define join_cfg to represent the join cfg for each module:
    // bit3-2: join type; bit1: dual key on/off; bit0: join on or bypass
    // bit4: bloom-filter on/off; bit5: probe/build flag
    hls::stream<ap_uint<6> > join_cfg_strm;
#pragma HLS stream variable = join_cfg_strm depth = 2
    hls::stream<ap_uint<36> > bf_cfg_strm;
#pragma HLS stream variable = bf_cfg_strm depth = 2

    load_cfg_and_scan<nch, ncol, burstlen>(build_probe_flag, din_krn_cfg, din_meta, din_col0, din_col1, din_col2,
                                           din_val, join_cfg_strm, bf_cfg_strm, filter_cfg_strm, ch_strms, e_ch_strms,
                                           write_out_cfg_strm);

#ifndef __SYNTHESIS__
    std::cout << "after scan" << std::endl;
    for (int ch = 0; ch < nch; ch++) {
        std::cout << "e_ch_strms[" << ch << "].size(): " << e_ch_strms[ch].size() << std::endl;
        for (int c = 0; c < ncol; c++) {
            std::cout << "ch_strms[" << ch << "][" << c << "].size(): " << ch_strms[ch][c].size() << std::endl;
        }
    }
#endif

    // filtering
    hls::stream<ap_uint<8 * TPCH_INT_SZ> > flt_strms[nch][ncol];
#pragma HLS stream variable = flt_strms depth = 32
#pragma HLS resource variable = flt_strms core = FIFO_LUTRAM
    hls::stream<bool> e_flt_strms[nch];
#pragma HLS stream variable = e_flt_strms depth = 32

    filter_ongoing<3, 3, 4>(filter_cfg_strm, ch_strms, e_ch_strms, flt_strms, e_flt_strms);

#ifndef __SYNTHESIS__
    std::cout << "after filtering" << std::endl;
    for (int ch = 0; ch < nch; ch++) {
        std::cout << "e_flt_strms[" << ch << "].size(): " << e_flt_strms[ch].size() << std::endl;
        for (int c = 0; c < ncol; c++) {
            std::cout << "flt_strms[" << ch << "][" << c << "].size(): " << flt_strms[ch][c].size() << std::endl;
        }
    }
#endif

    hls::stream<ap_uint<6> > join_cfg_demux_strm[2];
#pragma HLS stream variable = join_cfg_demux_strm depth = 2
    hls::stream<ap_uint<8 * TPCH_INT_SZ> > demux_strms_0[nch][ncol];
#pragma HLS stream variable = demux_strms_0 depth = 32
#pragma HLS resource variable = demux_strms_0 core = FIFO_LUTRAM
    hls::stream<bool> e_demux_strms_0[nch];
#pragma HLS stream variable = e_demux_strms_0 depth = 32

    hls::stream<ap_uint<8 * TPCH_INT_SZ> > demux_strms_1[nch][ncol];
#pragma HLS stream variable = demux_strms_1 depth = 32
#pragma HLS resource variable = demux_strms_1 core = FIFO_LUTRAM
    hls::stream<bool> e_demux_strms_1[nch];
#pragma HLS stream variable = e_demux_strms_1 depth = 32

    demux_wrapper<ncol, nch, 1>(join_cfg_strm, flt_strms, e_flt_strms, join_cfg_demux_strm, demux_strms_0,
                                demux_strms_1, e_demux_strms_0, e_demux_strms_1);
#ifndef __SYNTHESIS__
    std::cout << "after demux" << std::endl;
    for (int ch = 0; ch < nch; ch++) {
        std::cout << "e_demux_strms_0[" << ch << "].size(): " << e_demux_strms_0[ch].size() << std::endl;
        for (int c = 0; c < ncol; c++) {
            std::cout << "demux_strms_0[" << ch << "][" << c << "].size(): " << demux_strms_0[ch][c].size()
                      << std::endl;
        }
    }
    for (int ch = 0; ch < nch; ch++) {
        std::cout << "e_demux_strms_1[" << ch << "].size(): " << e_demux_strms_1[ch].size() << std::endl;
        for (int c = 0; c < ncol; c++) {
            std::cout << "demux_strms_1[" << ch << "][" << c << "].size(): " << demux_strms_1[ch][c].size()
                      << std::endl;
        }
    }
#endif

    // hash join
    hls::stream<ap_uint<6> > join_cfg_hj_strm;
#pragma HLS stream variable = join_cfg_hj_strm depth = 2
    hls::stream<ap_uint<64> > jn_strm[ncol_out];
#pragma HLS stream variable = jn_strm depth = 32
#pragma HLS array_partition variable = jn_strm complete
#pragma HLS resource variable = jn_strm core = FIFO_LUTRAM
    hls::stream<bool> e_jn_strm;
#pragma HLS stream variable = e_jn_strm depth = 32

    bloomfilter_join_wrapper<ncol, nch, ncol_out, 1>(
        bf_cfg_strm, join_cfg_demux_strm[1], join_cfg_hj_strm, demux_strms_1, e_demux_strms_1, jn_strm, e_jn_strm,
        htb_buf0, htb_buf1, htb_buf2, htb_buf3, htb_buf4, htb_buf5, htb_buf6, htb_buf7, stb_buf0, stb_buf1, stb_buf2,
        stb_buf3, stb_buf4, stb_buf5, stb_buf6, stb_buf7);

#ifndef __SYNTHESIS__
    std::cout << "passed join" << std::endl;
    std::cout << "e_jn_strm.size(): " << e_jn_strm.size() << std::endl;
    for (int c = 0; c < 5; c++) {
        std::cout << "jn_strm[c].size(): " << jn_strm[c].size() << std::endl;
    }
#endif

    // channel data when bypass is on
    hls::stream<ap_uint<8 * TPCH_INT_SZ> > jn_bp_strm[ncol];
#pragma HLS stream variable = jn_bp_strm depth = 32
#pragma HLS resource variable = jn_bp_strm core = FIFO_LUTRAM
    hls::stream<bool> e_jn_bp_strm;
#pragma HLS stream variable = e_jn_bp_strm depth = 32

    hash_join_bypass<ncol, nch>(join_cfg_demux_strm[0], demux_strms_0, e_demux_strms_0, jn_bp_strm, e_jn_bp_strm);

#ifndef __SYNTHESIS__
    std::cout << "after bypass" << std::endl;
    std::cout << "e_jn_bp_strm.size(): " << e_jn_bp_strm.size() << std::endl;
    for (int c = 0; c < ncol; c++) {
        std::cout << "jn_bp_strm[" << c << "].size(): " << jn_bp_strm[c].size() << std::endl;
    }
#endif

    // mux the 2-way data, one from hash join, another one fron bypass
    hls::stream<ap_uint<6> > join_cfg_mux_strm;
#pragma HLS stream variable = join_cfg_mux_strm depth = 2
    hls::stream<ap_uint<8 * TPCH_INT_SZ> > jn_mx_strm[ncol_out];
#pragma HLS stream variable = jn_mx_strm depth = 32
#pragma HLS resource variable = jn_mx_strm core = FIFO_LUTRAM
    hls::stream<bool> e_jn_mx_strm;
#pragma HLS stream variable = e_jn_mx_strm depth = 32
    stream1D_mux2To1<8 * TPCH_INT_SZ, ncol, ncol_out>(join_cfg_hj_strm, join_cfg_mux_strm, jn_bp_strm, jn_strm,
                                                      e_jn_bp_strm, e_jn_strm, jn_mx_strm, e_jn_mx_strm);

#ifndef __SYNTHESIS__
    std::cout << "after mux" << std::endl;
    std::cout << "e_jn_mx_strm.size(): " << e_jn_mx_strm.size() << std::endl;
    for (int c = 0; c < ncol; c++) {
        std::cout << "jn_mx_strm[" << c << "].size(): " << jn_mx_strm[c].size() << std::endl;
    }
#endif

    write_table_hj<burstlen, 8 * TPCH_INT_SZ, VEC_LEN, ncol_out>(join_cfg_mux_strm, jn_mx_strm, e_jn_mx_strm,
                                                                 write_out_cfg_strm, dout_col0, dout_col1, dout_col2,
                                                                 dout_col3, dout_meta);
}
