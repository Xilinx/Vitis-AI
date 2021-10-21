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

#include <ap_int.h>
#include <hls_stream.h>

#include "xf_database/gqe_blocks_v3/gqe_types.hpp"

#include "xf_database/gqe_blocks_v3/load_config.hpp"
#include "xf_database/gqe_blocks_v3/scan_cols.hpp"

#include "xf_database/gqe_blocks_v3/gqe_filter.hpp"
#include "xf_database/gqe_blocks_v3/hash_partition.hpp"
#include "xf_database/gqe_blocks_v3/write_out.hpp"

namespace xf {
namespace database {
namespace gqe {

// load kernel config and scan data in
template <int CH_NM, int COL_NM, int BLEN>
void load_cfg_and_scan(bool tab_index,
                       const int log_part,
                       const int bucket_depth,
                       ap_uint<512> din_krn_cfg[14],
                       ap_uint<512> din_meta[24],
                       hls::burst_maxi<ap_uint<8 * TPCH_INT_SZ * VEC_SCAN> > din_col0,
                       hls::burst_maxi<ap_uint<8 * TPCH_INT_SZ * VEC_SCAN> > din_col1,
                       hls::burst_maxi<ap_uint<8 * TPCH_INT_SZ * VEC_SCAN> > din_col2,
                       hls::burst_maxi<ap_uint<64> > din_val,
                       hls::stream<ap_uint<16> >& part_cfg_strm,
                       hls::stream<ap_uint<32> >& filter_cfg_strms,
                       hls::stream<int>& bit_num_strm_copy,
                       hls::stream<ap_uint<8 * TPCH_INT_SZ> > ch_strms[CH_NM][COL_NM],
                       hls::stream<bool> e_ch_strms[CH_NM],
                       hls::stream<ap_uint<8> >& write_out_cfg_strm) {
    int64_t nrow;
    int secID;
    ap_uint<3> din_col_en;
    ap_uint<2> rowID_flags;
    load_config(tab_index, log_part, bucket_depth, din_krn_cfg, din_meta, nrow, secID, part_cfg_strm, din_col_en,
                rowID_flags, filter_cfg_strms, bit_num_strm_copy, write_out_cfg_strm);
    scan_cols<CH_NM, COL_NM, BLEN>(rowID_flags, nrow, secID, din_col_en, din_col0, din_col1, din_col2, din_val,
                                   ch_strms, e_ch_strms);
}

// assign the input strms to key_strm and pld_strm
template <int COL_NM, int PLD_NM>
void hash_partition_channel_adapter(hls::stream<ap_uint<8 * TPCH_INT_SZ> > in_strm[COL_NM],
                                    hls::stream<bool>& e_in_strm,
                                    hls::stream<ap_uint<8 * TPCH_INT_SZ * 2> >& key_strm,
                                    hls::stream<ap_uint<8 * TPCH_INT_SZ * PLD_NM> >& pld_strm,
                                    hls::stream<bool>& e_strm) {
    bool e = e_in_strm.read();
    while (!e) {
#pragma HLS pipeline II = 1

        ap_uint<8 * TPCH_INT_SZ * 2> key_tmp;
        ap_uint<8 * TPCH_INT_SZ * PLD_NM> pld_tmp;

        ap_uint<8 * TPCH_INT_SZ> d_tmp[COL_NM];
#pragma HLS array_partition variable = d_tmp complete
        for (int c = 0; c < COL_NM; ++c) {
#pragma HLS unroll
            d_tmp[c] = in_strm[c].read();
            //#ifndef __SYNTHESIS__
            //            std::cout << "d_tmp[" << c << "]: " << d_tmp[c] << std::endl;
            //#endif
        }

        key_tmp.range(8 * TPCH_INT_SZ - 1, 0) = d_tmp[0];
        key_tmp.range(8 * TPCH_INT_SZ * 2 - 1, 8 * TPCH_INT_SZ) = d_tmp[1];

        for (int c = 0; c < PLD_NM; ++c) {
#pragma HLS unroll
            pld_tmp.range(8 * TPCH_INT_SZ * (c + 1) - 1, 8 * TPCH_INT_SZ * c) = d_tmp[2 + c];
        }

        key_strm.write(key_tmp);
        pld_strm.write(pld_tmp);
        e_strm.write(false);
        e = e_in_strm.read();
    }
    e_strm.write(true);
}

template <int COL_NM, int CH_NM, int PU>
void hash_partition_wrapper(hls::stream<ap_uint<16> >& part_cfg_strm,

                            hls::stream<ap_uint<8 * TPCH_INT_SZ> > in_strm[CH_NM][COL_NM],
                            hls::stream<bool> e_in_strm[CH_NM],

                            hls::stream<ap_uint<12> >& o_bkpu_num_strm,
                            hls::stream<ap_uint<10> >& o_nm_strm,
                            hls::stream<ap_uint<8 * TPCH_INT_SZ * PU> > out_strm[COL_NM]) {
#pragma HLS dataflow

    hls::stream<ap_uint<8 * TPCH_INT_SZ * 2> > key_strm[CH_NM];
#pragma HLS stream variable = key_strm depth = 16

    hls::stream<ap_uint<8 * TPCH_INT_SZ * 1> > pld_strm[CH_NM];
#pragma HLS stream variable = pld_strm depth = 16

    hls::stream<bool> e_strm[CH_NM];
#pragma HLS stream variable = e_strm depth = 16

    // let each channel adapt independently
    for (int ch = 0; ch < CH_NM; ++ch) {
#pragma HLS unroll
        hash_partition_channel_adapter<COL_NM, 1>(in_strm[ch], e_in_strm[ch], key_strm[ch], pld_strm[ch], e_strm[ch]);
    }

#ifndef __SYNTHESIS__
    printf("After adapt for hash partition\n");
    for (int ch = 0; ch < CH_NM; ++ch) {
        printf("ch:%d, key nrow = %ld, pld nrow = %ld\n", ch, key_strm[ch].size(), pld_strm[ch].size());
    }
#endif

    xf::database::hashPartition<1, 128, 64, 64, 2, 8, 18, CH_NM, COL_NM>(part_cfg_strm, key_strm, pld_strm, e_strm,
                                                                         o_bkpu_num_strm, o_nm_strm, out_strm);
}

} // namespace gqe
} // namespace database
} // namespace xf

/**
 * @breif GQE partition kernel (64-bit key version)
 *
 * @param bucket_depth bucket depth
 * @param table_index table index indicating build table or join table
 * @param log_part log of number of partitions
 *
 * @param din_col input table columns
 * @param din_val validation bits column
 * @param din_krn_cfg input kernel configurations
 *
 * @param din_meta input meta info
 * @param dout_meta output meta info
 *
 * @param dout_col output table columns
 *
 */
extern "C" void gqePart(const int bucket_depth, // bucket depth

                        // table index indicate build table or join table
                        const int tab_index,

                        // the log partition number
                        const int log_part,

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

                        // output data columns
                        ap_uint<8 * TPCH_INT_SZ * VEC_LEN>* dout_col0,
                        ap_uint<8 * TPCH_INT_SZ * VEC_LEN>* dout_col1,
                        ap_uint<8 * TPCH_INT_SZ * VEC_LEN>* dout_col2) {
#pragma HLS INTERFACE m_axi offset = slave latency = 64 num_write_outstanding = 1 num_read_outstanding = \
    8 max_write_burst_length = 8 max_read_burst_length = 64 bundle = gmem1_0 port = din_col0

#pragma HLS INTERFACE m_axi offset = slave latency = 64 num_write_outstanding = 1 num_read_outstanding = \
    8 max_write_burst_length = 8 max_read_burst_length = 64 bundle = gmem1_1 port = din_col1

#pragma HLS INTERFACE m_axi offset = slave latency = 64 num_write_outstanding = 1 num_read_outstanding = \
    8 max_write_burst_length = 8 max_read_burst_length = 64 bundle = gmem1_2 port = din_col2

#pragma HLS INTERFACE m_axi offset = slave latency = 64 num_write_outstanding = 1 num_read_outstanding = \
    8 max_write_burst_length = 8 max_read_burst_length = 64 bundle = gmem1_3 port = din_val

#pragma HLS INTERFACE m_axi offset = slave latency = 64 num_write_outstanding = 1 num_read_outstanding = \
    8 max_write_burst_length = 8 max_read_burst_length = 64 bundle = gmem1_4 port = din_krn_cfg

#pragma HLS INTERFACE m_axi offset = slave latency = 64 num_write_outstanding = 1 num_read_outstanding = \
    8 max_write_burst_length = 8 max_read_burst_length = 64 bundle = gmem1_4 port = din_meta

#pragma HLS INTERFACE m_axi offset = slave latency = 64 num_write_outstanding = 8 num_read_outstanding = \
    1 max_write_burst_length = 64 max_read_burst_length = 8 bundle = gmem0_3 port = dout_meta

#pragma HLS INTERFACE m_axi offset = slave latency = 64 num_write_outstanding = 8 num_read_outstanding = \
    1 max_write_burst_length = 64 max_read_burst_length = 8 bundle = gmem0_1 port = dout_col0

#pragma HLS INTERFACE m_axi offset = slave latency = 64 num_write_outstanding = 8 num_read_outstanding = \
    1 max_write_burst_length = 64 max_read_burst_length = 8 bundle = gmem0_2 port = dout_col1

#pragma HLS INTERFACE m_axi offset = slave latency = 64 num_write_outstanding = 8 num_read_outstanding = \
    1 max_write_burst_length = 64 max_read_burst_length = 8 bundle = gmem0_3 port = dout_col2

#pragma HLS INTERFACE s_axilite port = bucket_depth bundle = control
#pragma HLS INTERFACE s_axilite port = tab_index bundle = control
#pragma HLS INTERFACE s_axilite port = log_part bundle = control
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
#pragma HLS INTERFACE s_axilite port = return bundle = control

#ifndef __SYNTHESIS__
    std::cout << std::endl;
    std::cout << "---------------- in gqePart kernel -----------------" << std::endl;
#endif

    using namespace xf::database::gqe;

    const int nch = 4;
    const int ncol = 3;
    const int ncol_out = 3;
    const int nPU = 4;
    const int burstlen = 32;

#pragma HLS DATAFLOW

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
    hls::stream<int> bit_num_strm_copy;
#pragma HLS stream variable = bit_num_strm_copy depth = 2

    hls::stream<ap_uint<16> > part_cfg_strm;
#pragma HLS stream variable = part_cfg_strm depth = 2

    load_cfg_and_scan<nch, ncol, burstlen>(tab_index, log_part, bucket_depth, din_krn_cfg, din_meta, din_col0, din_col1,
                                           din_col2, din_val, part_cfg_strm, filter_cfg_strm, bit_num_strm_copy,
                                           ch_strms, e_ch_strms, write_out_cfg_strm);

#ifndef __SYNTHESIS__
    printf("***** after scan\n");
    for (int ch = 0; ch < nch; ++ch) {
        printf("e_ch:%d nrow=%ld\n", ch, e_ch_strms[ch].size());
        printf("ch:%d nrow=%ld\n", ch, ch_strms[ch][0].size());
    }
#endif

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

    // partition part
    hls::stream<ap_uint<12> > hp_bkpu_strm;
#pragma HLS stream variable = hp_bkpu_strm depth = 32
    hls::stream<ap_uint<10> > hp_nm_strm;
#pragma HLS stream variable = hp_nm_strm depth = 32
    hls::stream<ap_uint<8 * TPCH_INT_SZ * nPU> > hp_out_strms[ncol];
#pragma HLS stream variable = hp_out_strms depth = 256
#pragma HLS resource variable = hp_out_strms core = FIFO_BRAM

    hash_partition_wrapper<ncol, nch, 4>(part_cfg_strm, flt_strms, e_flt_strms, hp_bkpu_strm, hp_nm_strm, hp_out_strms);

    write_table_part<64, VEC_LEN, ncol, 2>(hp_out_strms, write_out_cfg_strm, bit_num_strm_copy, hp_nm_strm,
                                           hp_bkpu_strm, dout_col0, dout_col1, dout_col2, dout_meta);
}
