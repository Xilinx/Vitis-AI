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

#include "xf_database/gqe_kernel_part_v2.hpp"

#include "xf_database/gqe_blocks_v2/scan_col_part.hpp"
#include "xf_database/gqe_blocks/filter_part.hpp"
#include "xf_database/gqe_blocks_v2/hash_partition.hpp"
#include "xf_database/gqe_blocks_v2/write_out_part_aggr.hpp"

namespace xf {
namespace database {
namespace gqe {

// read in kernel cfg:
// - the nrow from tin_meta
// - the filter cfg from cfg
// - use dual key? mk_on
// - hash join is enable or bypass? join_on
// - which input col is valid
// - which output col is valid, notice that for partition kernel, output cols are the same as input, not controlled by
// write_out_cfg.
void load_config(ap_uint<8 * TPCH_INT_SZ * VEC_LEN> ptr[9],
                 ap_uint<512>* tin_meta,
                 const int col_index,
                 const int bit_num,
                 const int k_depth,
                 hls::stream<bool>& mk_on_strm,
                 hls::stream<int>& k_depth_strm,
                 hls::stream<int>& nrow_strm,
                 hls::stream<int8_t>& col_id_strm,
                 hls::stream<ap_uint<32> >& wr_cfg_strm,
                 hls::stream<ap_uint<32> >& filter_cfg_strm,
                 hls::stream<int>& bit_num_strm,
                 hls::stream<int>& bit_num_strm_copy) {
    // read in the number of rows for each column
    int nrowA = tin_meta[0].range(71, 8);
    nrow_strm.write(nrowA);

    bit_num_strm.write(bit_num);
    bit_num_strm_copy.write(bit_num);

    const int filter_cfg_depth = 45;

    ap_uint<8 * TPCH_INT_SZ * VEC_LEN> config[9];
#pragma HLS bind_storage variable = config type = ram_1p impl = lutram

    ap_uint<32> filter_cfg_a[filter_cfg_depth];
#pragma HLS bind_storage variable = filter_cfg_a type = ram_1p impl = lutram

    for (int i = 0; i < 9; i++) {
#pragma HLS PIPELINE II = 1
        config[i] = ptr[i];
    }

    bool mk_on = config[0][2] ? true : false;
    mk_on_strm.write(mk_on);
#ifndef __SYNTHESIS__
    if (mk_on)
        std::cout << "\nDual key is on\n";
    else
        std::cout << "\nDual key is off\n";
#endif

    k_depth_strm.write(k_depth);

    ap_uint<32> write_out_cfg;
    for (int i = 0; i < COL_NUM; i++) {
#pragma HLS PIPELINE II = 1
        int8_t t = config[0].range(64 * col_index + 56 + 8 * i + 7, 64 * col_index + 56 + 8 * i);
        col_id_strm.write(t);
        write_out_cfg[i] = (t >= 0) ? 1 : 0;
    }

    wr_cfg_strm.write(write_out_cfg);

    for (int i = 0; i < filter_cfg_depth; i++) {
        filter_cfg_a[i] = config[3 * col_index + 3 + i / 16].range(32 * ((i % 16) + 1) - 1, 32 * (i % 16));
        filter_cfg_strm.write(filter_cfg_a[i]);
    }
}

/* XXX if dual key, shift the payload,
 * so that for the 3rd col of A table becomes 1st payload
 */
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

template <int WD, int VEC, int NCH, int NCOL>
void load_scan_wrapper(ap_uint<8 * TPCH_INT_SZ * VEC_LEN> buf_D[9],
                       ap_uint<512>* tin_meta,
                       const int col_index,
                       const int bit_num,
                       const int k_depth,
                       hls::stream<bool>& mk_on_strm,
                       hls::stream<int>& k_depth_strm,
                       hls::stream<ap_uint<32> >& wr_cfg_strm,
                       hls::stream<ap_uint<32> >& filter_cfg_strm,
                       const ap_uint<WD * VEC>* buf_A1,
                       const ap_uint<WD * VEC>* buf_A2,
                       const ap_uint<WD * VEC>* buf_A3,
                       const ap_uint<WD * VEC>* buf_A4,
                       const ap_uint<WD * VEC>* buf_A5,
                       const ap_uint<WD * VEC>* buf_A6,
                       const ap_uint<WD * VEC>* buf_A7,
                       const ap_uint<WD * VEC>* buf_A8,
                       hls::stream<ap_uint<WD> > ch_strms[NCH][NCOL],
                       hls::stream<bool> e_ch_strms[NCH],
                       hls::stream<int>& bit_num_strm,
                       hls::stream<int>& bit_num_strm_copy) {
    hls::stream<int> nrow_strm;
#pragma HLS stream variable = nrow_strm depth = 2
    hls::stream<int8_t> cid_strm;
#pragma HLS stream variable = cid_strm depth = 16

    load_config(buf_D, tin_meta, col_index, bit_num, k_depth, mk_on_strm, k_depth_strm, nrow_strm, cid_strm,
                wr_cfg_strm, filter_cfg_strm, bit_num_strm, bit_num_strm_copy);

    scan_cols<8 * TPCH_INT_SZ, VEC_SCAN, CH_NUM, COL_NUM>(buf_A1, buf_A2, buf_A3, buf_A4, buf_A5, buf_A6, buf_A7,
                                                          buf_A8, nrow_strm, cid_strm, ch_strms, e_ch_strms);
}
template <int COL_IN_NM, int CH_NM, int COL_OUT_NM>
void hash_partition_wrapper(hls::stream<bool>& mk_on_strm,
                            hls::stream<int>& k_depth_strm,
                            hls::stream<int>& bit_num_strm,

                            hls::stream<ap_uint<8 * TPCH_INT_SZ> > in_strm[CH_NM][COL_IN_NM],
                            hls::stream<bool> e_in_strm[CH_NM],

                            hls::stream<ap_uint<12> >& o_bkpu_num_strm,
                            hls::stream<ap_uint<10> >& o_nm_strm,
                            hls::stream<ap_uint<8 * TPCH_INT_SZ * PU> > out_strm[COL_OUT_NM]) {
#pragma HLS dataflow

    hls::stream<ap_uint<8 * TPCH_INT_SZ * 2> > key_strm[CH_NM];
#pragma HLS stream variable = key_strm depth = 16

    hls::stream<ap_uint<8 * TPCH_INT_SZ * 6> > pld_strm[CH_NM];
#pragma HLS stream variable = pld_strm depth = 16

    hls::stream<bool> e_strm[CH_NM];
#pragma HLS stream variable = e_strm depth = 16

    // let each channel adapt independently
    for (int ch = 0; ch < CH_NM; ++ch) {
#pragma HLS unroll
        hash_partition_channel_adapter<COL_IN_NM, 6>(in_strm[ch], e_in_strm[ch], key_strm[ch], pld_strm[ch],
                                                     e_strm[ch]);
    }

#ifndef __SYNTHESIS__
    printf("After adapt for hash partition\n");
    for (int ch = 0; ch < CH_NM; ++ch) {
        printf("ch:%d, key nrow = %ld, pld nrow = %ld\n", ch, key_strm[ch].size(), pld_strm[ch].size());
    }
#endif

    xf::database::hashPartition<1, 64, 192, 32, HASHWH, HASHWL, 18, CH_NM, COL_OUT_NM>(
        mk_on_strm, k_depth_strm, bit_num_strm, key_strm, pld_strm, e_strm, o_bkpu_num_strm, o_nm_strm, out_strm);
}

} // namespace gqe
} // namespace database
} // namespace xf

/**
 * @breif GQE partition kernel (32-bit key version)
 *
 * @param k_depth depth of each hash bucket in URAM
 * @param col_index index of input column
 * @param bit_num number of defined partition, log2(number of partition)
 *
 * @param tin_meta input meta info
 * @param tout_meta output meta info
 *
 * @param buf_A input table buffer
 * @param buf_B output table buffer
 * @param buf_D configuration buffer
 *
 */
extern "C" void gqePart(const int k_depth,
                        const int col_index,
                        const int bit_num,
                        ap_uint<32 * VEC_SCAN> buf_A1[TEST_BUF_DEPTH],
                        ap_uint<32 * VEC_SCAN> buf_A2[TEST_BUF_DEPTH],
                        ap_uint<32 * VEC_SCAN> buf_A3[TEST_BUF_DEPTH],
                        ap_uint<32 * VEC_SCAN> buf_A4[TEST_BUF_DEPTH],
                        ap_uint<32 * VEC_SCAN> buf_A5[TEST_BUF_DEPTH],
                        ap_uint<32 * VEC_SCAN> buf_A6[TEST_BUF_DEPTH],
                        ap_uint<32 * VEC_SCAN> buf_A7[TEST_BUF_DEPTH],
                        ap_uint<32 * VEC_SCAN> buf_A8[TEST_BUF_DEPTH],
                        ap_uint<512> tin_meta[24],
                        ap_uint<512> tout_meta[24],
                        ap_uint<512> buf_B1[TEST_BUF_DEPTH],
                        ap_uint<512> buf_B2[TEST_BUF_DEPTH],
                        ap_uint<512> buf_B3[TEST_BUF_DEPTH],
                        ap_uint<512> buf_B4[TEST_BUF_DEPTH],
                        ap_uint<512> buf_B5[TEST_BUF_DEPTH],
                        ap_uint<512> buf_B6[TEST_BUF_DEPTH],
                        ap_uint<512> buf_B7[TEST_BUF_DEPTH],
                        ap_uint<512> buf_B8[TEST_BUF_DEPTH],

                        ap_uint<512> buf_D[9]) {
// clang-format off
#pragma HLS INTERFACE m_axi offset = slave latency = 64 \
	num_write_outstanding = 1 num_read_outstanding = 16 \
	max_write_burst_length = 8 max_read_burst_length = 64 \
	bundle = gmem1_0 port = buf_A1

#pragma HLS INTERFACE m_axi offset = slave latency = 64 \
	num_write_outstanding = 1 num_read_outstanding = 16 \
	max_write_burst_length = 8 max_read_burst_length = 64 \
	bundle = gmem1_1 port = buf_A2

#pragma HLS INTERFACE m_axi offset = slave latency = 64 \
	num_write_outstanding = 1 num_read_outstanding = 16 \
	max_write_burst_length = 8 max_read_burst_length = 64 \
	bundle = gmem1_2 port = buf_A3

#pragma HLS INTERFACE m_axi offset = slave latency = 64 \
	num_write_outstanding = 1 num_read_outstanding = 16 \
	max_write_burst_length = 8 max_read_burst_length = 64 \
	bundle = gmem1_0 port = buf_A4

#pragma HLS INTERFACE m_axi offset = slave latency = 64 \
	num_write_outstanding = 1 num_read_outstanding = 16 \
	max_write_burst_length = 8 max_read_burst_length = 64 \
	bundle = gmem1_1 port = buf_A5

#pragma HLS INTERFACE m_axi offset = slave latency = 64 \
	num_write_outstanding = 1 num_read_outstanding = 16 \
	max_write_burst_length = 8 max_read_burst_length = 64 \
	bundle = gmem1_2 port = buf_A6

#pragma HLS INTERFACE m_axi offset = slave latency = 64 \
	num_write_outstanding = 1 num_read_outstanding = 16 \
	max_write_burst_length = 8 max_read_burst_length = 64 \
	bundle = gmem1_0 port = buf_A7

#pragma HLS INTERFACE m_axi offset = slave latency = 64 \
	num_write_outstanding = 1 num_read_outstanding = 16 \
	max_write_burst_length = 8 max_read_burst_length = 64 \
	bundle = gmem1_1 port = buf_A8

#pragma HLS INTERFACE m_axi offset = slave latency = 64 \
	num_write_outstanding = 1 num_read_outstanding = 16 \
	max_write_burst_length = 8 max_read_burst_length = 64 \
	bundle = gmem1_2 port = tin_meta

#pragma HLS INTERFACE m_axi offset = slave latency = 64 \
	num_write_outstanding = 16 num_read_outstanding = 1 \
	max_write_burst_length = 64 max_read_burst_length = 8 \
	bundle = gmem0_3 port = tout_meta

#pragma HLS INTERFACE m_axi offset = slave latency = 64 \
	num_write_outstanding = 16 num_read_outstanding = 1 \
	max_write_burst_length = 64 max_read_burst_length = 8 \
	bundle = gmem0_1 port = buf_B1

#pragma HLS INTERFACE m_axi offset = slave latency = 64 \
	num_write_outstanding = 16 num_read_outstanding = 1 \
	max_write_burst_length = 64 max_read_burst_length = 8 \
	bundle = gmem0_2 port = buf_B2

#pragma HLS INTERFACE m_axi offset = slave latency = 64 \
	num_write_outstanding = 16 num_read_outstanding = 1 \
	max_write_burst_length = 64 max_read_burst_length = 8 \
	bundle = gmem0_3 port = buf_B3

#pragma HLS INTERFACE m_axi offset = slave latency = 64 \
	num_write_outstanding = 16 num_read_outstanding = 1 \
	max_write_burst_length = 64 max_read_burst_length = 8 \
	bundle = gmem0_1 port = buf_B4

#pragma HLS INTERFACE m_axi offset = slave latency = 64 \
	num_write_outstanding = 16 num_read_outstanding = 1 \
	max_write_burst_length = 64 max_read_burst_length = 8 \
	bundle = gmem0_2 port = buf_B5

#pragma HLS INTERFACE m_axi offset = slave latency = 64 \
	num_write_outstanding = 16 num_read_outstanding = 1 \
	max_write_burst_length = 64 max_read_burst_length = 8 \
	bundle = gmem0_3 port = buf_B6

#pragma HLS INTERFACE m_axi offset = slave latency = 64 \
	num_write_outstanding = 16 num_read_outstanding = 1 \
	max_write_burst_length = 64 max_read_burst_length = 8 \
	bundle = gmem0_1 port = buf_B7

#pragma HLS INTERFACE m_axi offset = slave latency = 64 \
	num_write_outstanding = 16 num_read_outstanding = 1 \
	max_write_burst_length = 64 max_read_burst_length = 8 \
	bundle = gmem0_2 port = buf_B8

#pragma HLS INTERFACE m_axi offset = slave latency = 64 \
	num_write_outstanding = 1 num_read_outstanding = 16 \
	max_write_burst_length = 8 max_read_burst_length = 64 \
	bundle = gmem1_2 port = buf_D

#pragma HLS INTERFACE s_axilite port = k_depth bundle = control
#pragma HLS INTERFACE s_axilite port = col_index bundle = control
#pragma HLS INTERFACE s_axilite port = bit_num bundle = control
#pragma HLS INTERFACE s_axilite port = buf_A1 bundle = control
#pragma HLS INTERFACE s_axilite port = buf_A2 bundle = control
#pragma HLS INTERFACE s_axilite port = buf_A3 bundle = control
#pragma HLS INTERFACE s_axilite port = buf_A4 bundle = control
#pragma HLS INTERFACE s_axilite port = buf_A5 bundle = control
#pragma HLS INTERFACE s_axilite port = buf_A6 bundle = control
#pragma HLS INTERFACE s_axilite port = buf_A7 bundle = control
#pragma HLS INTERFACE s_axilite port = buf_A8 bundle = control
#pragma HLS INTERFACE s_axilite port = tin_meta bundle = control
#pragma HLS INTERFACE s_axilite port = tout_meta bundle = control
#pragma HLS INTERFACE s_axilite port = buf_B1 bundle = control
#pragma HLS INTERFACE s_axilite port = buf_B2 bundle = control
#pragma HLS INTERFACE s_axilite port = buf_B3 bundle = control
#pragma HLS INTERFACE s_axilite port = buf_B4 bundle = control
#pragma HLS INTERFACE s_axilite port = buf_B5 bundle = control
#pragma HLS INTERFACE s_axilite port = buf_B6 bundle = control
#pragma HLS INTERFACE s_axilite port = buf_B7 bundle = control
#pragma HLS INTERFACE s_axilite port = buf_B8 bundle = control
#pragma HLS INTERFACE s_axilite port = buf_D bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

    // clang-format on
    using namespace xf::database::gqe;

#pragma HLS DATAFLOW

    bool mk_on;

    hls::stream<ap_uint<32> > wr_cfg_strm;
#pragma HLS stream variable = wr_cfg_strm depth = 8

    // scan part
    hls::stream<ap_uint<8 * TPCH_INT_SZ> > ch_strms[CH_NUM][COL_NUM];
#pragma HLS stream variable = ch_strms depth = 32
    hls::stream<bool> e_ch_strms[CH_NUM];
#pragma HLS stream variable = e_ch_strms depth = 32

    hls::stream<bool> mk_on_strm;
#pragma HLS stream variable = mk_on_strm depth = 2

    hls::stream<int> bit_num_strm;
#pragma HLS stream variable = bit_num_strm depth = 2
    hls::stream<int> bit_num_strm_copy;
#pragma HLS stream variable = bit_num_strm_copy depth = 2
    hls::stream<int> k_depth_strm;
#pragma HLS stream variable = k_depth_strm depth = 2

    hls::stream<ap_uint<32> > fcfg; // only 45x32bits
#pragma HLS stream variable = fcfg depth = 48
#pragma HLS bind_storage variable = fcfg type = fifo impl = lutram

    // filter part
    hls::stream<ap_uint<8 * TPCH_INT_SZ> > flt_strms[CH_NUM][COL_NUM];
#pragma HLS stream variable = flt_strms depth = 32
#pragma HLS bind_storage variable = flt_strms type = fifo impl = srl
    hls::stream<bool> e_flt_strms[CH_NUM];
#pragma HLS stream variable = e_flt_strms depth = 32

    // partition part
    hls::stream<ap_uint<12> > hp_bkpu_strm;
#pragma HLS stream variable = hp_bkpu_strm depth = 32

    hls::stream<ap_uint<10> > hp_nm_strm;
#pragma HLS stream variable = hp_nm_strm depth = 32

    hls::stream<ap_uint<8 * TPCH_INT_SZ * PU> > hp_out_strms[COL_NUM];
#pragma HLS stream variable = hp_out_strms depth = 256
#pragma HLS bind_storage variable = hp_out_strms type = fifo impl = bram

    load_scan_wrapper<8 * TPCH_INT_SZ, VEC_SCAN, CH_NUM, COL_NUM>(
        buf_D, tin_meta, col_index, bit_num, k_depth, mk_on_strm, k_depth_strm, wr_cfg_strm, fcfg, buf_A1, buf_A2,
        buf_A3, buf_A4, buf_A5, buf_A6, buf_A7, buf_A8, ch_strms, e_ch_strms, bit_num_strm, bit_num_strm_copy);

#ifndef __SYNTHESIS__
    printf("***** after scan\n");
    for (int ch = 0; ch < CH_NUM; ++ch) {
        printf("ch:%d nrow=%ld\n", ch, e_ch_strms[ch].size() - 1);
    }
#endif

    filter_ongoing<COL_NUM, COL_NUM, CH_NUM>(fcfg, ch_strms, e_ch_strms, flt_strms, e_flt_strms);

#ifndef __SYNTHESIS__
    {
        printf("***** after Filter\n");
        size_t ss = 0;
        for (int ch = 0; ch < CH_NUM; ++ch) {
            size_t s = e_flt_strms[ch].size() - 1;
            printf("ch:%d nrow=%ld\n", ch, s);
            ss += s;
        }
        printf("total: nrow=%ld\n", ss);
    }
#endif

    hash_partition_wrapper<COL_NUM, CH_NUM, COL_NUM>(mk_on_strm, k_depth_strm, bit_num_strm, flt_strms, e_flt_strms,
                                                     hp_bkpu_strm, hp_nm_strm, hp_out_strms);

    writeTablePartWrapper<32, VEC_LEN, COL_NUM, HASHWH>(hp_out_strms, wr_cfg_strm, bit_num_strm_copy, hp_nm_strm,
                                                        hp_bkpu_strm, buf_B1, buf_B2, buf_B3, buf_B4, buf_B5, buf_B6,
                                                        buf_B7, buf_B8, tout_meta);
}
