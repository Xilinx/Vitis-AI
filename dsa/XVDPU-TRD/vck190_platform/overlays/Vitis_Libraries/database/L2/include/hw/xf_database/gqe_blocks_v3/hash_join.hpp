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
#ifndef GQE_ISV_HASH_JOIN_PART_HPP
#define GQE_ISV_HASH_JOIN_PART_HPP

#include <hls_stream.h>
#include <ap_int.h>

#include "xf_database/types.hpp"
#include "xf_database/hash_multi_join_build_probe.hpp"
#include "xf_database/gqe_blocks_v3/bloom_filter_join.hpp"

#include "xf_utils_hw/stream_split.hpp"

#include "xf_database/gqe_blocks/gqe_types.hpp"
#include "xf_database/gqe_blocks/stream_helper.hpp"

#ifndef __SYNTHESIS__
#include <iostream>
//#define USER_DEBUG true
#endif

namespace xf {
namespace database {
namespace gqe {

template <int COL_NM>
void Split_1D(hls::stream<ap_uint<8 * TPCH_INT_SZ * COL_NM> >& in_strm,
              hls::stream<bool>& e_in_strm,
              hls::stream<ap_uint<8 * TPCH_INT_SZ> > out_strm[COL_NM],
              hls::stream<bool>& e_out_strm) {
    bool e;
    int cnt = 0;
    while (!(e = e_in_strm.read())) {
#pragma HLS pipeline II = 1
        ap_uint<8 * TPCH_INT_SZ* COL_NM> tmp = in_strm.read();
        for (int i = 0; i < COL_NM; i++) {
#pragma HLS unroll
            out_strm[i].write(tmp(8 * TPCH_INT_SZ * (i + 1) - 1, 8 * TPCH_INT_SZ * i));
        }
        cnt++;
        e_out_strm.write(false);
    }
    e_out_strm.write(true);
}

/* XXX if dual key, shift the payload,
 * so that for the 3rd col of A table becomes 1st payload
 */
template <int COL_NM, int PLD_NM, int ROUND_NM>
void hash_join_channel_adapter(bool mk_on,
                               hls::stream<ap_uint<8 * TPCH_INT_SZ> > in_strm[COL_NM],
                               hls::stream<bool>& e_in_strm,
                               hls::stream<ap_uint<8 * TPCH_INT_SZ * 2> >& key_strm,
                               hls::stream<ap_uint<8 * TPCH_INT_SZ * PLD_NM> >& pld_strm,
                               hls::stream<bool>& e_join_pld_strm) {
#if !defined __SYNTHESIS__ && USER_DEBUG == 1
    for (int i = 0; i < COL_NM; ++i) {
        printf("Hash join input %d data %d\n", i, in_strm[i].size());
    }
    printf("Hash join end flag %d\n", e_in_strm.size());
#endif
    // 3 round, S table, S table, B table;
    for (int r = 0; r < ROUND_NM; ++r) {
        bool e = true;
        int cnt = 0;
        while (!(e = e_in_strm.read())) {
#pragma HLS pipeline II = 1

            ap_uint<8 * TPCH_INT_SZ * 2> key_tmp;
            ap_uint<8 * TPCH_INT_SZ * PLD_NM> pld_tmp;

            ap_uint<8 * TPCH_INT_SZ> d_tmp[COL_NM];
#pragma HLS array_partition variable = d_tmp complete
            for (int c = 0; c < COL_NM; ++c) {
#pragma HLS unroll
                d_tmp[c] = in_strm[c].read();
                //#ifdef USER_DEBUG
                //                std::cout << "d_tmp[" << c << "]: " << d_tmp[c] << std::endl;
                //#endif
            }
            cnt++;

            key_tmp.range(8 * TPCH_INT_SZ - 1, 0) = d_tmp[0];
            key_tmp.range(8 * TPCH_INT_SZ * 2 - 1, 8 * TPCH_INT_SZ) = mk_on ? d_tmp[1] : ap_uint<8 * TPCH_INT_SZ>(0);

            for (int c = 0; c < PLD_NM; ++c) {
#pragma HLS unroll
                // pld_tmp.range(8 * TPCH_INT_SZ * (c + 1) - 1, 8 * TPCH_INT_SZ * c) = mk_on ? d_tmp[2 + c] : d_tmp[1 +
                // c];
                pld_tmp.range(8 * TPCH_INT_SZ * (c + 1) - 1, 8 * TPCH_INT_SZ * c) = d_tmp[2 + c];
            }

            key_strm.write(key_tmp);
            pld_strm.write(pld_tmp);
#ifdef USER_DEBUG
            std::cout << "key_tmp: " << key_tmp << std::endl;
            std::cout << "pld_tmp: " << pld_tmp << std::endl;
#endif
            e_join_pld_strm.write(false);
        }
        e_join_pld_strm.write(true);
    }
}

template <int COL_IN_NM, int CH_NM, int COL_OUT_NM, int ROUND_NM>
void hash_join_plus_adapter(bool build_probe_flag,
                            // bool jn_on,
                            bool mk_on,
                            hls::stream<ap_uint<3> >& join_flag_strm,
                            hls::stream<ap_uint<8 * TPCH_INT_SZ> > in_strm[CH_NM][COL_IN_NM],
                            hls::stream<bool> e_in_strm[CH_NM],
                            hls::stream<ap_uint<8 * TPCH_INT_SZ> > out_strm[COL_OUT_NM],
                            hls::stream<bool>& e_out_strm,
                            hls::stream<ap_uint<32> >& pu_b_status_strm,
                            hls::stream<ap_uint<32> >& pu_e_status_strm,
                            ap_uint<256>* htb_buf0,
                            ap_uint<256>* htb_buf1,
                            ap_uint<256>* htb_buf2,
                            ap_uint<256>* htb_buf3,
                            ap_uint<256>* htb_buf4,
                            ap_uint<256>* htb_buf5,
                            ap_uint<256>* htb_buf6,
                            ap_uint<256>* htb_buf7,
                            ap_uint<256>* stb_buf0,
                            ap_uint<256>* stb_buf1,
                            ap_uint<256>* stb_buf2,
                            ap_uint<256>* stb_buf3,
                            ap_uint<256>* stb_buf4,
                            ap_uint<256>* stb_buf5,
                            ap_uint<256>* stb_buf6,
                            ap_uint<256>* stb_buf7) {
#pragma HLS dataflow

    hls::stream<ap_uint<8 * TPCH_INT_SZ * 2> > key_strm[CH_NM];
    hls::stream<ap_uint<8 * TPCH_INT_SZ> > pld_strm[CH_NM];
    hls::stream<bool> e_join_pld_strm[CH_NM];

    hls::stream<ap_uint<8 * TPCH_INT_SZ * COL_OUT_NM> > joined_strm;
    hls::stream<bool> e_joined_strm;

#pragma HLS stream variable = key_strm depth = 16
#pragma HLS stream variable = pld_strm depth = 16
#pragma HLS stream variable = e_join_pld_strm depth = 16
#pragma HLS stream variable = joined_strm depth = 16
#pragma HLS stream variable = e_joined_strm depth = 16

    // let each channel adapt independently
    for (int ch = 0; ch < CH_NM; ++ch) {
#pragma HLS unroll
        hash_join_channel_adapter<COL_IN_NM, 1, ROUND_NM>(mk_on, in_strm[ch], e_in_strm[ch], key_strm[ch], pld_strm[ch],
                                                          e_join_pld_strm[ch]);
    }

#if !defined __SYNTHESIS__
    printf("***** after adapt for hash-join\n");
    for (int ch = 0; ch < CH_NM; ++ch) {
        printf("ch:%d nrow=%ld,%ld, nflag=%ld\n", ch, key_strm[ch].size(), pld_strm[ch].size(),
               e_join_pld_strm[ch].size());
    }
#endif

    // <int HASH_MODE, int KEYW, int PW, int S_PW, int B_PW, int HASHWH, int HASHWL, int ARW, int CH_NM>
    xf::database::hashMultiJoinBuildProbe<1, 128, 64, 64, 64, 3, 17, 24, 4>(
        build_probe_flag, join_flag_strm, key_strm, pld_strm, e_join_pld_strm, htb_buf0, htb_buf1, htb_buf2, htb_buf3,
        htb_buf4, htb_buf5, htb_buf6, htb_buf7, stb_buf0, stb_buf1, stb_buf2, stb_buf3, stb_buf4, stb_buf5, stb_buf6,
        stb_buf7, pu_b_status_strm, pu_e_status_strm, joined_strm, e_joined_strm);

#if !defined __SYNTHESIS__ && USER_DEBUG == 1
    printf("Hash joined row %d\n", joined_strm.size());
#endif

    Split_1D<COL_OUT_NM>(joined_strm, e_joined_strm, out_strm, e_out_strm);
}

template <int COL_IN_NM, int CH_NM, int COL_OUT_NM, int ROUND_NM>
void bloomfilter_join_plus_adapter(bool build_probe_flag,
                                   bool mk_on,
                                   bool bf_on,
                                   ap_uint<35> bf_size_in_bits,
                                   hls::stream<ap_uint<3> >& join_flag_strm,
                                   hls::stream<ap_uint<8 * TPCH_INT_SZ> > in_strm[CH_NM][COL_IN_NM],
                                   hls::stream<bool> e_in_strm[CH_NM],
                                   hls::stream<ap_uint<8 * TPCH_INT_SZ> > out_strm[COL_OUT_NM],
                                   hls::stream<bool>& e_out_strm,
                                   hls::stream<ap_uint<32> >& pu_b_status_strm,
                                   hls::stream<ap_uint<32> >& pu_e_status_strm,
                                   ap_uint<256>* htb_buf0,
                                   ap_uint<256>* htb_buf1,
                                   ap_uint<256>* htb_buf2,
                                   ap_uint<256>* htb_buf3,
                                   ap_uint<256>* htb_buf4,
                                   ap_uint<256>* htb_buf5,
                                   ap_uint<256>* htb_buf6,
                                   ap_uint<256>* htb_buf7,
                                   ap_uint<256>* stb_buf0,
                                   ap_uint<256>* stb_buf1,
                                   ap_uint<256>* stb_buf2,
                                   ap_uint<256>* stb_buf3,
                                   ap_uint<256>* stb_buf4,
                                   ap_uint<256>* stb_buf5,
                                   ap_uint<256>* stb_buf6,
                                   ap_uint<256>* stb_buf7) {
#pragma HLS dataflow

    hls::stream<ap_uint<8 * TPCH_INT_SZ * 2> > key_strm[CH_NM];
    hls::stream<ap_uint<8 * TPCH_INT_SZ> > pld_strm[CH_NM];
    hls::stream<bool> e_join_pld_strm[CH_NM];

    hls::stream<ap_uint<8 * TPCH_INT_SZ * COL_OUT_NM> > joined_strm;
    hls::stream<bool> e_joined_strm;

#pragma HLS stream variable = key_strm depth = 16
#pragma HLS stream variable = pld_strm depth = 16
#pragma HLS stream variable = e_join_pld_strm depth = 16
#pragma HLS stream variable = joined_strm depth = 16
#pragma HLS stream variable = e_joined_strm depth = 16

    // let each channel adapt independently
    for (int ch = 0; ch < CH_NM; ++ch) {
#pragma HLS unroll
        hash_join_channel_adapter<COL_IN_NM, 1, ROUND_NM>(mk_on, in_strm[ch], e_in_strm[ch], key_strm[ch], pld_strm[ch],
                                                          e_join_pld_strm[ch]);
    }

#if !defined __SYNTHESIS__
    printf("***** after adapt for hash-join\n");
    for (int ch = 0; ch < CH_NM; ++ch) {
        printf("ch:%d nrow=%ld,%ld, nflag=%ld\n", ch, key_strm[ch].size(), pld_strm[ch].size(),
               e_join_pld_strm[ch].size());
    }
#endif

    // <int HASH_MODE, int KEYW, int PW, int S_PW, int B_PW, int HASHWH, int HASHWL, int HASHWJ,int ARW, int CH_NM>
    xf::database::bloomFilterJoin<1, 128, 64, 64, 64, 3, 31, 17, 24, 4>(
        build_probe_flag, bf_on, bf_size_in_bits, join_flag_strm, key_strm, pld_strm, e_join_pld_strm, htb_buf0,
        htb_buf1, htb_buf2, htb_buf3, htb_buf4, htb_buf5, htb_buf6, htb_buf7, stb_buf0, stb_buf1, stb_buf2, stb_buf3,
        stb_buf4, stb_buf5, stb_buf6, stb_buf7, pu_b_status_strm, pu_e_status_strm, joined_strm, e_joined_strm);

#if !defined __SYNTHESIS__ && USER_DEBUG == 1
    printf("Hash joined row %d\n", joined_strm.size());
#endif

    Split_1D<COL_OUT_NM>(joined_strm, e_joined_strm, out_strm, e_out_strm);
}

template <int COL_IN_NM, int CH_NM, int COL_OUT_NM, int ROUND_NM>
void hash_join_wrapper(hls::stream<ap_uint<6> >& i_join_cfg_strm,
                       hls::stream<ap_uint<6> >& o_join_cfg_strm,
                       hls::stream<ap_uint<8 * TPCH_INT_SZ> > in_strm[CH_NM][COL_IN_NM],
                       hls::stream<bool> e_in_strm[CH_NM],
                       hls::stream<ap_uint<8 * TPCH_INT_SZ> > out_strm[COL_OUT_NM],
                       hls::stream<bool>& e_out_strm,
                       ap_uint<256>* htb_buf0,
                       ap_uint<256>* htb_buf1,
                       ap_uint<256>* htb_buf2,
                       ap_uint<256>* htb_buf3,
                       ap_uint<256>* htb_buf4,
                       ap_uint<256>* htb_buf5,
                       ap_uint<256>* htb_buf6,
                       ap_uint<256>* htb_buf7,
                       ap_uint<256>* stb_buf0,
                       ap_uint<256>* stb_buf1,
                       ap_uint<256>* stb_buf2,
                       ap_uint<256>* stb_buf3,
                       ap_uint<256>* stb_buf4,
                       ap_uint<256>* stb_buf5,
                       ap_uint<256>* stb_buf6,
                       ap_uint<256>* stb_buf7) {
    ap_uint<6> join_cfg = i_join_cfg_strm.read();
    o_join_cfg_strm.write(join_cfg);
    bool mk_on = join_cfg[1];
    bool jn_on = join_cfg[0];
    bool bp_flag = join_cfg[5];
    ap_uint<3> join_type = join_cfg[4, 2];

    hls::stream<ap_uint<32> > pu_begin_status_strm;
#pragma HLS stream variable = pu_begin_status_strm depth = 2
    hls::stream<ap_uint<32> > pu_end_status_strm;
#pragma HLS stream variable = pu_end_status_strm depth = 2
    hls::stream<ap_uint<3> > join_flag_strm;
#pragma HLS stream variable = join_flag_strm depth = 2
    join_flag_strm.write(join_type);

    if (jn_on) {
        pu_begin_status_strm.write(31);
        pu_begin_status_strm.write(0);
        hash_join_plus_adapter<COL_IN_NM, CH_NM, COL_OUT_NM, ROUND_NM>(
            bp_flag, mk_on, join_flag_strm, in_strm, e_in_strm, out_strm, e_out_strm, pu_begin_status_strm,
            pu_end_status_strm, htb_buf0, htb_buf1, htb_buf2, htb_buf3, htb_buf4, htb_buf5, htb_buf6, htb_buf7,
            stb_buf0, stb_buf1, stb_buf2, stb_buf3, stb_buf4, stb_buf5, stb_buf6, stb_buf7);
        int st1 = pu_end_status_strm.read();
        int st2 = pu_end_status_strm.read();
#if !defined __SYNTHESIS__ && USER_DEBUG == 1
        printf("Hash join finished. status(%d, %d).\n", st1, st2);
#endif
    }
}

template <int COL_IN_NM, int CH_NM, int COL_OUT_NM, int ROUND_NM>
void bloomfilter_join_wrapper(hls::stream<ap_uint<36> >& bf_cfg_strm,
                              hls::stream<ap_uint<6> >& i_join_cfg_strm,
                              hls::stream<ap_uint<6> >& o_join_cfg_strm,
                              hls::stream<ap_uint<8 * TPCH_INT_SZ> > in_strm[CH_NM][COL_IN_NM],
                              hls::stream<bool> e_in_strm[CH_NM],
                              hls::stream<ap_uint<8 * TPCH_INT_SZ> > out_strm[COL_OUT_NM],
                              hls::stream<bool>& e_out_strm,
                              ap_uint<256>* htb_buf0,
                              ap_uint<256>* htb_buf1,
                              ap_uint<256>* htb_buf2,
                              ap_uint<256>* htb_buf3,
                              ap_uint<256>* htb_buf4,
                              ap_uint<256>* htb_buf5,
                              ap_uint<256>* htb_buf6,
                              ap_uint<256>* htb_buf7,
                              ap_uint<256>* stb_buf0,
                              ap_uint<256>* stb_buf1,
                              ap_uint<256>* stb_buf2,
                              ap_uint<256>* stb_buf3,
                              ap_uint<256>* stb_buf4,
                              ap_uint<256>* stb_buf5,
                              ap_uint<256>* stb_buf6,
                              ap_uint<256>* stb_buf7) {
    ap_uint<6> join_cfg = i_join_cfg_strm.read();
    o_join_cfg_strm.write(join_cfg);
    bool mk_on = join_cfg[1];
    bool jn_on = join_cfg[0];
    bool bp_flag = join_cfg[5];
    ap_uint<3> join_type = join_cfg[4, 2];

    ap_uint<36> bf_cfg = bf_cfg_strm.read();
    ap_uint<35> bf_size_in_bits = bf_cfg.range(35, 1);
    bool bf_on = bf_cfg[0];

    hls::stream<ap_uint<32> > pu_begin_status_strm;
#pragma HLS stream variable = pu_begin_status_strm depth = 2
    hls::stream<ap_uint<32> > pu_end_status_strm;
#pragma HLS stream variable = pu_end_status_strm depth = 2
    hls::stream<ap_uint<3> > join_flag_strm;
#pragma HLS stream variable = join_flag_strm depth = 2
    join_flag_strm.write(join_type);

    if (jn_on) {
        pu_begin_status_strm.write(31);
        pu_begin_status_strm.write(0);
        bloomfilter_join_plus_adapter<COL_IN_NM, CH_NM, COL_OUT_NM, ROUND_NM>(
            bp_flag, mk_on, bf_on, bf_size_in_bits, join_flag_strm, in_strm, e_in_strm, out_strm, e_out_strm,
            pu_begin_status_strm, pu_end_status_strm, htb_buf0, htb_buf1, htb_buf2, htb_buf3, htb_buf4, htb_buf5,
            htb_buf6, htb_buf7, stb_buf0, stb_buf1, stb_buf2, stb_buf3, stb_buf4, stb_buf5, stb_buf6, stb_buf7);
        int st1 = pu_end_status_strm.read();
        int st2 = pu_end_status_strm.read();
#if !defined __SYNTHESIS__ && USER_DEBUG == 1
        printf("Hash join finished. status(%d, %d).\n", st1, st2);
#endif
    }
}

template <int COL, int CH_NM>
void hash_join_bypass(hls::stream<ap_uint<6> >& join_cfg_strm,
                      hls::stream<ap_uint<8 * TPCH_INT_SZ> > i_jrow_strm[CH_NM][COL],
                      hls::stream<bool> i_e_strm[CH_NM],
                      hls::stream<ap_uint<8 * TPCH_INT_SZ> > o_jrow_strm[COL],
                      hls::stream<bool>& o_e_strm) {
    ap_uint<6> join_cfg = join_cfg_strm.read();
    const int MAX = (1 << CH_NM) - 1;
    ap_uint<8 * TPCH_INT_SZ> j_array[COL];
#pragma HLS array_partition variable = j_array
    ap_uint<CH_NM> empty_e = 0;
    ap_uint<CH_NM> last = 0;
#if !defined __SYNTHESIS__ && USER_DEBUG == 1
    unsigned int cnt = 0;
    std::cout << "CH_NM=" << CH_NM << std::endl;
#endif
    bool jn_on = join_cfg[0];
    ap_uint<CH_NM> id = 0;
    if (!jn_on) {
        do {
#pragma HLS pipeline II = 1
            for (int i = 0; i < CH_NM; i++) {
#pragma HLS unroll
                if (id == i && !i_e_strm[i].empty()) {
                    last[i] = i_e_strm[i].read();
                    for (int c = 0; c < COL; ++c) {
#pragma HLS unroll
                        ap_uint<8 * TPCH_INT_SZ> t = i_jrow_strm[i][c].read();
                        if (!last[i]) o_jrow_strm[c].write(t);
                    }
                    if (!last[i]) o_e_strm.write(false);
                }
            }

            if (id == CH_NM - 1)
                id = 0;
            else
                id++;
        } while (last != MAX);

#if !defined __SYNTHESIS__ && USER_DEBUG == 1
        std::cout << "Collect " << cnt << " rows" << std::endl;
#endif
        o_e_strm.write(true);
    }
}

} // namespace gqe
} // namespace database
} // namespace xf

#endif
