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

// top header
#include "q5kernel.hpp"
// used modules
#include <hls_stream.h>
#include "xf_database/scan_col_2.hpp"
#include "xf_database/hash_join_v2.hpp"

// for debug
#ifndef __SYNTHESIS__
#include <iostream>
#endif

// ------------------------------- kernel entry --------------------------------
static void filter_date(hls::stream<ap_uint<8 * TPCH_INT_SZ> >& i_key1_strm,
                        hls::stream<ap_uint<8 * TPCH_INT_SZ> >& i_pay1_strm,
                        hls::stream<ap_uint<8 * TPCH_INT_SZ> >& i_pay2_strm,
                        hls::stream<ap_uint<8 * TPCH_INT_SZ> >& i_pay3_strm,
                        hls::stream<bool>& i_e_strm,
                        hls::stream<ap_uint<8 * TPCH_INT_SZ> >& o_key1_strm,
                        hls::stream<ap_uint<8 * TPCH_INT_SZ> >& o_pay1_strm,
                        hls::stream<ap_uint<8 * TPCH_INT_SZ> >& o_pay2_strm,
                        hls::stream<ap_uint<8 * TPCH_INT_SZ> >& o_pay3_strm,
                        hls::stream<bool>& o_e_strm,
                        const int enable_filter) {
// const int config) {
#ifndef __SYNTHESIS__
    int cnt = 0;
    int nd = 0;
#endif
    bool e = i_e_strm.read();
#ifndef __SYNTHESIS__
    cnt = 0;
    nd = 0;
#endif
FILTER_DATE_P:
    while (!e) {
#pragma HLS pipeline
        ap_uint<8 * TPCH_INT_SZ> date = i_pay2_strm.read();
        ap_uint<8 * TPCH_INT_SZ> key1 = i_key1_strm.read();
        ap_uint<8 * TPCH_INT_SZ> pld1 = i_pay1_strm.read();
        ap_uint<8 * TPCH_INT_SZ> pld3 = i_pay3_strm.read();
        e = i_e_strm.read();
        if (enable_filter) {
            if (date >= 19940101L && date < 19950101L) {
                o_key1_strm.write(key1);
                o_pay2_strm.write(date);
                o_pay1_strm.write(pld1);
                o_pay3_strm.write(pld3);
                o_e_strm.write(false);
#ifndef __SYNTHESIS__
                ++cnt;
#if defined(Q5DEBUG) && defined(VERBOSE)
                printf("okey = %d\n", pld1.to_uint());
#endif
#endif
            }
#ifndef __SYNTHESIS__
            else {
                ++nd;
            }
#endif
        } else {
            o_key1_strm.write(key1);
            o_pay2_strm.write(date);
            o_pay1_strm.write(pld1);
            o_pay3_strm.write(pld3);
            o_e_strm.write(false);
#ifndef __SYNTHESIS__
            ++cnt;
#if defined(Q5DEBUG) && defined(VERBOSE)
            if (cnt < 100) {
                // std::cout<<"key = "<<key1<<std::endl;
                std::cout << "pld1 = " << pld1 << std::endl;
                // std::cout<<"pld2 = "<<date<<std::endl;
                // std::cout<<"pld3 = "<<pld3<<std::endl;
            }
#endif
#endif
        }
#ifndef __SYNTHESIS__
#if defined(Q5DEBUG) && defined(VERBOSE)
        if (cnt + nd == 1 || (cnt + nd == 375000)) {
            std::cout << "date=" << date << std::endl;
        }
#endif
#endif
    }
    o_e_strm.write(true);
#ifndef __SYNTHESIS__
    printf("filtered %d rows, dropped %d rows.\n", cnt, nd);
#endif
    //}
}
static void burst_write(hls::stream<ap_uint<8 * TPCH_INT_SZ> >& pld_strm,
                        hls::stream<ap_uint<10> >& nrow_strm,
                        hls::stream<bool>& i_e_strm,
                        ap_uint<8 * TPCH_INT_SZ * VEC_LEN>* o_buf) {
    ap_uint<8 * TPCH_INT_SZ* VEC_LEN>* ptr = o_buf + 1;
    unsigned int tot = 0;
    bool e = 0; // i_e_strm.read();
    do {
        e = i_e_strm.read();
        ap_uint<10> nm = nrow_strm.read();
    BURST_WR_LOOP:
        for (unsigned int r = 0; r < 32; ++r) {
#pragma HLS pipeline II = 16
            ap_uint<8 * TPCH_INT_SZ* VEC_LEN> out = 0;
            for (unsigned int len = 0; len < VEC_LEN; len++) {
                if (r * VEC_LEN + len < nm) {
                    ap_uint<8 * TPCH_INT_SZ> jn = pld_strm.read();
                    out.range(8 * TPCH_INT_SZ * (len + 1) - 1, 8 * TPCH_INT_SZ * len) = jn;
                    tot++;
                }
            }
            *ptr++ = out;
        }
    } while (!e);
    *o_buf = tot;
}

/* small table at MSB, big table at LSB */
static void col_splitter(hls::stream<ap_uint<8 * TPCH_INT_SZ * OUT_COL_NUM> >& jstrm,
                         hls::stream<bool>& ejstrm,
                         //
                         hls::stream<ap_uint<8 * TPCH_INT_SZ> > o_pld[OUT_COL_NUM],
                         hls::stream<ap_uint<10> > o_nrow_strm[OUT_COL_NUM],
                         hls::stream<bool> o_2e[OUT_COL_NUM],
                         const int idx) {
    const ap_uint<8> idx_u = idx;
    const ap_uint<2> key_idx = idx_u(1, 0);
    const ap_uint<2> pld1_idx = idx_u(3, 2);
    const ap_uint<2> pld2_idx = idx_u(5, 4);
    const ap_uint<2> pld3_idx = idx_u(7, 6);
    ap_uint<10> nm = 0;
#ifndef __SYNTHESIS__
    int cnt = 0;
#endif
    bool e = ejstrm.read();
COL_SPLITTER_P:
    while (!e) {
#pragma HLS pipeline
        if (nm == 511) {
            nm = 0;
            for (int i = 0; i < OUT_COL_NUM; ++i) {
#pragma HLS unroll
                o_2e[i].write(false);
                o_nrow_strm[i].write(512);
            }
        } else {
            nm++;
        }
        ap_uint<8 * TPCH_INT_SZ* OUT_COL_NUM> jrow = jstrm.read();
        // drop 1bit from small table dummy payload
        ap_uint<8 * TPCH_INT_SZ> k = jrow.range(8 * TPCH_INT_SZ * (key_idx + 1) - 1, 8 * TPCH_INT_SZ * key_idx);
        ap_uint<8 * TPCH_INT_SZ> pld1 = jrow.range(8 * TPCH_INT_SZ * (pld1_idx + 1) - 1, 8 * TPCH_INT_SZ * pld1_idx);
        ap_uint<8 * TPCH_INT_SZ> pld2 = jrow.range(8 * TPCH_INT_SZ * (pld2_idx + 1) - 1, 8 * TPCH_INT_SZ * pld2_idx);
        ap_uint<8 * TPCH_INT_SZ> pld3 = jrow.range(8 * TPCH_INT_SZ * (pld3_idx + 1) - 1, 8 * TPCH_INT_SZ * pld3_idx);
        e = ejstrm.read();
        o_pld[0].write(k);
        o_pld[1].write(pld1);
        o_pld[2].write(pld2);
        o_pld[3].write(pld3);
#ifndef __SYNTHESIS__
        ++cnt;
#if defined(Q5DEBUG) && defined(VERBOSE)
        if (cnt < 10) {
            std::cout << "key =" << k << std::endl;
            std::cout << "pld1 =" << pld1 << std::endl;
            std::cout << "pld2 =" << pld2 << std::endl;
            std::cout << "pld3 =" << pld3 << std::endl;
        }
#endif
#endif
    }
    if (nm != 0) {
        for (int i = 0; i < OUT_COL_NUM; ++i) {
#pragma HLS unroll
            o_2e[i].write(true);
            o_nrow_strm[i].write(nm);
        }
    } else {
        for (int i = 0; i < OUT_COL_NUM; ++i) {
#pragma HLS unroll
            o_2e[i].write(true);
            o_nrow_strm[i].write(0);
        }
    }
#ifndef __SYNTHESIS__
    printf("hash-joined %d rows.\n", cnt);
#endif
}

template <int CH_NM>
static void scan_table(
    // small table buffer
    ap_uint<8 * TPCH_INT_SZ * VEC_LEN>* buf_skey1,
    ap_uint<8 * TPCH_INT_SZ * VEC_LEN>* buf_spay1,
    // big table buffer
    ap_uint<8 * TPCH_INT_SZ * VEC_LEN>* buf_bkey1,
    ap_uint<8 * TPCH_INT_SZ * VEC_LEN>* buf_bpay1,
    ap_uint<8 * TPCH_INT_SZ * VEC_LEN>* buf_bpay2,
    ap_uint<8 * TPCH_INT_SZ * VEC_LEN>* buf_bpay3,
    hls::stream<ap_uint<8 * TPCH_INT_SZ> > o_skey_1[CH_NM],
    hls::stream<ap_uint<8 * TPCH_INT_SZ> > o_spld_1[CH_NM],
    hls::stream<bool> s_1e[CH_NM],
    hls::stream<ap_uint<8 * TPCH_INT_SZ> > o_bkey_1[CH_NM],
    hls::stream<ap_uint<8 * TPCH_INT_SZ> > o_bpay_1[CH_NM],
    hls::stream<ap_uint<8 * TPCH_INT_SZ> > o_bpay_2[CH_NM],
    hls::stream<ap_uint<8 * TPCH_INT_SZ> > o_bpay_3[CH_NM],
    hls::stream<bool> b_1e[CH_NM]) {
    for (int r = 0; r < 2; ++r) {
        xf::database::scanCol<64, VEC_LEN, CH_NM, TPCH_INT_SZ, TPCH_INT_SZ>(buf_skey1, buf_spay1, o_skey_1, o_spld_1,
                                                                            s_1e);
    }

    xf::database::scanCol<64, VEC_LEN, CH_NM, TPCH_INT_SZ, TPCH_INT_SZ, TPCH_INT_SZ, TPCH_INT_SZ>(
        buf_bkey1, buf_bpay1, buf_bpay2, buf_bpay3, o_bkey_1, o_bpay_1, o_bpay_2, o_bpay_3, b_1e);
}

static void small_big_table_feeder_for_part(
    // small table input
    hls::stream<ap_uint<8 * TPCH_INT_SZ> >& skey_1,
    hls::stream<ap_uint<8 * TPCH_INT_SZ> >& spay_1,
    hls::stream<bool>& s_1e,
    // big table input
    hls::stream<ap_uint<8 * TPCH_INT_SZ> >& bkey_1,
    hls::stream<ap_uint<8 * TPCH_INT_SZ> >& bpay_1,
    hls::stream<ap_uint<8 * TPCH_INT_SZ> >& bpay_2,
    hls::stream<ap_uint<8 * TPCH_INT_SZ> >& bpay_3,
    hls::stream<bool>& b_1e,
    // output
    hls::stream<ap_uint<8 * TPCH_INT_SZ * KEY_NUM> >& kstrm,
    hls::stream<ap_uint<8 * TPCH_INT_SZ * PLD_NUM> >& pstrm,
    hls::stream<bool>& estrm,
    const int config) {
#ifndef __SYNTHESIS__
    int cnt = 0;
#endif
    bool se = 0;
    // send small table
    for (int r = 0; r < 2; r++) {
        se = s_1e.read();
#ifndef __SYNTHESIS__
        int cnt = 0;
#endif
    SMALL_FEEDER_P:
        while (!se) {
#pragma HLS pipeline
            ap_uint<8 * TPCH_INT_SZ* KEY_NUM> k = 0;
            ap_uint<8 * TPCH_INT_SZ> sp = spay_1.read();
            k(8 * TPCH_INT_SZ - 1, 0) = skey_1.read();

            if (config == 2) {
                k(8 * TPCH_INT_SZ * 2 - 1, 8 * TPCH_INT_SZ) = sp;
            }

            ap_uint<8 * TPCH_INT_SZ* PLD_NUM> p = 0; // no payload.
            p(8 * TPCH_INT_SZ - 1, 0) = sp;
            // FIXME
            se = s_1e.read();

            kstrm.write(k);
            pstrm.write(p);
            estrm.write(false);
#ifndef __SYNTHESIS__
            ++cnt;
#if defined(Q5DEBUG) && defined(VERBOSE)
            if (cnt < 1000) std::cout << "key =" << k << std::endl;
#endif
#endif
        }
        estrm.write(true);
#ifndef __SYNTHESIS__
        printf("feeded %d rows from small table to hash-join.\n", cnt);
        cnt = 0;
#endif
    }

    // send big table
    bool be = b_1e.read();
BIG_FEEDER_P:
    while (!be) {
#pragma HLS pipeline
        ap_uint<8 * TPCH_INT_SZ> k_in = bkey_1.read();
        ap_uint<8 * TPCH_INT_SZ> bp3 = bpay_3.read();
        ap_uint<8 * TPCH_INT_SZ> bp2 = bpay_2.read();

        ap_uint<8 * TPCH_INT_SZ* KEY_NUM> k = 0;
        k(8 * TPCH_INT_SZ - 1, 0) = k_in;
        if (config == 2) {
            k(8 * 2 * TPCH_INT_SZ - 1, 8 * TPCH_INT_SZ) = bp3;
        }

        // write back key, price and discount
        ap_uint<8 * TPCH_INT_SZ* PLD_NUM> p = 0;
        p.range(8 * TPCH_INT_SZ - 1, 0) = bpay_1.read();
        if (config == 3)
            p.range(8 * TPCH_INT_SZ * 2 - 1, 8 * TPCH_INT_SZ) = k_in;
        else
            p.range(8 * TPCH_INT_SZ * 2 - 1, 8 * TPCH_INT_SZ) = bp2;

        p.range(8 * TPCH_INT_SZ * 3 - 1, 8 * TPCH_INT_SZ * 2) = bp3;
        be = b_1e.read();
        kstrm.write(k);
        pstrm.write(p);
        estrm.write(false);
#ifndef __SYNTHESIS__
        ++cnt;
#if defined(Q5DEBUG) && defined(VERBOSE)
        if (cnt < 100) {
            ap_uint<8 * TPCH_INT_SZ> t = 0;
            t = p.range(8 * TPCH_INT_SZ - 1, 0);
            std::cout << "pld1 = " << t << std::endl;
            // t = p.range(8 * TPCH_INT_SZ * 2 - 1, 8 * TPCH_INT_SZ);
            // std::cout<<"pld2 = "<<t<<std::endl;
            // std::cout<<"pld3 = "<<bp<<std::endl;
        }
#endif
#endif
    }
    estrm.write(true);
#ifndef __SYNTHESIS__
    printf("feeded %d rows from big table to hash-join.\n", cnt);
#endif
}
// CH_NM channel number of key input
template <int CH_NM>
static void scan_wrapper(ap_uint<8 * TPCH_INT_SZ * VEC_LEN>* buf_skey1,
                         ap_uint<8 * TPCH_INT_SZ * VEC_LEN>* buf_spay1,
                         // big table buffer
                         ap_uint<8 * TPCH_INT_SZ * VEC_LEN>* buf_bkey1,
                         ap_uint<8 * TPCH_INT_SZ * VEC_LEN>* buf_bpay1,
                         ap_uint<8 * TPCH_INT_SZ * VEC_LEN>* buf_bpay2,
                         ap_uint<8 * TPCH_INT_SZ * VEC_LEN>* buf_bpay3,
                         hls::stream<ap_uint<8 * KEY_NUM * TPCH_INT_SZ> > o_key_strm[CH_NM],
                         hls::stream<ap_uint<8 * PLD_NUM * TPCH_INT_SZ> > o_pld_strm[CH_NM],
                         hls::stream<bool> o_e_strm[CH_NM],
                         const int enable_filter,
                         const int config) {
    hls::stream<ap_uint<8 * (TPCH_INT_SZ)> > skey_1[CH_NM];
#pragma HLS stream variable = skey_1 depth = 32
#pragma HLS array_partition variable = skey_1 dim = 0
#pragma HLS bind_storage variable = skey_1 type = fifo impl = srl
    hls::stream<ap_uint<8 * (TPCH_INT_SZ)> > spay_1[CH_NM];
#pragma HLS stream variable = spay_1 depth = 32
#pragma HLS array_partition variable = spay_1 dim = 0
#pragma HLS bind_storage variable = spay_1 type = fifo impl = srl
    hls::stream<bool> s_1e[CH_NM]; // flag stream, when true, end of data.
#pragma HLS stream variable = s_1e depth = 32
#pragma HLS array_partition variable = s_1e dim = 0

    // ------------ scan lineitem -----------
    hls::stream<ap_uint<8 * (TPCH_INT_SZ)> > bkey1_1[CH_NM];
#pragma HLS stream variable = bkey1_1 depth = 32
#pragma HLS bind_storage variable = bkey1_1 type = fifo impl = srl
#pragma HLS array_partition variable = bkey1_1 dim = 0
    hls::stream<ap_uint<8 * (TPCH_INT_SZ)> > bpay1_1[CH_NM];
#pragma HLS stream variable = bpay1_1 depth = 32
#pragma HLS bind_storage variable = bpay1_1 type = fifo impl = srl
#pragma HLS array_partition variable = bpay1_1 dim = 0
    hls::stream<ap_uint<8 * (TPCH_INT_SZ)> > bpay2_1[CH_NM];
#pragma HLS stream variable = bpay2_1 depth = 32
#pragma HLS bind_storage variable = bpay2_1 type = fifo impl = srl
#pragma HLS array_partition variable = bpay2_1 dim = 0
    hls::stream<ap_uint<8 * (TPCH_INT_SZ)> > bpay3_1[CH_NM];
#pragma HLS stream variable = bpay3_1 depth = 32
#pragma HLS bind_storage variable = bpay3_1 type = fifo impl = srl
#pragma HLS array_partition variable = bpay3_1 dim = 0
    hls::stream<bool> b_1e[CH_NM]; // flag stream, when true, end of data.
#pragma HLS stream variable = b_1e depth = 32
#pragma HLS array_partition variable = b_1e dim = 0
    //----------------------------------
    hls::stream<ap_uint<8 * (TPCH_INT_SZ)> > bkey1_2[CH_NM];
#pragma HLS stream variable = bkey1_2 depth = 16
#pragma HLS array_partition variable = bkey1_2 dim = 0
#pragma HLS bind_storage variable = bkey1_2 type = fifo impl = srl
    hls::stream<ap_uint<8 * (TPCH_INT_SZ)> > bpay1_2[CH_NM];
#pragma HLS stream variable = bpay1_2 depth = 16
#pragma HLS array_partition variable = bpay1_2 dim = 0
#pragma HLS bind_storage variable = bpay1_2 type = fifo impl = srl
    hls::stream<ap_uint<8 * (TPCH_INT_SZ)> > bpay2_2[CH_NM];
#pragma HLS stream variable = bpay2_2 depth = 16
#pragma HLS array_partition variable = bpay2_2 dim = 0
#pragma HLS bind_storage variable = bpay2_2 type = fifo impl = srl
    hls::stream<ap_uint<8 * (TPCH_INT_SZ)> > bpay3_2[CH_NM];
#pragma HLS stream variable = bpay3_2 depth = 16
#pragma HLS array_partition variable = bpay3_2 dim = 0
#pragma HLS bind_storage variable = bpay3_2 type = fifo impl = srl
    hls::stream<bool> b_2e[CH_NM]; // flag stream, when true, end of data.
#pragma HLS stream variable = b_2e depth = 16
#pragma HLS array_partition variable = b_2e dim = 0
//-------------------------------
#pragma HLS DATAFLOW
    scan_table<CH_NM>(buf_skey1, buf_spay1, buf_bkey1, buf_bpay1, buf_bpay2, buf_bpay3, skey_1, spay_1, s_1e, bkey1_1,
                      bpay1_1, bpay2_1, bpay3_1, b_1e);
    for (int c = 0; c < CH_NM; ++c) {
#pragma HLS unroll
        filter_date(bkey1_1[c], bpay1_1[c], bpay2_1[c], bpay3_1[c], b_1e[c], bkey1_2[c], bpay1_2[c], bpay2_2[c],
                    bpay3_2[c], b_2e[c], enable_filter);
        small_big_table_feeder_for_part(skey_1[c], spay_1[c], s_1e[c], bkey1_2[c], bpay1_2[c], bpay2_2[c], bpay3_2[c],
                                        b_2e[c], o_key_strm[c], o_pld_strm[c], o_e_strm[c], config);
    }
}

// CH_NM_K channel number of row input
template <int HASH_MODE,
          int KEYW,
          int PW,
          int S_PW,
          int B_PW,
          int HASHWH,
          int HASHWL,
          int ARW,
          int BFW,
          int CH_NM,
          int PU,
          int BF_W,
          int EN_BF>
static void hash_join_wrapper(ap_uint<8 * TPCH_INT_SZ * VEC_LEN>* buf_skey1,
                              ap_uint<8 * TPCH_INT_SZ * VEC_LEN>* buf_spay1,
                              // big table
                              ap_uint<8 * TPCH_INT_SZ * VEC_LEN>* buf_bkey1,
                              ap_uint<8 * TPCH_INT_SZ * VEC_LEN>* buf_bpay1,
                              ap_uint<8 * TPCH_INT_SZ * VEC_LEN>* buf_bpay2,
                              ap_uint<8 * TPCH_INT_SZ * VEC_LEN>* buf_bpay3,
                              // output
                              ap_uint<8 * TPCH_INT_SZ * VEC_LEN>* buf_okey1,
                              ap_uint<8 * TPCH_INT_SZ * VEC_LEN>* buf_opay1,
                              ap_uint<8 * TPCH_INT_SZ * VEC_LEN>* buf_opay2,
                              ap_uint<8 * TPCH_INT_SZ * VEC_LEN>* buf_opay3,
                              // buffer
                              ap_uint<BFW>* buf0,
                              ap_uint<BFW>* buf1,
                              ap_uint<BFW>* buf2,
                              ap_uint<BFW>* buf3,
                              ap_uint<BFW>* buf4,
                              ap_uint<BFW>* buf5,
                              ap_uint<BFW>* buf6,
                              ap_uint<BFW>* buf7,
                              //
                              const int idx,
                              const int enable_filter,
                              const int config) {
#pragma HLS dataflow

    hls::stream<ap_uint<KEYW> > k0_strm_arry[CH_NM];
#pragma HLS stream variable = k0_strm_arry depth = 8
#pragma HLS array_partition variable = k0_strm_arry dim = 0
#pragma HLS bind_storage variable = k0_strm_arry type = fifo impl = srl
    hls::stream<ap_uint<PW> > p0_strm_arry[CH_NM];
#pragma HLS stream variable = p0_strm_arry depth = 18
#pragma HLS array_partition variable = p0_strm_arry dim = 0
    hls::stream<bool> e0_strm_arry[CH_NM];
#pragma HLS stream variable = e0_strm_arry depth = 8
#pragma HLS array_partition variable = e0_strm_arry dim = 0

    // extend when multiple channel
    scan_wrapper<Q5_HJ_CH_NM>(buf_skey1, buf_spay1, buf_bkey1, buf_bpay1, buf_bpay2, buf_bpay3, k0_strm_arry,
                              p0_strm_arry, e0_strm_arry, enable_filter, config);

    //----------------------------hash join-------------------------//

    hls::stream<ap_uint<8 * TPCH_INT_SZ * OUT_COL_NUM> > j1_strm;
#pragma HLS stream variable = j1_strm depth = 70
#pragma HLS bind_storage variable = j1_strm type = fifo impl = srl
    hls::stream<bool> e5_strm;
#pragma HLS stream variable = e5_strm depth = 8

    xf::database::hashJoinMPU<HASH_MODE,                 // hash algorithm
                              KEYW,                      // key width
                              PW,                        // payload width max
                              S_PW,                      // payload width small
                              B_PW,                      // payload width big
                              HASHWH,                    // log2(number of PU)
                              HASHWL,                    // hash width for join
                              ARW,                       // address width
                              BFW,                       // buffer width
                              CH_NM,                     // input channel number
                              BF_W,                      // BF width
                              EN_BF>                     // enable BF
        (k0_strm_arry, p0_strm_arry, e0_strm_arry,       // in
         buf0, buf1, buf2, buf3, buf4, buf5, buf6, buf7, // tmp
         j1_strm, e5_strm);                              // out

    //--------------------------------------------------------------//

    hls::stream<ap_uint<8 * (TPCH_INT_SZ)> > opld_strm_arry[4];
#pragma HLS stream variable = opld_strm_arry depth = 640
#pragma HLS array_partition variable = opld_strm_arry dim = 1
    hls::stream<ap_uint<10> > nm2_strm_arry[4];
#pragma HLS stream variable = nm2_strm_arry depth = 8
    hls::stream<bool> e6_strm_arry[4];
#pragma HLS stream variable = e6_strm_arry depth = 8
    col_splitter(j1_strm, e5_strm, opld_strm_arry, nm2_strm_arry, e6_strm_arry, idx);

    //-----------------------------burst write----------------------//

    burst_write(opld_strm_arry[0], nm2_strm_arry[0], e6_strm_arry[0], buf_okey1);
    burst_write(opld_strm_arry[1], nm2_strm_arry[1], e6_strm_arry[1], buf_opay1);
    burst_write(opld_strm_arry[2], nm2_strm_arry[2], e6_strm_arry[2], buf_opay2);
    burst_write(opld_strm_arry[3], nm2_strm_arry[3], e6_strm_arry[3], buf_opay3);
}

extern "C" void q5_hash_join(ap_uint<8 * TPCH_INT_SZ * VEC_LEN> buf_skey1[D_DEPTH],
                             ap_uint<8 * TPCH_INT_SZ * VEC_LEN> buf_spay1[D_DEPTH],
                             //
                             ap_uint<8 * TPCH_INT_SZ * VEC_LEN> buf_bkey1[D_DEPTH],
                             ap_uint<8 * TPCH_INT_SZ * VEC_LEN> buf_bpay1[D_DEPTH],
                             ap_uint<8 * TPCH_INT_SZ * VEC_LEN> buf_bpay2[D_DEPTH],
                             ap_uint<8 * TPCH_INT_SZ * VEC_LEN> buf_bpay3[D_DEPTH],
                             //
                             ap_uint<8 * TPCH_INT_SZ * VEC_LEN> buf_okey1[D_DEPTH],
                             ap_uint<8 * TPCH_INT_SZ * VEC_LEN> buf_opay1[D_DEPTH],
                             ap_uint<8 * TPCH_INT_SZ * VEC_LEN> buf_opay2[D_DEPTH],
                             ap_uint<8 * TPCH_INT_SZ * VEC_LEN> buf_opay3[D_DEPTH],
                             //
                             ap_uint<8 * TPCH_INT_SZ * 4> buf0[BUFF_DEPTH],
                             ap_uint<8 * TPCH_INT_SZ * 4> buf1[BUFF_DEPTH],
                             ap_uint<8 * TPCH_INT_SZ * 4> buf2[BUFF_DEPTH],
                             ap_uint<8 * TPCH_INT_SZ * 4> buf3[BUFF_DEPTH],
                             ap_uint<8 * TPCH_INT_SZ * 4> buf4[BUFF_DEPTH],
                             ap_uint<8 * TPCH_INT_SZ * 4> buf5[BUFF_DEPTH],
                             ap_uint<8 * TPCH_INT_SZ * 4> buf6[BUFF_DEPTH],
                             ap_uint<8 * TPCH_INT_SZ * 4> buf7[BUFF_DEPTH],
                             const int idx, // idx3||idx2||idx1||idx0
                             const int enable_filter,
                             const int config) { // 0: nothing||1:enable filter||2:use_two_key||3:output key

// clang-format off
#pragma HLS INTERFACE m_axi offset = slave latency = 64 \
  num_read_outstanding = 16 num_write_outstanding = 16\
  max_read_burst_length = 64 max_write_burst_length = 64\
  bundle = gmem0_0 port = buf_skey1

#pragma HLS INTERFACE m_axi offset = slave latency = 64 \
  num_read_outstanding = 16  num_write_outstanding = 16\
  max_read_burst_length = 64 max_write_burst_length = 64\
  bundle = gmem0_1 port = buf_spay1

#pragma HLS INTERFACE m_axi offset = slave latency = 64 \
  num_read_outstanding = 16 num_write_outstanding = 16\
  max_read_burst_length = 64 max_write_burst_length = 64\
  bundle = gmem0_2 port = buf_bkey1

#pragma HLS INTERFACE m_axi offset = slave latency = 64 \
  num_read_outstanding = 16 num_write_outstanding = 16\
  max_read_burst_length = 64 max_write_burst_length = 64\
  bundle = gmem0_3 port = buf_bpay1

#pragma HLS INTERFACE m_axi offset = slave latency = 64 \
  num_read_outstanding = 16 num_write_outstanding = 16\
  max_read_burst_length = 64 max_write_burst_length = 64\
  bundle = gmem0_4 port = buf_bpay2

#pragma HLS INTERFACE m_axi offset = slave latency = 64 \
  num_read_outstanding = 16 num_write_outstanding = 16\
  max_read_burst_length = 64 max_write_burst_length = 64\
  bundle = gmem0_5 port = buf_bpay3

#pragma HLS INTERFACE m_axi offset = slave \
  num_write_outstanding = 16 num_read_outstanding = 16\
  max_write_burst_length = 64 max_read_burst_length = 64\
  bundle = gmem0_6 port = buf_okey1

#pragma HLS INTERFACE m_axi offset = slave \
  num_write_outstanding = 16 num_read_outstanding = 16\
  max_write_burst_length = 64 max_read_burst_length = 64\
  bundle = gmem0_7 port = buf_opay1

#pragma HLS INTERFACE m_axi offset = slave \
  num_write_outstanding = 16 num_read_outstanding = 16\
  max_write_burst_length = 64 max_read_burst_length = 64\
  bundle = gmem0_8 port = buf_opay2

#pragma HLS INTERFACE m_axi offset = slave \
  num_write_outstanding = 16 num_read_outstanding = 16\
  max_write_burst_length = 64 max_read_burst_length = 64\
  bundle = gmem0_9 port = buf_opay3

#pragma HLS INTERFACE m_axi port = buf0 bundle = gmem1_0 \
  num_write_outstanding = 32 num_read_outstanding = 32 \
  max_read_burst_length = 8 latency = 125

#pragma HLS INTERFACE m_axi port = buf1 bundle = gmem1_1 \
  num_write_outstanding = 32 num_read_outstanding = 32 \
  max_read_burst_length = 8 latency = 125

#pragma HLS INTERFACE m_axi port = buf2 bundle = gmem1_2 \
  num_write_outstanding = 32 num_read_outstanding = 32 \
  max_read_burst_length = 8 latency = 125

#pragma HLS INTERFACE m_axi port = buf3 bundle = gmem1_3 \
  num_write_outstanding = 32 num_read_outstanding = 32 \
  max_read_burst_length = 8 latency = 125

#pragma HLS INTERFACE m_axi port = buf4 bundle = gmem1_4 \
  num_write_outstanding = 32 num_read_outstanding = 32 \
  max_read_burst_length = 8 latency = 125

#pragma HLS INTERFACE m_axi port = buf5 bundle = gmem1_5 \
  num_write_outstanding = 32 num_read_outstanding = 32 \
  max_read_burst_length = 8 latency = 125

#pragma HLS INTERFACE m_axi port = buf6 bundle = gmem1_6 \
  num_write_outstanding = 32 num_read_outstanding = 32 \
  max_read_burst_length = 8 latency = 125

#pragma HLS INTERFACE m_axi port = buf7 bundle = gmem1_7 \
  num_write_outstanding = 32 num_read_outstanding = 32 \
  max_read_burst_length = 8 latency = 125

#pragma HLS INTERFACE s_axilite port = buf_skey1 bundle = control
#pragma HLS INTERFACE s_axilite port = buf_spay1 bundle = control
#pragma HLS INTERFACE s_axilite port = buf_bkey1 bundle = control
#pragma HLS INTERFACE s_axilite port = buf_bpay1 bundle = control
#pragma HLS INTERFACE s_axilite port = buf_bpay2 bundle = control
#pragma HLS INTERFACE s_axilite port = buf_bpay3 bundle = control
#pragma HLS INTERFACE s_axilite port = buf_okey1 bundle = control
#pragma HLS INTERFACE s_axilite port = buf_opay1 bundle = control
#pragma HLS INTERFACE s_axilite port = buf_opay2 bundle = control
#pragma HLS INTERFACE s_axilite port = buf_opay3 bundle = control

#pragma HLS INTERFACE s_axilite port = buf0 bundle = control
#pragma HLS INTERFACE s_axilite port = buf1 bundle = control
#pragma HLS INTERFACE s_axilite port = buf2 bundle = control
#pragma HLS INTERFACE s_axilite port = buf3 bundle = control
#pragma HLS INTERFACE s_axilite port = buf4 bundle = control
#pragma HLS INTERFACE s_axilite port = buf5 bundle = control
#pragma HLS INTERFACE s_axilite port = buf6 bundle = control
#pragma HLS INTERFACE s_axilite port = buf7 bundle = control



#pragma HLS INTERFACE s_axilite port = idx bundle = control
#pragma HLS INTERFACE s_axilite port = config bundle = control
#pragma HLS INTERFACE s_axilite port = enable_filter bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control
    // clang-format on

    hash_join_wrapper<Q5_HJ_MODE,
                      8 * TPCH_INT_SZ * 2, // KEYW
                      8 * TPCH_INT_SZ * 3, // PW
                      8 * TPCH_INT_SZ,     // S_PW
                      8 * TPCH_INT_SZ * 3, // B_PW
                      Q5_HJ_HW_P,          // HASHWH
                      Q5_HJ_HW_J,          // HASHWL
                      Q5_HJ_AW,            // ARW
                      8 * TPCH_INT_SZ * 4, // BFW
                      Q5_HJ_CH_NM, Q5_HJ_PU_NM, Q5_HJ_BVW, 0>(
        buf_skey1, buf_spay1, buf_bkey1, buf_bpay1, buf_bpay2, buf_bpay3, buf_okey1, buf_opay1, buf_opay2, buf_opay3,
        buf0, buf1, buf2, buf3, buf4, buf5, buf6, buf7, idx, enable_filter, config);
}
