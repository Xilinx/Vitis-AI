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
#include "join_kernel.hpp"
// used modules
#include <hls_stream.h>
#define URAM_SPLITTING 1

#include "xf_database/aggregate.hpp"
#include "xf_database/hash_join_v3.hpp"
#include "xf_database/scan_col.hpp"

/* small table at MSB, big table at LSB */
// key 1, spay 1 , tpay 2.
static void col_splitter(hls::stream<ap_uint<W_TPCH_INT * 4> >& jstrm,
                         hls::stream<bool>& ejstrm,
                         //
                         hls::stream<ap_uint<W_TPCH_INT> >& l_eprice_2,
                         hls::stream<ap_uint<W_TPCH_INT> >& l_discount_2,
                         hls::stream<bool>& l_2e) {
#ifndef __SYNTHESIS__
    int cnt = 0;
#endif
    bool e = ejstrm.read();
COL_SPLITTER_P:
    while (!e) {
#pragma HLS pipeline
        ap_uint<W_TPCH_INT* 3> jrow = jstrm.read();
        // drop 1bit from small table dummy payload
        ap_uint<W_TPCH_INT> eprice = jrow.range(W_TPCH_INT - 1, 0);
        ap_uint<W_TPCH_INT> discount = jrow.range(W_TPCH_INT * 2 - 1, W_TPCH_INT);
        ap_uint<W_TPCH_INT> key = jrow.range(W_TPCH_INT * 3 - 1, 2 * W_TPCH_INT);
        e = ejstrm.read();

        l_eprice_2.write(eprice);
        l_discount_2.write(discount);
        l_2e.write(false);
#ifndef __SYNTHESIS__
        ++cnt;
        printf("%ld\n", (long)key);
#if defined(Q5DEBUG) && defined(VERBOSE)
        printf("eprice = %ld, discount = %ld\n", (long)eprice, (long)discount);
#endif
#endif
    }
    l_2e.write(true);
#ifndef __SYNTHESIS__
    printf("hash-joined %d rows.\n", cnt);
#endif
}

// TODO not to hard-code
/* data type is money, which is represent by cent. */
static void multiply_cal(hls::stream<ap_uint<W_TPCH_INT> >& eprice_strm,
                         hls::stream<ap_uint<W_TPCH_INT> >& discount_strm,
                         hls::stream<bool>& i_e_strm,
                         hls::stream<ap_uint<W_TPCH_INT> >& revenue_strm,
                         hls::stream<bool>& o_e_strm) {
#ifndef __SYNTHESIS__
    int cnt = 0;
#endif
    bool e = i_e_strm.read();
MULTIPLY_P:
    while (!e) {
#pragma HLS pipeline
        ap_uint<W_TPCH_INT> p = eprice_strm.read();
        ap_uint<W_TPCH_INT> d = discount_strm.read();
        e = i_e_strm.read();

        ap_uint<W_TPCH_INT> r = p * (100 - d); // FIXME overflow?
        revenue_strm.write(r);
#ifndef __SYNTHESIS__
        ++cnt;
// printf("revenue = %ld\n", (long)r);
#endif
        o_e_strm.write(false);
    }
    o_e_strm.write(true);
#ifndef __SYNTHESIS__
    printf("calculated %d revenue values.\n", cnt);
#endif
}

static void write_result(ap_uint<W_TPCH_INT * 2> buf_result[],
                         hls::stream<ap_uint<W_TPCH_INT * 2> >& d_strm,
                         hls::stream<bool>& e_strm) {
    bool e = e_strm.read();
    int i = 0;
WRITE_RESULT_P:
    while (!e) {
#pragma HLS pipeline
        ap_uint<W_TPCH_INT* 2> t = d_strm.read();
#ifndef __SYNTHESIS__
        long long tl = t.to_int64();
        printf("writing result %d: %lu (%lx).\n", i, tl, tl);
#endif
        buf_result[i++] = t;
        e = e_strm.read();
    }
}

template <int CH_NM>
static void scan_table(
    // small table buffer
    ap_uint<W_TPCH_INT * VEC_LEN> buf_o_orderkey[O_DEPTH],
    const int o_nrow,
    // big table buffer
    ap_uint<W_TPCH_INT * VEC_LEN> buf_l_orderkey[L_DEPTH],
    ap_uint<W_TPCH_INT * VEC_LEN> buf_l_extendedprice[L_DEPTH],
    ap_uint<W_TPCH_INT * VEC_LEN> buf_l_discount[L_DEPTH],
    const int l_nrow,
    // small table stream
    hls::stream<ap_uint<W_TPCH_INT> > o_okey_1[CH_NM],
    hls::stream<bool> o_1e[CH_NM],
    // big table stream
    hls::stream<ap_uint<W_TPCH_INT> > l_okey_1[CH_NM],
    hls::stream<ap_uint<W_TPCH_INT> > l_eprice_1[CH_NM],
    hls::stream<ap_uint<W_TPCH_INT> > l_discount_1[CH_NM],
    hls::stream<bool> l_1e[CH_NM]) {
    xf::database::scanCol<64, VEC_LEN, CH_NM, TPCH_INT_SZ>( //
        buf_o_orderkey, o_nrow,                             //
        o_okey_1, o_1e);

    xf::database::scanCol<64, VEC_LEN, CH_NM, TPCH_INT_SZ, TPCH_INT_SZ, TPCH_INT_SZ>( //
        buf_l_orderkey, buf_l_extendedprice, buf_l_discount, l_nrow,                  //
        l_okey_1, l_eprice_1, l_discount_1, l_1e);
}

static void combine_hj_input(
    // small table input
    hls::stream<ap_uint<W_TPCH_INT> >& o_okey_1,
    hls::stream<bool>& o_1e,
    // big table input
    hls::stream<ap_uint<W_TPCH_INT> >& l_okey_1,
    hls::stream<ap_uint<W_TPCH_INT> >& l_eprice_1,
    hls::stream<ap_uint<W_TPCH_INT> >& l_discount_1,
    hls::stream<bool>& l_1e,
    // output
    hls::stream<ap_uint<W_TPCH_INT> >& kstrm,
    hls::stream<ap_uint<W_TPCH_INT * 2> >& pstrm,
    hls::stream<bool>& estrm) {
#ifndef __SYNTHESIS__
    int cnt = 0;
#endif
    bool oe = 0;
    // send small table
    {
        oe = o_1e.read();
    SMALL_FEEDER_P:
        while (!oe) {
#pragma HLS pipeline
            ap_uint<W_TPCH_INT> k = o_okey_1.read();
            ap_uint<W_TPCH_INT* 2> p = k; // XXX no payload.
            oe = o_1e.read();

            kstrm.write(k);
            pstrm.write(p);
            estrm.write(false);
#ifndef __SYNTHESIS__
            ++cnt;
#endif
        }
        estrm.write(true);
    }
#ifndef __SYNTHESIS__
    printf("feeded %d rows from small table to hash-join.\n", cnt);
    cnt = 0;
#endif
    // send big table
    bool le = l_1e.read();
BIG_FEEDER_P:
    while (!le) {
#pragma HLS pipeline
        ap_uint<W_TPCH_INT> k = l_okey_1.read();
        // write back key, price and discount
        ap_uint<W_TPCH_INT * 3> p;
        p.range(W_TPCH_INT - 1, 0) = l_eprice_1.read();
        p.range(W_TPCH_INT * 2 - 1, W_TPCH_INT) = l_discount_1.read();
        p.range(W_TPCH_INT * 3 - 1, W_TPCH_INT * 2) = k;
        le = l_1e.read();

        kstrm.write(k);
        pstrm.write(p);
        estrm.write(false);
#ifndef __SYNTHESIS__
        ++cnt;
#endif
    }
    estrm.write(true);
#ifndef __SYNTHESIS__
    printf("feeded %d rows from big table to hash-join.\n", cnt);
#endif
}

// CH_NM channel number of key input
template <int CH_NM>
static void scan_wrapper(
    // small table buffer
    ap_uint<W_TPCH_INT * VEC_LEN> buf_o_orderkey[O_DEPTH],
    const int o_nrow,
    // big table buffer
    ap_uint<W_TPCH_INT * VEC_LEN> buf_l_orderkey[L_DEPTH],
    ap_uint<W_TPCH_INT * VEC_LEN> buf_l_extendedprice[L_DEPTH],
    ap_uint<W_TPCH_INT * VEC_LEN> buf_l_discount[L_DEPTH],
    const int l_nrow,
    hls::stream<ap_uint<W_TPCH_INT> > key_ostrm[CH_NM],
    hls::stream<ap_uint<W_TPCH_INT * 2> > pld_ostrm[CH_NM],
    hls::stream<bool> e_ostrm[CH_NM]) {
    hls::stream<ap_uint<W_TPCH_INT> > o_okey_1[CH_NM];
#pragma HLS stream variable = o_okey_1 depth = 32
#pragma HLS array_partition variable = o_okey_1 dim = 0
#pragma HLS bind_storage variable = o_okey_1 type = fifo impl = srl

    hls::stream<bool> o_1e[CH_NM]; // flag stream, when true, end of data.
#pragma HLS stream variable = o_1e depth = 32
#pragma HLS array_partition variable = o_1e dim = 0

    // ------------ scan lineitem -----------
    hls::stream<ap_uint<W_TPCH_INT> > l_okey_1[CH_NM];
#pragma HLS stream variable = l_okey_1 depth = 32
#pragma HLS bind_storage variable = l_okey_1 type = fifo impl = srl
#pragma HLS array_partition variable = l_okey_1 dim = 0

    hls::stream<ap_uint<W_TPCH_INT> > l_eprice_1[CH_NM];
#pragma HLS stream variable = l_eprice_1 depth = 32
#pragma HLS bind_storage variable = l_eprice_1 type = fifo impl = srl
#pragma HLS array_partition variable = l_eprice_1 dim = 0

    hls::stream<ap_uint<W_TPCH_INT> > l_discount_1[CH_NM];
#pragma HLS stream variable = l_discount_1 depth = 32
#pragma HLS bind_storage variable = l_discount_1 type = fifo impl = srl
#pragma HLS array_partition variable = l_discount_1 dim = 0

    hls::stream<bool> l_1e[CH_NM]; // flag stream, when true, end of data.
#pragma HLS stream variable = l_1e depth = 32
#pragma HLS array_partition variable = l_1e dim = 0

    //-------------------------------
    ;

#pragma HLS DATAFLOW

    scan_table<CH_NM>(buf_o_orderkey, o_nrow, //
                      buf_l_orderkey, buf_l_extendedprice, buf_l_discount,
                      l_nrow,         //
                      o_okey_1, o_1e, //
                      l_okey_1, l_eprice_1, l_discount_1, l_1e);

    for (int c = 0; c < CH_NM; ++c) {
#pragma HLS unroll
        combine_hj_input(o_okey_1[c], o_1e[c],                                 //
                         l_okey_1[c], l_eprice_1[c], l_discount_1[c], l_1e[c], //
                         key_ostrm[c], pld_ostrm[c], e_ostrm[c]);
    }
}

void read_status(const int k_bucket, hls::stream<ap_uint<32> >& pu_begin_status_strms) {
    const int hj_begin_status[BUILD_CFG_DEPTH] = {k_bucket, 0};
    for (int i = 0; i < BUILD_CFG_DEPTH; i++) {
        pu_begin_status_strms.write(hj_begin_status[i]);
    }
}

void write_status(hls::stream<ap_uint<32> >& pu_end_status_strms) {
    for (int i = 0; i < BUILD_CFG_DEPTH; i++) pu_end_status_strms.read();
}

extern "C" void join_kernel(ap_uint<W_TPCH_INT * VEC_LEN> buf_o_orderkey[O_DEPTH],
                            const int o_nrow,
                            ap_uint<W_TPCH_INT * VEC_LEN> buf_l_orderkey[L_DEPTH],
                            ap_uint<W_TPCH_INT * VEC_LEN> buf_l_extendedprice[L_DEPTH],
                            ap_uint<W_TPCH_INT * VEC_LEN> buf_l_discount[L_DEPTH],
                            const int l_nrow,
                            // tune
                            const int k_bucket,
                            //
                            ap_uint<256> pu0_ht[PU_HT_DEPTH], // PU0 hash-tables
                            ap_uint<256> pu1_ht[PU_HT_DEPTH], // PU0 hash-tables
                            ap_uint<256> pu2_ht[PU_HT_DEPTH], // PU0 hash-tables
                            ap_uint<256> pu3_ht[PU_HT_DEPTH], // PU0 hash-tables
                            ap_uint<256> pu4_ht[PU_HT_DEPTH], // PU0 hash-tables
                            ap_uint<256> pu5_ht[PU_HT_DEPTH], // PU0 hash-tables
                            ap_uint<256> pu6_ht[PU_HT_DEPTH], // PU0 hash-tables
                            ap_uint<256> pu7_ht[PU_HT_DEPTH], // PU0 hash-tables

                            ap_uint<256> pu0_s[PU_S_DEPTH], // PU0 S units
                            ap_uint<256> pu1_s[PU_S_DEPTH], // PU0 S units
                            ap_uint<256> pu2_s[PU_S_DEPTH], // PU0 S units
                            ap_uint<256> pu3_s[PU_S_DEPTH], // PU0 S units
                            ap_uint<256> pu4_s[PU_S_DEPTH], // PU0 S units
                            ap_uint<256> pu5_s[PU_S_DEPTH], // PU0 S units
                            ap_uint<256> pu6_s[PU_S_DEPTH], // PU0 S units
                            ap_uint<256> pu7_s[PU_S_DEPTH], // PU0 S units
                            // output
                            ap_uint<W_TPCH_INT * 2> buf_result[1]) {
    // clang-format off
  ;
#pragma HLS INTERFACE m_axi offset = slave latency = 64  \
  num_write_outstanding = 16 num_read_outstanding = 16   \
  max_write_burst_length = 64 max_read_burst_length = 64 \
  bundle = gmem0_0 port = buf_o_orderkey

#pragma HLS INTERFACE m_axi offset = slave latency = 64  \
  num_write_outstanding = 16 num_read_outstanding = 16   \
  max_write_burst_length = 64 max_read_burst_length = 64 \
  bundle = gmem0_1 port = buf_l_orderkey
#pragma HLS INTERFACE m_axi offset = slave latency = 64  \
  num_write_outstanding = 16 num_read_outstanding = 16   \
  max_write_burst_length = 64 max_read_burst_length = 64 \
  bundle = gmem0_2 port = buf_l_extendedprice
#pragma HLS INTERFACE m_axi offset = slave latency = 64  \
  num_write_outstanding = 16 num_read_outstanding = 16   \
  max_write_burst_length = 64 max_read_burst_length = 64 \
  bundle = gmem0_3 port = buf_l_discount

#pragma HLS INTERFACE m_axi port=pu0_ht bundle=gmem1_0 \
  num_read_outstanding=32 max_read_burst_length=8      \
  num_write_outstanding=32 latency=125
#pragma HLS INTERFACE m_axi port=pu1_ht bundle=gmem1_1 \
  num_read_outstanding=32 max_read_burst_length=8      \
  num_write_outstanding=32 latency=125
#pragma HLS INTERFACE m_axi port=pu2_ht bundle=gmem1_2 \
  num_read_outstanding=32 max_read_burst_length=8      \
  num_write_outstanding=32 latency=125
#pragma HLS INTERFACE m_axi port=pu3_ht bundle=gmem1_3 \
  num_read_outstanding=32 max_read_burst_length=8      \
  num_write_outstanding=32 latency=125
#pragma HLS INTERFACE m_axi port=pu4_ht bundle=gmem1_4 \
  num_read_outstanding=32 max_read_burst_length=8      \
  num_write_outstanding=32 latency=125
#pragma HLS INTERFACE m_axi port=pu5_ht bundle=gmem1_5 \
  num_read_outstanding=32 max_read_burst_length=8      \
  num_write_outstanding=32 latency=125
#pragma HLS INTERFACE m_axi port=pu6_ht bundle=gmem1_6 \
  num_read_outstanding=32 max_read_burst_length=8      \
  num_write_outstanding=32 latency=125
#pragma HLS INTERFACE m_axi port=pu7_ht bundle=gmem1_7 \
  num_read_outstanding=32 max_read_burst_length=8      \
  num_write_outstanding=32 latency=125

#pragma HLS INTERFACE m_axi port=pu0_s bundle=gmem2_0 \
  num_read_outstanding=32 max_read_burst_length=8     \
  num_write_outstanding=32 latency=125
#pragma HLS INTERFACE m_axi port=pu1_s bundle=gmem2_1 \
  num_read_outstanding=32 max_read_burst_length=8     \
  num_write_outstanding=32 latency=125
#pragma HLS INTERFACE m_axi port=pu2_s bundle=gmem2_2 \
  num_read_outstanding=32 max_read_burst_length=8     \
  num_write_outstanding=32 latency=125
#pragma HLS INTERFACE m_axi port=pu3_s bundle=gmem2_3 \
  num_read_outstanding=32 max_read_burst_length=8     \
  num_write_outstanding=32 latency=125
#pragma HLS INTERFACE m_axi port=pu4_s bundle=gmem2_4 \
  num_read_outstanding=32 max_read_burst_length=8     \
  num_write_outstanding=32 latency=125
#pragma HLS INTERFACE m_axi port=pu5_s bundle=gmem2_5 \
  num_read_outstanding=32 max_read_burst_length=8     \
  num_write_outstanding=32 latency=125
#pragma HLS INTERFACE m_axi port=pu6_s bundle=gmem2_6 \
  num_read_outstanding=32 max_read_burst_length=8     \
  num_write_outstanding=32 latency=125
#pragma HLS INTERFACE m_axi port=pu7_s bundle=gmem2_7 \
  num_read_outstanding=32 max_read_burst_length=8     \
  num_write_outstanding=32 latency=125

#pragma HLS INTERFACE m_axi offset = slave latency = 64  \
  num_write_outstanding = 16 num_read_outstanding = 16   \
  max_write_burst_length = 64 max_read_burst_length = 64 \
  bundle = gmem0_4 port = buf_result

#pragma HLS INTERFACE s_axilite port = buf_o_orderkey bundle = control
#pragma HLS INTERFACE s_axilite port = o_nrow bundle = control
#pragma HLS INTERFACE s_axilite port = buf_l_orderkey bundle = control
#pragma HLS INTERFACE s_axilite port = buf_l_extendedprice bundle = control
#pragma HLS INTERFACE s_axilite port = buf_l_discount bundle = control
#pragma HLS INTERFACE s_axilite port = l_nrow bundle = control
#pragma HLS INTERFACE s_axilite port = k_bucket bundle = control

#pragma HLS INTERFACE s_axilite port = pu0_ht bundle = control
#pragma HLS INTERFACE s_axilite port = pu1_ht bundle = control
#pragma HLS INTERFACE s_axilite port = pu2_ht bundle = control
#pragma HLS INTERFACE s_axilite port = pu3_ht bundle = control
#pragma HLS INTERFACE s_axilite port = pu4_ht bundle = control
#pragma HLS INTERFACE s_axilite port = pu5_ht bundle = control
#pragma HLS INTERFACE s_axilite port = pu6_ht bundle = control
#pragma HLS INTERFACE s_axilite port = pu7_ht bundle = control

#pragma HLS INTERFACE s_axilite port = pu0_s bundle = control
#pragma HLS INTERFACE s_axilite port = pu1_s bundle = control
#pragma HLS INTERFACE s_axilite port = pu2_s bundle = control
#pragma HLS INTERFACE s_axilite port = pu3_s bundle = control
#pragma HLS INTERFACE s_axilite port = pu4_s bundle = control
#pragma HLS INTERFACE s_axilite port = pu5_s bundle = control
#pragma HLS INTERFACE s_axilite port = pu6_s bundle = control
#pragma HLS INTERFACE s_axilite port = pu7_s bundle = control

#pragma HLS INTERFACE s_axilite port = buf_result bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control
    // clang-format on
    ;

#ifndef __SYNTHESIS__
    printf("W_TPCH_INT = %ld\n", W_TPCH_INT);
#endif

#pragma HLS dataflow

    //---------------------------- scan+filter -----------------------//

    hls::stream<ap_uint<W_TPCH_INT> > k0_strm_arry[HJ_CH_NM];
#pragma HLS stream variable = k0_strm_arry depth = 8
#pragma HLS array_partition variable = k0_strm_arry dim = 0
#pragma HLS bind_storage variable = k0_strm_arry type = fifo impl = srl
    hls::stream<ap_uint<W_TPCH_INT * 2> > p0_strm_arry[HJ_CH_NM];
#pragma HLS stream variable = p0_strm_arry depth = 18
#pragma HLS array_partition variable = p0_strm_arry dim = 0
    //#pragma HLS resource variable = p0_strm_arry core = FIFO_SRL
    hls::stream<bool> e0_strm_arry[HJ_CH_NM];
#pragma HLS stream variable = e0_strm_arry depth = 8
#pragma HLS array_partition variable = e0_strm_arry dim = 0

    scan_wrapper<HJ_CH_NM>(buf_o_orderkey, o_nrow, //
                           buf_l_orderkey, buf_l_extendedprice, buf_l_discount,
                           l_nrow, //
                           k0_strm_arry, p0_strm_arry, e0_strm_arry);

#ifndef __SYNTHESIS__
    for (int i = 0; i < HJ_CH_NM; ++i) {
        std::cout << "e0_strm_arry[" << i << "].size() = " << e0_strm_arry[i].size() << std::endl;
    }
#endif

    //---------------------------- hash-join ----------------------//

    hls::stream<ap_uint<W_TPCH_INT * 4> > j1_strm;
#pragma HLS stream variable = j1_strm depth = 70
#pragma HLS bind_storage variable = j1_strm type = fifo impl = srl
    hls::stream<bool> e5_strm;
#pragma HLS stream variable = e5_strm depth = 8

    hls::stream<ap_uint<32> > pu_begin_status_strm;
#pragma HLS stream variable = pu_begin_status_strm depth = 8
    hls::stream<ap_uint<32> > pu_end_status_strm;
#pragma HLS stream variable = pu_end_status_strm depth = 8

    read_status(k_bucket, pu_begin_status_strm);

#ifndef __SYNTHESIS__
    std::cout << "HJ kernel start" << std::endl;
#endif

    xf::database::hashJoinV3<HJ_MODE,        // 0 - radix, 1 - Jenkins
                             W_TPCH_INT,     // key size
                             W_TPCH_INT * 2, // payload maxsize
                             W_TPCH_INT,     // S_PW width of payload of small table.
                             W_TPCH_INT * 2, // B_PW width of payload of big table.
                             HJ_HW_P,        // log2 of PU number
                             HJ_HW_J,        // width of lower hash value
                             24,             // addr width
                             HJ_CH_NM        // channel number
                             >(
        // input
        k0_strm_arry, p0_strm_arry, e0_strm_arry,

        // output hash table
        pu0_ht, // PU0 hash-tables
        pu1_ht, // PU0 hash-tables
        pu2_ht, // PU0 hash-tables
        pu3_ht, // PU0 hash-tables
        pu4_ht, // PU0 hash-tables
        pu5_ht, // PU0 hash-tables
        pu6_ht, // PU0 hash-tables
        pu7_ht, // PU0 hash-tables

        // output PU S unit
        pu0_s, // PU0 S units
        pu1_s, // PU0 S units
        pu2_s, // PU0 S units
        pu3_s, // PU0 S units
        pu4_s, // PU0 S units
        pu5_s, // PU0 S units
        pu6_s, // PU0 S units
        pu7_s, // PU0 S units

        // output join result
        pu_begin_status_strm, pu_end_status_strm,

        j1_strm, e5_strm);

#ifndef __SYNTHESIS__
    std::cout << "HJ kernel end" << std::endl;
#endif

    write_status(pu_end_status_strm);

#ifndef __SYNTHESIS__
    for (int i = 0; i < HJ_CH_NM; ++i) {
        std::cout << "e0_strm_arry[" << i << "].size() = " << e0_strm_arry[i].size() << std::endl;
    }
#endif

    //---------------------------- splitter ----------------------//

    hls::stream<ap_uint<W_TPCH_INT> > l_eprice_2;
#pragma HLS stream variable = l_eprice_2 depth = 8
    hls::stream<ap_uint<W_TPCH_INT> > l_discount_2;
#pragma HLS stream variable = l_discount_2 depth = 8
    hls::stream<bool> e6_strm;
#pragma HLS stream variable = e6_strm depth = 8

    col_splitter(j1_strm, e5_strm, l_eprice_2, l_discount_2, e6_strm);

    //---------------------------- aggregate ----------------------//

    hls::stream<ap_uint<W_TPCH_INT> > x_revenue_1;
#pragma HLS stream variable = x_revenue_1 depth = 4
    hls::stream<bool> x_1e;
#pragma HLS stream variable = x_1e depth = 4

    multiply_cal(l_eprice_2, l_discount_2, e6_strm, // input
                 x_revenue_1, x_1e);                // output

    // extra width for aggreate result
    hls::stream<ap_uint<W_TPCH_INT * 2> > x_revenue_2;
#pragma HLS stream variable = x_revenue_2 depth = 4
    hls::stream<bool> x_2e;
#pragma HLS stream variable = x_2e depth = 4

    xf::database::aggregate<xf::database::AOP_SUM>(x_revenue_1, x_1e,  // input
                                                   x_revenue_2, x_2e); // output

    //-------------------------- write back ----------------------//

    write_result(buf_result, x_revenue_2, x_2e); // write to buffer. (only one row)
}
