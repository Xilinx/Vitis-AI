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
#include "hashjoinkernel.hpp"
// used modules
#include <hls_stream.h>
#define URAM_SPLITTING 1

#include "xf_database/scan_col.hpp"
#include "xf_database/hash_semi_join.hpp"
#include "xf_database/aggregate.hpp"

template <int PAY_SZ>
static void filter_date(hls::stream<ap_uint<8 * DATE_SZ> >& i_date_strm,
                        hls::stream<ap_uint<PAY_SZ> >& i_payload_strm,
                        hls::stream<bool>& i_e_strm,
                        hls::stream<ap_uint<PAY_SZ> >& o_payload_strm,
                        hls::stream<bool>& o_e_strm) {
#ifndef __SYNTHESIS__
    int cnt = 0;
    int nd = 0;
#endif
    bool e = 0; // i_e_strm.read();
    for (int i = 0; i < 2; i++) {
        e = i_e_strm.read();
#ifndef __SYNTHESIS__
        cnt = 0;
        nd = 0;
#endif
    FILTER_DATE_P:
        while (!e) {
#pragma HLS pipeline
            ap_uint<8 * DATE_SZ> date = i_date_strm.read();
            ap_uint<PAY_SZ> payload = i_payload_strm.read();
            e = i_e_strm.read();

            if (date >= 19940101L && date < 19950101L) {
                o_payload_strm.write(payload);
                o_e_strm.write(false);
#ifndef __SYNTHESIS__
                ++cnt;
#if defined(Q5DEBUG) && defined(VERBOSE)
                printf("okey = %d\n", payload.to_uint());
#endif
#endif
            }
#ifndef __SYNTHESIS__
            else {
                ++nd;
            }
#endif
        }
        o_e_strm.write(true);
#ifndef __SYNTHESIS__
        printf("filtered %d rows, dropped %d rows.\n", cnt, nd);
#endif
    }
}

/* small table at MSB, big table at LSB */
static void col_splitter(hls::stream<ap_uint<8 * MONEY_SZ * 2> >& jstrm,
                         hls::stream<bool>& ejstrm,
                         //
                         hls::stream<ap_uint<8 * MONEY_SZ> >& l_eprice_2,
                         hls::stream<ap_uint<8 * MONEY_SZ> >& l_discount_2,
                         hls::stream<bool>& l_2e) {
#ifndef __SYNTHESIS__
    int cnt = 0;
#endif
    bool e = ejstrm.read();
COL_SPLITTER_P:
    while (!e) {
#pragma HLS pipeline
        ap_uint<8 * MONEY_SZ* 2> jrow = jstrm.read();
        // drop 1bit from small table dummy payload
        ap_uint<8 * MONEY_SZ> eprice = jrow.range(8 * MONEY_SZ - 1, 0);
        ap_uint<8 * MONEY_SZ> discount = jrow.range(8 * MONEY_SZ * 2 - 1, 8 * MONEY_SZ);
        // ap_uint<8 * KEY_SZ> okey = jrow.range(8 * (KEY_SZ + MONEY_SZ * 2) - 1, 8
        // * MONEY_SZ * 2);
        e = ejstrm.read();

        l_eprice_2.write(eprice);
        l_discount_2.write(discount);
        l_2e.write(false);
#ifndef __SYNTHESIS__
        ++cnt;
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
static void multiply_cal(hls::stream<ap_uint<8 * MONEY_SZ> >& eprice_strm,
                         hls::stream<ap_uint<8 * MONEY_SZ> >& discount_strm,
                         hls::stream<bool>& i_e_strm,
                         hls::stream<ap_uint<8 * MONEY_SZ> >& revenue_strm,
                         hls::stream<bool>& o_e_strm) {
#ifndef __SYNTHESIS__
    int cnt = 0;
#endif
    bool e = i_e_strm.read();
MULTIPLY_P:
    while (!e) {
#pragma HLS pipeline
        ap_uint<8 * MONEY_SZ> p = eprice_strm.read();
        ap_uint<8 * MONEY_SZ> d = discount_strm.read();
        e = i_e_strm.read();

        ap_uint<8 * MONEY_SZ> r = p * (100 - d); // FIXME overflow?
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

static void write_result(ap_uint<8 * MONEY_SZ * 2> buf_result[],
                         hls::stream<ap_uint<8 * MONEY_SZ * 2> >& d_strm,
                         hls::stream<bool>& e_strm) {
    bool e = e_strm.read();
    int i = 0;
WRITE_RESULT_P:
    while (!e) {
#pragma HLS pipeline
        ap_uint<8 * MONEY_SZ* 2> t = d_strm.read();
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
    ap_uint<8 * KEY_SZ * VEC_LEN> buf_o_orderkey[O_DEPTH],
    ap_uint<8 * DATE_SZ * VEC_LEN> buf_o_orderdate[O_DEPTH],
    const int o_nrow,
    // big table buffer
    ap_uint<8 * KEY_SZ * VEC_LEN> buf_l_orderkey[L_DEPTH],
    ap_uint<8 * MONEY_SZ * VEC_LEN> buf_l_extendedprice[L_DEPTH],
    ap_uint<8 * MONEY_SZ * VEC_LEN> buf_l_discount[L_DEPTH],
    const int l_nrow,
    // small table stream
    hls::stream<ap_uint<8 * KEY_SZ> > o_okey_1[CH_NM],
    hls::stream<ap_uint<8 * DATE_SZ> > o_odate_1[CH_NM],
    hls::stream<bool> o_1e[CH_NM],
    // big table stream
    hls::stream<ap_uint<8 * KEY_SZ> > l_okey_1[CH_NM],
    hls::stream<ap_uint<8 * MONEY_SZ> > l_eprice_1[CH_NM],
    hls::stream<ap_uint<8 * MONEY_SZ> > l_discount_1[CH_NM],
    hls::stream<bool> l_1e[CH_NM]) {
    for (int r = 0; r < 2; ++r) {
        xf::database::scanCol<64, VEC_LEN, CH_NM, KEY_SZ, DATE_SZ>( //
            buf_o_orderkey, buf_o_orderdate, o_nrow,                //
            o_okey_1, o_odate_1, o_1e);
    }

    xf::database::scanCol<64, VEC_LEN, CH_NM, KEY_SZ, MONEY_SZ, MONEY_SZ>( //
        buf_l_orderkey, buf_l_extendedprice, buf_l_discount, l_nrow,       //
        l_okey_1, l_eprice_1, l_discount_1, l_1e);
}

static void small_big_table_feeder_for_part(
    // small table input
    hls::stream<ap_uint<8 * KEY_SZ> >& o_okey_1,
    hls::stream<bool>& o_1e,
    // big table input
    hls::stream<ap_uint<8 * KEY_SZ> >& l_okey_1,
    hls::stream<ap_uint<8 * MONEY_SZ> >& l_eprice_1,
    hls::stream<ap_uint<8 * MONEY_SZ> >& l_discount_1,
    hls::stream<bool>& l_1e,
    // output
    hls::stream<ap_uint<8 * KEY_SZ> >& kstrm,
    hls::stream<ap_uint<8 * MONEY_SZ * 2> >& pstrm,
    hls::stream<bool>& estrm) {
#ifndef __SYNTHESIS__
    int cnt = 0;
#endif
    bool oe = 0;
    // send small table
    for (int r = 0; r < 2; r++) {
        oe = o_1e.read();
    SMALL_FEEDER_P:
        while (!oe) {
#pragma HLS pipeline
            ap_uint<8 * KEY_SZ> k = o_okey_1.read();
            ap_uint<8 * (MONEY_SZ + MONEY_SZ)> p = k; // no payload.
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
        ap_uint<8 * KEY_SZ> k = l_okey_1.read();
        // write back key, price and discount
        ap_uint<8 * (KEY_SZ + MONEY_SZ + MONEY_SZ)> p;
        p.range(8 * MONEY_SZ - 1, 0) = l_eprice_1.read();
        p.range(8 * MONEY_SZ * 2 - 1, 8 * MONEY_SZ) = l_discount_1.read();
        p.range(8 * (KEY_SZ + MONEY_SZ + MONEY_SZ) - 1, 8 * MONEY_SZ * 2) = k;
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
static void scan_wrapper(ap_uint<8 * KEY_SZ * VEC_LEN> buf_o_orderkey[O_DEPTH],
                         ap_uint<8 * DATE_SZ * VEC_LEN> buf_o_orderdate[O_DEPTH],
                         const int o_nrow,
                         // big table buffer
                         ap_uint<8 * KEY_SZ * VEC_LEN> buf_l_orderkey[L_DEPTH],
                         ap_uint<8 * MONEY_SZ * VEC_LEN> buf_l_extendedprice[L_DEPTH],
                         ap_uint<8 * MONEY_SZ * VEC_LEN> buf_l_discount[L_DEPTH],
                         const int l_nrow,
                         hls::stream<ap_uint<8 * KEY_SZ> > o_key_strm[CH_NM],
                         hls::stream<ap_uint<8 * 2 * MONEY_SZ> > o_pld_strm[CH_NM],
                         hls::stream<bool> o_e_strm[CH_NM]) {
    hls::stream<ap_uint<8 * (KEY_SZ)> > o_okey_1[CH_NM];
#pragma HLS stream variable = o_okey_1 depth = 32
#pragma HLS array_partition variable = o_okey_1 dim = 0
#pragma HLS bind_storage variable = o_okey_1 type = fifo impl = srl
    hls::stream<ap_uint<8 * (DATE_SZ)> > o_odate_1[CH_NM];
#pragma HLS stream variable = o_odate_1 depth = 32
#pragma HLS array_partition variable = o_odate_1 dim = 0
#pragma HLS bind_storage variable = o_odate_1 type = fifo impl = srl
    hls::stream<bool> o_1e[CH_NM]; // flag stream, when true, end of data.
#pragma HLS stream variable = o_1e depth = 32
#pragma HLS array_partition variable = o_1e dim = 0

    // ------------ scan lineitem -----------
    hls::stream<ap_uint<8 * (KEY_SZ)> > l_okey_1[CH_NM];
#pragma HLS stream variable = l_okey_1 depth = 32
#pragma HLS bind_storage variable = l_okey_1 type = fifo impl = srl
#pragma HLS array_partition variable = l_okey_1 dim = 0
    hls::stream<ap_uint<8 * (MONEY_SZ)> > l_eprice_1[CH_NM];
#pragma HLS stream variable = l_eprice_1 depth = 32
#pragma HLS bind_storage variable = l_eprice_1 type = fifo impl = srl
#pragma HLS array_partition variable = l_eprice_1 dim = 0
    hls::stream<ap_uint<8 * (MONEY_SZ)> > l_discount_1[CH_NM];
#pragma HLS stream variable = l_discount_1 depth = 32
#pragma HLS bind_storage variable = l_discount_1 type = fifo impl = srl
#pragma HLS array_partition variable = l_discount_1 dim = 0
    hls::stream<bool> l_1e[CH_NM]; // flag stream, when true, end of data.
#pragma HLS stream variable = l_1e depth = 32
#pragma HLS array_partition variable = l_1e dim = 0
    //----------------------------------
    hls::stream<ap_uint<8 * (KEY_SZ)> > o_okey_2[CH_NM];
#pragma HLS stream variable = o_okey_2 depth = 16
#pragma HLS array_partition variable = o_okey_2 dim = 0
#pragma HLS bind_storage variable = o_okey_2 type = fifo impl = srl
    hls::stream<bool> o_2e[CH_NM]; // flag stream, when true, end of data.
#pragma HLS stream variable = o_2e depth = 16
#pragma HLS array_partition variable = o_2e dim = 0
//-------------------------------
#pragma HLS DATAFLOW
    scan_table<CH_NM>(buf_o_orderkey, buf_o_orderdate, o_nrow, buf_l_orderkey, buf_l_extendedprice, buf_l_discount,
                      l_nrow, o_okey_1, o_odate_1, o_1e, l_okey_1, l_eprice_1, l_discount_1, l_1e);
    for (int c = 0; c < CH_NM; ++c) {
#pragma HLS unroll
        filter_date(o_odate_1[c], o_okey_1[c], o_1e[c], o_okey_2[c], o_2e[c]);
        small_big_table_feeder_for_part(o_okey_2[c], o_2e[c], l_okey_1[c], l_eprice_1[c], l_discount_1[c], l_1e[c],
                                        o_key_strm[c], o_pld_strm[c], o_e_strm[c]);
    }
}

extern "C" void join_kernel(ap_uint<8 * KEY_SZ * VEC_LEN> buf_l_orderkey[L_DEPTH],
                            ap_uint<8 * MONEY_SZ * VEC_LEN> buf_l_extendedprice[L_DEPTH],
                            ap_uint<8 * MONEY_SZ * VEC_LEN> buf_l_discount[L_DEPTH],
                            const int l_nrow,
                            ap_uint<8 * KEY_SZ * VEC_LEN> buf_o_orderkey[O_DEPTH],
                            ap_uint<8 * DATE_SZ * VEC_LEN> buf_o_orderdate[O_DEPTH],
                            const int o_nrow,
                            // temp PU = 8
                            ap_uint<8 * KEY_SZ> buf0[BUFF_DEPTH],
                            ap_uint<8 * KEY_SZ> buf1[BUFF_DEPTH],
                            ap_uint<8 * KEY_SZ> buf2[BUFF_DEPTH],
                            ap_uint<8 * KEY_SZ> buf3[BUFF_DEPTH],
                            ap_uint<8 * KEY_SZ> buf4[BUFF_DEPTH],
                            ap_uint<8 * KEY_SZ> buf5[BUFF_DEPTH],
                            ap_uint<8 * KEY_SZ> buf6[BUFF_DEPTH],
                            ap_uint<8 * KEY_SZ> buf7[BUFF_DEPTH],
                            // output
                            ap_uint<8 * MONEY_SZ * 2> buf_result[1]) {
    // clang-format off
  ;
#pragma HLS INTERFACE m_axi offset = slave latency = 64 \
  num_write_outstanding = 16 num_read_outstanding = 16 \
  max_write_burst_length = 64 max_read_burst_length = 64 \
  bundle = gmem0_0 port = buf_l_orderkey

#pragma HLS INTERFACE m_axi offset = slave latency = 64 \
  num_write_outstanding = 16 num_read_outstanding = 16 \
  max_write_burst_length = 64 max_read_burst_length = 64 \
  bundle = gmem0_1 port = buf_l_extendedprice

#pragma HLS INTERFACE m_axi offset = slave latency = 64 \
  num_write_outstanding = 16 num_read_outstanding = 16 \
  max_write_burst_length = 64 max_read_burst_length = 64 \
  bundle = gmem0_2 port = buf_l_discount

#pragma HLS INTERFACE m_axi offset = slave latency = 64 \
  num_write_outstanding = 16 num_read_outstanding = 16 \
  max_write_burst_length = 64 max_read_burst_length = 64 \
  bundle = gmem0_3 port = buf_o_orderkey

#pragma HLS INTERFACE m_axi offset = slave latency = 64 \
  num_write_outstanding = 16 num_read_outstanding = 16 \
  max_write_burst_length = 64 max_read_burst_length = 64 \
  bundle = gmem0_4 port = buf_o_orderdate

#pragma HLS INTERFACE m_axi offset = slave latency = 64 \
  num_write_outstanding = 16 num_read_outstanding = 16 \
  max_write_burst_length = 64 max_read_burst_length = 64 \
  bundle = gmem0_5 port = buf_result

#pragma HLS INTERFACE m_axi port=buf0 bundle=gmem1_0 num_write_outstanding=32 num_read_outstanding=32 max_read_burst_length=8 latency=125
#pragma HLS INTERFACE m_axi port=buf1 bundle=gmem1_1 num_write_outstanding=32 num_read_outstanding=32 max_read_burst_length=8 latency=125
#pragma HLS INTERFACE m_axi port=buf2 bundle=gmem1_2 num_write_outstanding=32 num_read_outstanding=32 max_read_burst_length=8 latency=125
#pragma HLS INTERFACE m_axi port=buf3 bundle=gmem1_3 num_write_outstanding=32 num_read_outstanding=32 max_read_burst_length=8 latency=125
#pragma HLS INTERFACE m_axi port=buf4 bundle=gmem1_4 num_write_outstanding=32 num_read_outstanding=32 max_read_burst_length=8 latency=125
#pragma HLS INTERFACE m_axi port=buf5 bundle=gmem1_5 num_write_outstanding=32 num_read_outstanding=32 max_read_burst_length=8 latency=125
#pragma HLS INTERFACE m_axi port=buf6 bundle=gmem1_6 num_write_outstanding=32 num_read_outstanding=32 max_read_burst_length=8 latency=125
#pragma HLS INTERFACE m_axi port=buf7 bundle=gmem1_7 num_write_outstanding=32 num_read_outstanding=32 max_read_burst_length=8 latency=125

#pragma HLS INTERFACE s_axilite port = buf_l_orderkey bundle = control
#pragma HLS INTERFACE s_axilite port = buf_l_extendedprice bundle = control
#pragma HLS INTERFACE s_axilite port = buf_l_discount bundle = control
#pragma HLS INTERFACE s_axilite port = l_nrow bundle = control
#pragma HLS INTERFACE s_axilite port = buf_o_orderkey bundle = control
#pragma HLS INTERFACE s_axilite port = buf_o_orderdate bundle = control
#pragma HLS INTERFACE s_axilite port = o_nrow bundle = control
#pragma HLS INTERFACE s_axilite port = buf_result bundle = control
#pragma HLS INTERFACE s_axilite port = buf0 bundle = control
#pragma HLS INTERFACE s_axilite port = buf1 bundle = control
#pragma HLS INTERFACE s_axilite port = buf2 bundle = control
#pragma HLS INTERFACE s_axilite port = buf3 bundle = control
#pragma HLS INTERFACE s_axilite port = buf4 bundle = control
#pragma HLS INTERFACE s_axilite port = buf5 bundle = control
#pragma HLS INTERFACE s_axilite port = buf6 bundle = control
#pragma HLS INTERFACE s_axilite port = buf7 bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control
    // clang-format on
    ;

#ifndef __SYNTHESIS__
    printf("KEY_SZ = %ld, MONEY_SZ = %ld, DATE_SZ = %ld\n", KEY_SZ, MONEY_SZ, DATE_SZ);
#endif

#pragma HLS dataflow

    //---------------------------- scan+filter -----------------------//

    hls::stream<ap_uint<8 * KEY_SZ> > k0_strm_arry[HJ_CH_NM];
#pragma HLS stream variable = k0_strm_arry depth = 8
#pragma HLS array_partition variable = k0_strm_arry dim = 0
#pragma HLS bind_storage variable = k0_strm_arry type = fifo impl = srl
    hls::stream<ap_uint<8 * (MONEY_SZ + MONEY_SZ)> > p0_strm_arry[HJ_CH_NM];
#pragma HLS stream variable = p0_strm_arry depth = 18
#pragma HLS array_partition variable = p0_strm_arry dim = 0
    //#pragma HLS resource variable = p0_strm_arry core = FIFO_SRL
    hls::stream<bool> e0_strm_arry[HJ_CH_NM];
#pragma HLS stream variable = e0_strm_arry depth = 8
#pragma HLS array_partition variable = e0_strm_arry dim = 0

    scan_wrapper<HJ_CH_NM>(buf_o_orderkey, buf_o_orderdate, o_nrow, buf_l_orderkey, buf_l_extendedprice, buf_l_discount,
                           l_nrow, k0_strm_arry, p0_strm_arry, e0_strm_arry);

    //---------------------------- hash-join ----------------------//

    hls::stream<ap_uint<8 * (MONEY_SZ + MONEY_SZ)> > j1_strm;
#pragma HLS stream variable = j1_strm depth = 70
#pragma HLS bind_storage variable = j1_strm type = fifo impl = srl
    hls::stream<bool> e5_strm;
#pragma HLS stream variable = e5_strm depth = 8

    xf::database::hashSemiJoin<HJ_MODE,                   // hash algorithm
                               8 * KEY_SZ,                // key width
                               8 * (MONEY_SZ + MONEY_SZ), // payload max width
                               HJ_HW_P,                   // log2(number of PU)
                               HJ_HW_J,                   // hash width for join
                               HJ_AW,                     // address width
                               8 * KEY_SZ,                // buffer width
                               HJ_CH_NM,                  // channel number
                               32, 0>                     //
        (k0_strm_arry, p0_strm_arry, e0_strm_arry,        //
         buf0, buf1, buf2, buf3, buf4, buf5, buf6, buf7,  //
         j1_strm, e5_strm);

    //---------------------------- splitter ----------------------//

    hls::stream<ap_uint<8 * (MONEY_SZ)> > l_eprice_2;
#pragma HLS stream variable = l_eprice_2 depth = 8
    hls::stream<ap_uint<8 * (MONEY_SZ)> > l_discount_2;
#pragma HLS stream variable = l_discount_2 depth = 8
    hls::stream<bool> e6_strm;
#pragma HLS stream variable = e6_strm depth = 8

    col_splitter(j1_strm, e5_strm, l_eprice_2, l_discount_2, e6_strm);

    //---------------------------- aggregate ----------------------//

    hls::stream<ap_uint<8 * (MONEY_SZ)> > x_revenue_1;
#pragma HLS stream variable = x_revenue_1 depth = 4
    hls::stream<bool> x_1e;
#pragma HLS stream variable = x_1e depth = 4

    multiply_cal(l_eprice_2, l_discount_2, e6_strm, // input
                 x_revenue_1, x_1e);                // output

    // extra width for aggreate result
    hls::stream<ap_uint<8 * (MONEY_SZ)*2> > x_revenue_2;
#pragma HLS stream variable = x_revenue_2 depth = 4
    hls::stream<bool> x_2e;
#pragma HLS stream variable = x_2e depth = 4

    xf::database::aggregate<xf::database::AOP_SUM>(x_revenue_1, x_1e,  // input
                                                   x_revenue_2, x_2e); // output

    //-------------------------- write back ----------------------//

    write_result(buf_result, x_revenue_2, x_2e); // write to buffer. (only one row)
}
