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
#include "filter_kernel.hpp"
// used modules
#include <hls_stream.h>

// clang-format off
// In dataflow order
#include "xf_database/scan_col.hpp"
#include "xf_database/combine_split_col.hpp"
#include "xf_database/duplicate_col.hpp"
#include "xf_database/dynamic_filter.hpp"
#include "xf_database/static_eval.hpp"
#include "xf_database/aggregate.hpp"
// clang-format on

ap_uint<8 * MONEY_SZ * 2> calc_revenue(ap_uint<8 * MONEY_SZ> discount, ap_uint<8 * MONEY_SZ> extprice) {
    return discount * extprice;
}

static void flag_sink(hls::stream<bool>& e_strm) {
    bool e;
    while (!(e = e_strm.read()))
        ;
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

extern "C" void filter_kernel(
    // config/op
    ap_uint<32> buf_filter_cfg[xf::database::DynamicFilterInfo<4, 32>::dwords_num],
    // input, condition columns
    ap_uint<8 * KEY_SZ * VEC_LEN> buf_l_shipdate[L_DEPTH],
    ap_uint<8 * MONEY_SZ * VEC_LEN> buf_l_discount[L_DEPTH],
    ap_uint<8 * TPCH_INT_SZ * VEC_LEN> buf_l_quantity[L_DEPTH],
    ap_uint<8 * KEY_SZ * VEC_LEN> buf_l_commitdate[L_DEPTH],
    // input, payload column
    ap_uint<8 * MONEY_SZ * VEC_LEN> buf_l_extendedprice[L_DEPTH],
    // input, size of workload
    const int l_nrow,
    // output
    ap_uint<8 * MONEY_SZ * 2> buf_revenue[1]) {
    // clang-format off
  ;
#pragma HLS INTERFACE m_axi offset = slave latency = 64 \
  num_write_outstanding = 16 num_read_outstanding = 16 \
  max_write_burst_length = 64 max_read_burst_length = 64 \
  bundle = gmem0_0 port = buf_filter_cfg

#pragma HLS INTERFACE m_axi offset = slave latency = 64 \
  num_write_outstanding = 16 num_read_outstanding = 16 \
  max_write_burst_length = 64 max_read_burst_length = 64 \
  bundle = gmem0_1 port = buf_l_shipdate

#pragma HLS INTERFACE m_axi offset = slave latency = 64 \
  num_write_outstanding = 16 num_read_outstanding = 16 \
  max_write_burst_length = 64 max_read_burst_length = 64 \
  bundle = gmem0_2 port = buf_l_discount

#pragma HLS INTERFACE m_axi offset = slave latency = 64 \
  num_write_outstanding = 16 num_read_outstanding = 16 \
  max_write_burst_length = 64 max_read_burst_length = 64 \
  bundle = gmem0_3 port = buf_l_quantity

#pragma HLS INTERFACE m_axi offset = slave latency = 64 \
  num_write_outstanding = 16 num_read_outstanding = 16 \
  max_write_burst_length = 64 max_read_burst_length = 64 \
  bundle = gmem0_4 port = buf_l_commitdate

#pragma HLS INTERFACE m_axi offset = slave latency = 64 \
  num_write_outstanding = 16 num_read_outstanding = 16 \
  max_write_burst_length = 64 max_read_burst_length = 64 \
  bundle = gmem0_5 port = buf_l_extendedprice

#pragma HLS INTERFACE m_axi offset = slave latency = 64 \
  num_write_outstanding = 16 num_read_outstanding = 16 \
  max_write_burst_length = 64 max_read_burst_length = 64 \
  bundle = gmem0_6 port = buf_revenue

#pragma HLS INTERFACE s_axilite port = buf_filter_cfg bundle = control
#pragma HLS INTERFACE s_axilite port = buf_l_shipdate bundle = control
#pragma HLS INTERFACE s_axilite port = buf_l_discount bundle = control
#pragma HLS INTERFACE s_axilite port = buf_l_quantity bundle = control
#pragma HLS INTERFACE s_axilite port = buf_l_commitdate bundle = control
#pragma HLS INTERFACE s_axilite port = buf_l_extendedprice bundle = control
#pragma HLS INTERFACE s_axilite port = l_nrow bundle = control
#pragma HLS INTERFACE s_axilite port = buf_revenue bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control
    // clang-format on
    ;

#pragma HLS dataflow

    hls::stream<typename xf::database::DynamicFilterInfo<4, 32>::cfg_type> filter_cfg_strm;
    hls::stream<bool> e_cfg_strm;

#pragma HLS stream variable = filter_cfg_strm depth = 16
#pragma HLS stream variable = e_cfg_strm depth = 16

#ifndef __SYNTHESIS__
    std::cout << "INFO: reading " << xf::database::DynamicFilterInfo<4, 32>::dwords_num
              << " 32-bit words from buf_filter_cfg..." << std::endl;
#endif

    xf::database::scanCol<BURST_LEN, 1, sizeof(uint32_t)>(
        buf_filter_cfg, xf::database::DynamicFilterInfo<4, 32>::dwords_num, filter_cfg_strm, e_cfg_strm);

    // discard the end flag, as dynamic_filter knows how many dwords to read.
    flag_sink(e_cfg_strm);

    hls::stream<ap_uint<8 * DATE_SZ> > shipdate_strm0;
    hls::stream<ap_uint<8 * MONEY_SZ> > discount_strm0;
    hls::stream<ap_uint<8 * TPCH_INT_SZ> > quantity_strm0;
    hls::stream<ap_uint<8 * DATE_SZ> > commitdate_strm0;
    hls::stream<ap_uint<8 * MONEY_SZ> > extendedprice_strm0;
    hls::stream<bool> e_strm0("e_strm0");

#pragma HLS stream variable = shipdate_strm0 depth = 48
#pragma HLS stream variable = discount_strm0 depth = 16
#pragma HLS stream variable = quantity_strm0 depth = 48
#pragma HLS stream variable = commitdate_strm0 depth = 48
#pragma HLS stream variable = extendedprice_strm0 depth = 48
#pragma HLS stream variable = e_strm0 depth = 16

    xf::database::scanCol<BURST_LEN, VEC_LEN, //
                          DATE_SZ, MONEY_SZ, TPCH_INT_SZ, DATE_SZ, MONEY_SZ>(
        buf_l_shipdate, buf_l_discount, buf_l_quantity, buf_l_commitdate, buf_l_extendedprice, l_nrow, //
        shipdate_strm0, discount_strm0, quantity_strm0, commitdate_strm0, extendedprice_strm0, e_strm0);

    hls::stream<ap_uint<8 * MONEY_SZ> > discount_strm1a("discount_strm1a");
    hls::stream<ap_uint<8 * MONEY_SZ> > discount_strm1b("discount_strm1b");
    hls::stream<bool> e_strm1("e_strm1");

#pragma HLS stream variable = discount_strm1a depth = 32
#pragma HLS stream variable = discount_strm1b depth = 16
#pragma HLS stream variable = e_strm1 depth = 16

#ifndef __SYNTHESIS__
    std::cout << "DEBUG: e_strm0 count = " << e_strm0.size() << std::endl;
#endif

    xf::database::duplicateCol(discount_strm0, e_strm0, discount_strm1a, discount_strm1b, e_strm1);

    hls::stream<ap_uint<8 * (MONEY_SZ + MONEY_SZ)> > pay_strm2("pay_strm2");
    hls::stream<bool> e_strm2("e_strm2");

#pragma HLS stream variable = pay_strm2 depth = 16
#pragma HLS stream variable = e_strm2 depth = 16

#ifndef __SYNTHESIS__
    std::cout << "DEBUG: e_strm1 count = " << e_strm1.size() << std::endl;
#endif

    xf::database::combineCol(discount_strm1b, extendedprice_strm0, e_strm1, pay_strm2, e_strm2);

    hls::stream<ap_uint<8 * (MONEY_SZ + MONEY_SZ)> > pay_strm3("pay_strm3");
    hls::stream<bool> e_strm3("e_strm3");

#pragma HLS stream variable = pay_strm3 depth = 16
#pragma HLS stream variable = e_strm3 depth = 16

#ifndef __SYNTHESIS__
    std::cout << "DEBUG: e_strm2 count = " << e_strm2.size() << std::endl;
#endif

    xf::database::dynamicFilter(filter_cfg_strm, shipdate_strm0, discount_strm1a, quantity_strm0, commitdate_strm0,
                                pay_strm2, e_strm2, //
                                pay_strm3, e_strm3);

    hls::stream<ap_uint<8 * MONEY_SZ> > discount_strm4("discount_strm4");
    hls::stream<ap_uint<8 * MONEY_SZ> > extendedprice_strm4("extendedprice_strm4");
    hls::stream<bool> e_strm4("e_strm4");

#pragma HLS stream variable = discount_strm4 depth = 16
#pragma HLS stream variable = extendedprice_strm4 depth = 16
#pragma HLS stream variable = e_strm4 depth = 16

#ifndef __SYNTHESIS__
    std::cout << "DEBUG: e_strm3 count = " << e_strm3.size() << std::endl;
#endif

    xf::database::splitCol(pay_strm3, e_strm3, //
                           discount_strm4, extendedprice_strm4, e_strm4);

    hls::stream<ap_uint<8 * MONEY_SZ * 2> > revenue_strm5("revenue_strm5");
    hls::stream<bool> e_strm5("e_strm5");

#pragma HLS stream variable = revenue_strm5 depth = 16
#pragma HLS stream variable = e_strm5 depth = 16

#ifndef __SYNTHESIS__
    std::cout << "DEBUG: e_strm4 count = " << e_strm4.size() << std::endl;
#endif

    xf::database::staticEval<ap_uint<8 * MONEY_SZ>, ap_uint<8 * MONEY_SZ>, ap_uint<8 * MONEY_SZ * 2>, calc_revenue>(
        discount_strm4, extendedprice_strm4, e_strm4, revenue_strm5, e_strm5);

    hls::stream<ap_uint<8 * MONEY_SZ * 2> > revenue_strm6("revenue_strm6");
    hls::stream<bool> e_strm6("e_strm6");

#pragma HLS stream variable = revenue_strm6 depth = 16
#pragma HLS stream variable = e_strm6 depth = 16

#ifndef __SYNTHESIS__
    std::cout << "DEBUG: e_strm5 count = " << e_strm5.size() << std::endl;
#endif

    xf::database::aggregate<xf::database::AOP_SUM>(revenue_strm5, e_strm5, //
                                                   revenue_strm6, e_strm6);

#ifndef __SYNTHESIS__
    std::cout << "DEBUG: e_strm6 count = " << e_strm6.size() << std::endl;
#endif

    write_result(buf_revenue, //
                 revenue_strm6, e_strm6);
}
