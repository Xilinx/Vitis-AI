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
#ifndef GQE_FILTER_PART_HPP
#define GQE_FILTER_PART_HPP

#ifndef __SYNTHESIS__
#include <stdio.h>
#include <iostream>
//#define USER_DEBUG true
#endif

#include <hls_stream.h>
#include <ap_int.h>

#include "xf_database/dynamic_filter.hpp"

#include "xf_database/gqe_blocks_v3/gqe_types.hpp"

namespace xf {
namespace database {
namespace gqe {

template <int STRM_NM>
void dup_filter_config(hls::stream<ap_uint<32> >& filter_cfg_single,
                       hls::stream<ap_uint<32> > filter_cfg_strm[STRM_NM]) {
    for (int i = 0; i < xf::database::DynamicFilterInfo<4, 8 * TPCH_INT_SZ>::dwords_num; i++) {
        ap_uint<32> cfg = filter_cfg_single.read();
        for (int j = 0; j < STRM_NM; j++) {
#pragma HLS unroll
            filter_cfg_strm[j].write(cfg);
        }
    }
}

template <int COL_IN_NM, int COL_OUT_NM>
void single_adapter(hls::stream<ap_uint<8 * TPCH_INT_SZ> > in_strm[COL_IN_NM],
                    hls::stream<bool>& e_in_strm,
                    hls::stream<ap_uint<8 * TPCH_INT_SZ> > filter_key_strm[3],
                    hls::stream<ap_uint<8 * TPCH_INT_SZ * COL_OUT_NM> >& filter_pld_strm,
                    hls::stream<bool>& e_filter_pld_strm) {
    bool e;
    while (!(e = e_in_strm.read())) {
#pragma HLS pipeline II = 1
        ap_uint<8 * TPCH_INT_SZ> tmp[COL_IN_NM];
        ap_uint<8 * TPCH_INT_SZ * COL_OUT_NM> tmp_pld;
        for (int i = 0; i < COL_IN_NM; i++) {
#pragma HLS unroll
            tmp[i] = in_strm[i].read();
#ifdef USER_DEBUG
            std::cout << "i: " << i << ", dat: " << tmp[i] << std::endl;
#endif
        }
        for (int i = 0; i < COL_OUT_NM; i++) {
#pragma HLS unroll
            tmp_pld(8 * TPCH_INT_SZ * (i + 1) - 1, 8 * TPCH_INT_SZ * i) = tmp[i];
        }
        for (int i = 0; i < 3; i++) {
#pragma HLS unroll
            filter_key_strm[i].write(tmp[i]);
        }
        filter_pld_strm.write(tmp_pld);
        e_filter_pld_strm.write(false);
    }
    e_filter_pld_strm.write(true);
}

/* XXX this module will insert one extra data when join_on is false. */
template <int COL_NM>
void split_1D_alt(hls::stream<ap_uint<8 * TPCH_INT_SZ * COL_NM> >& in_strm,
                  hls::stream<bool>& e_in_strm,
                  hls::stream<ap_uint<8 * TPCH_INT_SZ> > out_strm[COL_NM],
                  hls::stream<bool>& e_out_strm) {
    bool e;
    while (!(e = e_in_strm.read())) {
#pragma HLS pipeline II = 1
        ap_uint<8 * TPCH_INT_SZ* COL_NM> tmp = in_strm.read();
#ifdef USER_DEBUG
        std::cout << "split1d, " << tmp.range(63, 0) << std::endl;
        std::cout << "split1d, " << tmp.range(127, 64) << std::endl;
        std::cout << "split1d, " << tmp.range(191, 128) << std::endl;
#endif
        for (int i = 0; i < COL_NM; i++) {
#pragma HLS unroll
            out_strm[i].write(tmp(8 * TPCH_INT_SZ * (i + 1) - 1, 8 * TPCH_INT_SZ * i));
        }
        e_out_strm.write(false);
    }
    e_out_strm.write(true);
}

template <int COL_IN_NM, int COL_OUT_NM, int CH_NM>
void filter_ongoing(hls::stream<ap_uint<32> >& filter_cfg_strm,
                    hls::stream<ap_uint<8 * TPCH_INT_SZ> > in_strm[CH_NM][COL_IN_NM],
                    hls::stream<bool> e_in_strm[CH_NM],
                    hls::stream<ap_uint<8 * TPCH_INT_SZ> > out_strm[CH_NM][COL_OUT_NM],
                    hls::stream<bool> e_out_strm[CH_NM]) {
    hls::stream<ap_uint<32> > fcfg_strm[CH_NM];
#pragma HLS stream variable = fcfg_strm depth = 64
#pragma HLS resource variable = fcfg_strm core = FIFO_LUTRAM
    hls::stream<ap_uint<8 * TPCH_INT_SZ> > filter_key_strm[CH_NM][3];
#pragma HLS stream variable = filter_key_strm depth = 64
#pragma HLS resource variable = filter_key_strm core = FIFO_LUTRAM
    hls::stream<ap_uint<8 * TPCH_INT_SZ * COL_OUT_NM> > filter_pld_strm[CH_NM];
#pragma HLS stream variable = filter_pld_strm depth = 64
#pragma HLS resource variable = filter_pld_strm core = FIFO_LUTRAM

    hls::stream<bool> e_filter_pld_strm[CH_NM];
#pragma HLS stream variable = e_filter_pld_strm depth = 64

    hls::stream<ap_uint<8 * TPCH_INT_SZ * COL_OUT_NM> > filter_result_strm[CH_NM];
#pragma HLS stream variable = filter_result_strm depth = 64
#pragma HLS resource variable = filter_result_strm core = FIFO_LUTRAM

    hls::stream<bool> e_filter_result_strm[CH_NM];
#pragma HLS stream variable = e_filter_result_strm depth = 64

#pragma HLS dataflow

    dup_filter_config<CH_NM>(filter_cfg_strm, fcfg_strm);

    for (int i = 0; i < CH_NM; i++) {
#pragma HLS unroll
        single_adapter<COL_IN_NM, COL_OUT_NM>(in_strm[i], e_in_strm[i], filter_key_strm[i], filter_pld_strm[i],
                                              e_filter_pld_strm[i]);

        xf::database::dynamicFilter<8 * TPCH_INT_SZ, 8 * TPCH_INT_SZ * COL_OUT_NM>(
            fcfg_strm[i], filter_key_strm[i][0], filter_key_strm[i][1], filter_key_strm[i][2], filter_pld_strm[i],
            e_filter_pld_strm[i], filter_result_strm[i], e_filter_result_strm[i]);

        split_1D_alt<COL_OUT_NM>(filter_result_strm[i], e_filter_result_strm[i], out_strm[i], e_out_strm[i]);
    }
}

} // namespace gqe
} // namespace database
} // namespace xf

#endif
