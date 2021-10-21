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
#ifndef GQE_AGGR_PART_HPP
#define GQE_AGGR_PART_HPP

#include <ap_int.h>
#include <hls_stream.h>

#include "xf_database/dynamic_eval.hpp"

#include "xf_database/gqe_blocks/gqe_types.hpp"

#ifndef __SYNTHESIS__
#include <iostream>
#endif

namespace xf {
namespace database {
namespace gqe {

template <int N>
void e_sink(hls::stream<bool> e_strm[N]) {
    for (int i = 0; i < N; i++) {
#pragma HLS unroll
        bool e = e_strm[i].read();
        while (!e) {
#pragma HLS pipeline II = 1
            e = e_strm[i].read();
        }
    }
}
// read the dummy end flag
template <int N>
void e_sink(hls::stream<bool> e_strm[N], hls::stream<bool>& o_e_strm) {
    bool e[N];
#pragma HLS array_partition variable = e dim = 0
    for (int i = 0; i < N; i++) {
#pragma HLS unroll
        e[i] = e_strm[i].read();
    }
    while (!e[0]) {
#pragma HLS pipeline II = 1
        for (int i = 0; i < N; i++) {
#pragma HLS unroll
            e[i] = e_strm[i].read();
        }
        o_e_strm.write(false);
    }
    o_e_strm.write(true);
}
template <int N>
void e_dup(hls::stream<bool>& i_e_strm, hls::stream<bool> o_e_strm[N]) {
    bool e = i_e_strm.read();
    while (!e) {
#pragma HLS pipeline II = 1
        e = i_e_strm.read();
        for (int k = 0; k < N; ++k) {
#pragma HLS unroll
            o_e_strm[k].write(false);
        }
    }
    for (int k = 0; k < N; ++k) {
#pragma HLS unroll
        o_e_strm[k].write(true);
    }
}
// duplicate the e_stream for COL_NM times.
template <int COL_NM>
static void add_e_strm(hls::stream<ap_uint<8 * TPCH_INT_SZ> >& in_strm,
                       hls::stream<bool>& e_in_strm,
                       hls::stream<ap_uint<8 * TPCH_INT_SZ> > out_strm[COL_NM],
                       hls::stream<bool> e_out_strm[COL_NM]) {
    bool e = e_in_strm.read();
    while (!e) {
#pragma HLS pipeline II = 1
        e = e_in_strm.read();
        ap_uint<8 * TPCH_INT_SZ> in = in_strm.read();
        for (int i = 0; i < COL_NM; ++i) {
#pragma HLS unroll
            out_strm[i].write(in);
            e_out_strm[i].write(false);
        }
    }
    for (int i = 0; i < COL_NM; ++i) {
#pragma HLS unroll
        e_out_strm[i].write(true);
    }
}
// double stream
template <int COL_NM>
void dup_strm(hls::stream<ap_uint<8 * TPCH_INT_SZ> > in_strm[COL_NM],
              hls::stream<bool>& e_in_strm,
              hls::stream<ap_uint<8 * TPCH_INT_SZ> > out_strm_1[COL_NM],
              hls::stream<bool>& e_out_strm_1,
              hls::stream<ap_uint<8 * TPCH_INT_SZ> > out_strm_2[COL_NM],
              hls::stream<bool>& e_out_strm_2) {
    bool e = e_in_strm.read();
    while (!e) {
#pragma HLS pipeline II = 1
        e = e_in_strm.read();
        for (int i = 0; i < COL_NM; i++) {
#pragma HLS unroll
            ap_uint<8 * TPCH_INT_SZ> tmp = in_strm[i].read();
            out_strm_1[i].write(tmp);
            out_strm_2[i].write(tmp);
        }
        e_out_strm_1.write(false);
        e_out_strm_2.write(false);
    }
    e_out_strm_1.write(true);
    e_out_strm_2.write(true);
}

template <int _Wp>
void aggregate(hls::stream<ap_uint<_Wp> >& in_strm,
               hls::stream<bool>& in_e_strm,
               hls::stream<ap_uint<_Wp> >& out_strm,
               hls::stream<bool>& out_e_strm) {
    ap_int<_Wp> min;
    ap_int<_Wp> max;

    if (_Wp == 32) {
        min = 0x7fffffff;
        max = 0x80000000;
    } else if (_Wp == 64) {
        min = 0x7fffffffffffffff;
        max = 0x8000000000000000;
    } else {
        min = -1;
        min >= 1;
        max = 1;
        max <= (_Wp > 1 ? (_Wp - 1) : 0);
    }

    ap_int<_Wp* 2> sum = 0;
    ap_uint<_Wp> cnt = 0;
    ap_uint<_Wp> cnt_nz = 0;

    bool e = in_e_strm.read();
    if (!e) {
        while (!e) {
#pragma HLS pipeline
            e = in_e_strm.read();
            ap_int<_Wp> t = in_strm.read();
            min = (t < min) ? t : min;
            max = (t > max) ? t : max;
            sum += t;
            ++cnt;
            cnt_nz = t == 0 ? cnt_nz : (ap_uint<_Wp>)(cnt_nz + 1);
        }
        out_strm.write(min);
        out_e_strm.write(false);
        out_strm.write(max);
        out_e_strm.write(false);
        out_strm.write(sum.range(_Wp - 1, 0));
        out_e_strm.write(false);
        out_strm.write(sum.range(_Wp * 2 - 1, _Wp));
        out_e_strm.write(false);
        out_strm.write(cnt);
        out_e_strm.write(false);
        out_strm.write(cnt_nz);
        out_e_strm.write(false);
    }
    out_e_strm.write(true);
}

template <int N>
void multi_agg(hls::stream<ap_uint<8 * TPCH_INT_SZ> > in_strm[N],
               hls::stream<bool>& e_in_strm,
               hls::stream<ap_uint<8 * TPCH_INT_SZ> > out_strm[N],
               hls::stream<bool>& e_out_strm) {
#pragma HLS dataflow

    hls::stream<bool> mid_e_strm[N];
#pragma HLS stream variable = mid_e_strm depth = 2
#pragma HLS array_partition variable = mid_e_strm dim = 0
    hls::stream<bool> e_strm[N];
#pragma HLS stream variable = e_strm depth = 2
#pragma HLS array_partition variable = e_strm dim = 0
    e_dup<N>(e_in_strm, mid_e_strm);
    for (int k = 0; k < N; ++k) {
#pragma HLS unroll
        aggregate(in_strm[k], mid_e_strm[k], out_strm[k], e_strm[k]);
    }
    e_sink<N>(e_strm, e_out_strm);
}

template <int N>
void agg_wrapper(hls::stream<bool>& agg_on_strm,
                 hls::stream<ap_uint<8 * TPCH_INT_SZ> > in_strm[N],
                 hls::stream<bool>& e_in_strm,
                 hls::stream<ap_uint<8 * TPCH_INT_SZ> > out_strm[N],
                 hls::stream<bool>& e_out_strm) {
    ap_uint<8 * TPCH_INT_SZ> temp[N];
    bool agg_on = agg_on_strm.read();
    if (agg_on) {
        multi_agg<N>(in_strm, e_in_strm, out_strm, e_out_strm);
    } else {
        // bypass
        bool e = e_in_strm.read();

#if !defined __SYNTHESIS__ && XDEBUG == 1
        int cnt = 0;
        std::cout << "Column number:" << N << std::endl;
#endif

        while (!e) {
            for (int i = 0; i < N; i++) {
#pragma HLS unroll
                temp[i] = in_strm[i].read();
                out_strm[i].write(temp[i]);
            }

#if !defined __SYNTHESIS__ && XDEBUG == 1
            if (cnt < 10) {
                for (int i = 0; i < 8; i++) {
                    std::cout << "col" << i << ": " << temp[i] << " ";
                }
                std::cout << std::endl;
                for (int i = 8; i < N; i++) {
                    std::cout << "col" << i << ": " << temp[i] << " ";
                }
                std::cout << std::endl;
            }
            cnt++;
#endif

            e = e_in_strm.read();
            e_out_strm.write(false);
        };
        e_out_strm.write(true);
    }
}

static void eval_wrapper(hls::stream<ap_uint<289> >& alu_cfg_strm,
                         hls::stream<ap_uint<8 * TPCH_INT_SZ> >& key0_strm,
                         hls::stream<ap_uint<8 * TPCH_INT_SZ> >& key1_strm,
                         hls::stream<ap_uint<8 * TPCH_INT_SZ> >& key2_strm,
                         hls::stream<ap_uint<8 * TPCH_INT_SZ> >& key3_strm,
                         hls::stream<bool>& e_keys_strm,
                         hls::stream<ap_uint<8 * TPCH_INT_SZ> >& out_strm,
                         hls::stream<bool>& e_out_strm);
static void dynamic_eval_stage(hls::stream<ap_uint<8 * TPCH_INT_SZ> > in_strm[4],
                               hls::stream<bool>& e_in_strm,
                               hls::stream<ap_uint<8 * TPCH_INT_SZ> > out_strm[5],
                               hls::stream<bool>& e_out_strm,
                               hls::stream<ap_uint<289> >& alu_cfg_strm);
static void aggregate(hls::stream<ap_uint<8 * TPCH_INT_SZ> > in_strm[5],
                      hls::stream<bool>& e_in_strm,
                      hls::stream<ap_uint<8 * TPCH_INT_SZ> > out_strm[5],
                      hls::stream<bool>& e_out_strm);

} // namespace gqe
} // namespace database
} // namespace xf

#endif
