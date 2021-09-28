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
#ifndef GQE_ISV_LOAD_CONFIG_AGGR_HPP
#define GQE_ISV_LOAD_CONFIG_AGGR_HPP

#ifndef __SYNTHESIS__
#include <stdio.h>
#include <iostream>
//#define USER_DEBUG true
#endif

#include <ap_int.h>
#include <hls_stream.h>

#include "xf_database/gqe_blocks/gqe_types.hpp"

namespace xf {
namespace database {
namespace gqe {

// for gqeAggr
template <int CHNM, int ColNM>
void load_config(ap_uint<8 * TPCH_INT_SZ>* ptr,
                 ap_uint<512>* tin_meta,
                 hls::stream<int>& nrow_strm,
                 hls::stream<int8_t>& col_id_strm,
                 hls::stream<ap_uint<32> >& alu1_cfg_strm,
                 hls::stream<ap_uint<32> >& alu2_cfg_strm,
                 hls::stream<ap_uint<32> >& filter_cfg_strm,
                 hls::stream<ap_uint<ColNM * ColNM> > shuffle1_cfg_strm[CHNM],
                 hls::stream<ap_uint<ColNM * ColNM> > shuffle2_cfg_strm[CHNM],
                 hls::stream<ap_uint<ColNM * ColNM> > shuffle3_cfg_strm[CHNM],
                 hls::stream<ap_uint<ColNM * ColNM> > shuffle4_cfg_strm[CHNM],
                 hls::stream<ap_uint<32> >& merge_column_cfg_strm,
                 hls::stream<ap_uint<32> >& group_aggr_cfg_strm,
                 hls::stream<bool>& direct_aggr_cfg_strm,
                 hls::stream<ap_uint<32> >& write_out_cfg_strm) {
    // read in the number of rows for each column
    int nrow = tin_meta[0].range(71, 8);
    nrow_strm.write(nrow);

    ap_uint<8 * TPCH_INT_SZ> config[128];
#pragma HLS bind_storage variable = config type = ram_1p impl = bram

    int8_t col_id[8];
    ap_uint<32> alu1_cfg[10];
    ap_uint<32> alu2_cfg[10];
    ap_uint<32> filter_cfg[45];
#pragma HLS bind_storage variable = filter_cfg type = ram_1p impl = lutram
    ap_uint<ColNM * ColNM> shuffle1_cfg;
    ap_uint<ColNM * ColNM> shuffle2_cfg;
    ap_uint<ColNM * ColNM> shuffle3_cfg;
    ap_uint<ColNM * ColNM> shuffle4_cfg;
    ap_uint<32> merge_column_cfg[2];
    ap_uint<32> group_aggr_cfg[4];
    bool direct_aggr_cfg;
    ap_uint<32> write_out_cfg;

    // store
    for (int i = 0; i < 128; i++) {
        config[i] = ptr[i];

#if !defined __SYNTHESIS__ && XDEBUG == 1
        std::cout << std::dec << "config: id=" << i << std::hex << " value=" << config[i] << std::endl;
#endif // !defined __SYNTHESIS__ && XDEBUG == 1
    }

    // scan
    for (int i = 0; i < 4; i++) {
        col_id[i] = config[0].range(8 * (i + 1) - 1, 8 * i);
        col_id[i + 4] = config[1].range(8 * (i + 1) - 1, 8 * i);
    }

    // alu
    alu1_cfg[0] = config[11];
    alu1_cfg[1] = config[10];
    alu1_cfg[2] = config[9];
    alu1_cfg[3] = config[8];
    alu1_cfg[4] = config[7];
    alu1_cfg[5] = config[6];
    alu1_cfg[6] = config[5];
    alu1_cfg[7] = config[4];
    alu1_cfg[8] = config[3];
    alu1_cfg[9] = config[2];

    alu2_cfg[0] = config[21];
    alu2_cfg[1] = config[20];
    alu2_cfg[2] = config[19];
    alu2_cfg[3] = config[18];
    alu2_cfg[4] = config[17];
    alu2_cfg[5] = config[16];
    alu2_cfg[6] = config[15];
    alu2_cfg[7] = config[14];
    alu2_cfg[8] = config[13];
    alu2_cfg[9] = config[12];

    // filter
    for (int i = 0; i < 45; i++) {
        filter_cfg[i] = config[22 + i];
    }

    // shuffle
    shuffle1_cfg(31, 0) = config[67];
    shuffle1_cfg(63, 32) = config[68];

    shuffle2_cfg(31, 0) = config[69];
    shuffle2_cfg(63, 32) = config[70];

    shuffle3_cfg(31, 0) = config[71];
    shuffle3_cfg(63, 32) = config[72];

    shuffle4_cfg(31, 0) = config[73];
    shuffle4_cfg(63, 32) = config[74];

    // group aggr
    for (int i = 0; i < 4; i++) {
        group_aggr_cfg[i] = config[i + 75];
    }

    // merge column
    merge_column_cfg[0] = config[79];
    merge_column_cfg[1] = config[80];

    // direct aggr
    direct_aggr_cfg = config[81][0] == 1 ? true : false;

    // write out
    write_out_cfg = config[82];

#if !defined __SYNTHESIS__ && XDEBUG == 1
    std::cout << std::hex << "write out config" << write_out_cfg << std::endl;
#endif // !defined __SYNTHESIS__ && XDEBUG == 1

    // output
    for (int i = 0; i < 8; i++) {
        col_id_strm.write(col_id[i]);
    }

    for (int i = 0; i < 4; i++) {
        shuffle1_cfg_strm[i].write(shuffle1_cfg);
        shuffle2_cfg_strm[i].write(shuffle2_cfg);
        shuffle3_cfg_strm[i].write(shuffle3_cfg);
        shuffle4_cfg_strm[i].write(shuffle4_cfg);

        group_aggr_cfg_strm.write(group_aggr_cfg[i]);
    }

    for (int i = 0; i < 10; i++) {
        alu1_cfg_strm.write(alu1_cfg[i]);
        alu2_cfg_strm.write(alu2_cfg[i]);
    }

    for (int i = 0; i < 45; i++) {
        filter_cfg_strm.write(filter_cfg[i]);
    }

    merge_column_cfg_strm.write(merge_column_cfg[0]);
    merge_column_cfg_strm.write(merge_column_cfg[1]);
    direct_aggr_cfg_strm.write(direct_aggr_cfg);
    write_out_cfg_strm.write(write_out_cfg);
}

} // namespace gqe
} // namespace database
} // namespace xf

#endif
