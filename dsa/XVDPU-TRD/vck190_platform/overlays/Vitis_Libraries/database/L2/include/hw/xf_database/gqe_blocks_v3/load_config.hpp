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
#ifndef GQE_ISV_LOAD_CONFIG_HPP
#define GQE_ISV_LOAD_CONFIG_HPP

#ifndef __SYNTHESIS__
#include <stdio.h>
#include <iostream>
//#define USER_DEBUG true
#endif

#include <ap_int.h>
#include <hls_stream.h>

#include "xf_database/gqe_blocks_v3/gqe_types.hpp"

namespace xf {
namespace database {
namespace gqe {

/*
// load kernel config for gqeJoin kernel
static void load_config(bool build_probe_flag,
                        ap_uint<512> din_krn_cfg[14],
                        ap_uint<512> din_meta[24],
                        int64_t& nrow,
                        int& secID,
                        hls::stream<ap_uint<6> >& join_cfg_strm,
                        hls::stream<ap_uint<32> >& filter_cfg_strm,
                        ap_uint<3>& din_col_en,
                        ap_uint<2>& rowid_flags,
                        hls::stream<ap_uint<8> >& write_out_cfg_strm) {
#ifdef USER_DEBUG
    std::cout << "-------- load kernel config --------" << std::endl;
#endif
    const int filter_cfg_depth = 53;

    // read in the number of rows
    nrow = din_meta[0].range(71, 8);
    // read in the secID and pass to scan_cols module
    secID = din_meta[0].range(167, 136);
#ifdef USER_DEBUG
    std::cout << "nrow = " << nrow << std::endl;
#endif
    // read in krn_cfg from ddr
    ap_uint<512> config[14];
    for (int i = 0; i < 14; ++i) {
        config[i] = din_krn_cfg[i];
    }

    // define join_cfg to represent the join cfg for each module:
    // bit5: build_probe_flag; bit4-2: join type; bit1: dual key on or off; bit0: join on or bypass
    ap_uint<6> join_cfg;

    // join or bypass
    join_cfg[0] = config[0][0];
#ifdef USER_DEBUG
    std::cout << "in load_cfg: join_on: " << join_cfg[0] << std::endl;
#endif

    // single key or dual key?
    join_cfg[1] = config[0][2];

    // join type: normal, semi- or anti-
    join_cfg.range(3, 2) = config[0].range(4, 3);
    // build_probe_flag
    join_cfg[5] = build_probe_flag;
    join_cfg_strm.write(join_cfg);

    // cfg flags used in build phase
    if (!build_probe_flag) {
        // read in input col enable flag for table A
        din_col_en = config[0].range(8, 6);
    } else { // cfg flags used in probe phase
        // read in input col enable flag for table B
        din_col_en = config[0].range(11, 9);
    }

    // gen_rowid and valid en flag
    // 19/21: gen_rowid_en; 20/22: valid_en
    if (!build_probe_flag) {
        rowid_flags = config[0].range(20, 19);
    } else {
        rowid_flags = config[0].range(22, 21);
    }

    // write out col cfg
    ap_uint<8> write_out_en;
    // append mode
    write_out_en[7] = config[0][5];
    // write out en
    write_out_en.range(3, 0) = config[0].range(15, 12);
    write_out_cfg_strm.write(write_out_en);
#ifdef USER_DEBUG
    std::cout << "write_out_en: " << (int)write_out_en << std::endl;
#endif

    ap_uint<32> filter_cfg_a[filter_cfg_depth];
#pragma HLS resource variable = filter_cfg_a core = RAM_1P_LUTRAM
    ap_uint<32> filter_cfg_b[filter_cfg_depth];
#pragma HLS resource variable = filter_cfg_b core = RAM_1P_LUTRAM

    // filter cfg
    for (int i = 0; i < filter_cfg_depth; i++) {
        filter_cfg_a[i] = config[6 + i / 16].range(32 * ((i % 16) + 1) - 1, 32 * (i % 16));
    }

    for (int i = 0; i < filter_cfg_depth; i++) {
        filter_cfg_b[i] = config[10 + i / 16].range(32 * ((i % 16) + 1) - 1, 32 * (i % 16));
    }

    if (!build_probe_flag) {
        for (int i = 0; i < filter_cfg_depth; i++) {
            filter_cfg_strm.write(filter_cfg_a[i]);
        }
    } else {
        for (int i = 0; i < filter_cfg_depth; i++) {
            filter_cfg_strm.write(filter_cfg_b[i]);
        }
    }
}
*/

// load kernel config for gqeJoin(with bloom-filter) kernel
static void load_config(bool build_probe_flag,
                        ap_uint<512> din_krn_cfg[14],
                        ap_uint<512> din_meta[24],
                        int64_t& nrow,
                        int& secID,
                        hls::stream<ap_uint<6> >& join_cfg_strm,
                        hls::stream<ap_uint<36> >& bf_cfg_strm,
                        hls::stream<ap_uint<32> >& filter_cfg_strm,
                        ap_uint<3>& din_col_en,
                        ap_uint<2>& rowid_flags,
                        hls::stream<ap_uint<8> >& write_out_cfg_strm) {
#ifdef USER_DEBUG
    std::cout << "-------- load kernel config --------" << std::endl;
#endif
    const int filter_cfg_depth = 53;

    // read in the number of rows
    nrow = din_meta[0].range(71, 8);
    // read in the secID and pass to scan_cols module
    secID = din_meta[0].range(167, 136);
#ifdef USER_DEBUG
    std::cout << "nrow = " << nrow << std::endl;
#endif
    // read in krn_cfg from ddr
    ap_uint<512> config[14];
    for (int i = 0; i < 14; ++i) {
        config[i] = din_krn_cfg[i];
    }

    ap_uint<36> bf_cfg;
    // bloom-filter on
    bf_cfg[0] = config[0][1];
#ifdef USER_DEBUG
    std::cout << "in load_cfg: bloomfilter_on: " << bf_cfg[0] << std::endl;
#endif
    // bloom-filter size
    bf_cfg.range(35, 1) = config[2].range(54, 20);
    bf_cfg_strm.write(bf_cfg);

    // define join_cfg to represent the join cfg for each module:
    // bit5: build_probe_flag; bit4: bf on or off; bit3-2: join type; bit1: dual key on or off; bit0: join on or bypass
    ap_uint<6> join_cfg;

    // join or bypass
    join_cfg[0] = config[0][0];
    // single key or dual key?
    if (config[0][1]) {
        join_cfg[1] = config[2][2];
    } else {
        join_cfg[1] = config[0][2];
    }
    // join type: normal, semi- or anti-
    join_cfg.range(3, 2) = config[0].range(4, 3);
#ifdef USER_DEBUG
    std::cout << "in load_cfg: join_on: " << join_cfg[0] << std::endl;
#endif
    // build_probe_flag
    join_cfg[5] = build_probe_flag;
    join_cfg_strm.write(join_cfg);

    // cfg flags used in build phase
    if (!build_probe_flag) {
        // read in input col enable flag for table A
        din_col_en = config[0].range(8, 6);
    } else { // cfg flags used in probe phase
        // read in input col enable flag for table B
        // read from bloom-filter config: config[2]
        if (config[0][1]) {
            din_col_en = config[2].range(11, 9);
            // read from join config: config[0]
        } else {
            din_col_en = config[0].range(11, 9);
        }
    }

    // gen_rowid and valid en flag
    // 16/18: gen_rowid_en; 17/19: valid_en
    if (!build_probe_flag) {
        rowid_flags = config[0].range(17, 16);
    } else {
        rowid_flags = config[0].range(19, 18);
    }

    // write out col cfg
    ap_uint<8> write_out_en;
    // append mode
    write_out_en[7] = config[0][5];
    // write out en
    // read from bloom-filter config: config[2]
    if (config[0][1]) {
        write_out_en.range(3, 0) = config[2].range(15, 12);
        // read from join config: config[0]
    } else {
        write_out_en.range(3, 0) = config[0].range(15, 12);
    }
    write_out_cfg_strm.write(write_out_en);
#ifdef USER_DEBUG
    std::cout << "write_out_en: " << (int)write_out_en << std::endl;
#endif

    ap_uint<32> filter_cfg_a[filter_cfg_depth];
#pragma HLS resource variable = filter_cfg_a core = RAM_1P_LUTRAM
    ap_uint<32> filter_cfg_b[filter_cfg_depth];
#pragma HLS resource variable = filter_cfg_b core = RAM_1P_LUTRAM

    // filter cfg
    for (int i = 0; i < filter_cfg_depth; i++) {
        filter_cfg_a[i] = config[6 + i / 16].range(32 * ((i % 16) + 1) - 1, 32 * (i % 16));
    }

    for (int i = 0; i < filter_cfg_depth; i++) {
        filter_cfg_b[i] = config[10 + i / 16].range(32 * ((i % 16) + 1) - 1, 32 * (i % 16));
    }

    if (!build_probe_flag) {
        for (int i = 0; i < filter_cfg_depth; i++) {
            filter_cfg_strm.write(filter_cfg_a[i]);
        }
    } else {
        for (int i = 0; i < filter_cfg_depth; i++) {
            filter_cfg_strm.write(filter_cfg_b[i]);
        }
    }
}

// load kernel config for gqePart kernel
static void load_config(bool tab_index,
                        const int log_part,
                        const int bucket_depth,
                        ap_uint<512> din_krn_cfg[14],
                        ap_uint<512> din_meta[24],
                        int64_t& nrow,
                        int32_t& secID,
                        hls::stream<ap_uint<16> >& part_cfg_strm,
                        ap_uint<3>& din_col_en,
                        ap_uint<2>& rowID_flags,
                        hls::stream<ap_uint<32> >& filter_cfg_strm,
                        hls::stream<int>& bit_num_strm_copy,
                        hls::stream<ap_uint<8> >& write_out_cfg_strm) {
#ifdef USER_DEBUG
    std::cout << "-------- load kernel config --------" << std::endl;
#endif
    const int filter_cfg_depth = 53;

    // read in the number of rows
    nrow = din_meta[0].range(71, 8);
    // read in the secID and pass to scan_cols module
    secID = din_meta[0].range(167, 136);
#ifdef USER_DEBUG
    std::cout << "nrow = " << nrow << std::endl;
#endif
    // read in krn_cfg from ddr
    ap_uint<512> config[14];
    for (int i = 0; i < 14; ++i) {
        config[i] = din_krn_cfg[i];
    }

    // define part_cfg to represent the part cfg for each module:
    // part_cfg 16 - bits:
    // bit0: dual key or single key; bit4-1: log_part; bit 15-5: kernel_depth
    ap_uint<16> part_cfg;
    // dual key or single key
    part_cfg[0] = config[1][2];
    part_cfg.range(4, 1) = log_part;
    part_cfg.range(15, 5) = bucket_depth;
    part_cfg_strm.write(part_cfg);

    bit_num_strm_copy.write(log_part);

    // read in input col enable flag for table A
    // build table
    if (!tab_index) {
        din_col_en = config[1].range(8, 6);
    } else { // join table
        din_col_en = config[1].range(11, 9);
    }

    // gen_rowid and valid en flag
    // 19: gen_rowid_en; 20: valid_en
    if (!tab_index) {
        rowID_flags = config[1].range(20, 19);
    } else {
        rowID_flags = config[1].range(22, 21);
    }

    // filter cfg
    ap_uint<32> filter_cfg_a[filter_cfg_depth];
#pragma HLS resource variable = filter_cfg_a core = RAM_1P_LUTRAM

    for (int i = 0; i < filter_cfg_depth; i++) {
        filter_cfg_a[i] = config[4 * tab_index + 6 + i / 16].range(32 * ((i % 16) + 1) - 1, 32 * (i % 16));
        filter_cfg_strm.write(filter_cfg_a[i]);
    }

    // write out col cfg
    ap_uint<8> write_out_en;
    if (!tab_index) {
        // build
        write_out_en.range(2, 0) = config[1].range(14, 12);
    } else {
        // probe
        write_out_en.range(2, 0) = config[1].range(18, 16);
    }
    write_out_cfg_strm.write(write_out_en);
#ifdef USER_DEBUG
    std::cout << "write_out_en: " << (int)write_out_en << std::endl;
#endif
}

} // namespace gqe
} // namespace database
} // namespace xf

#endif
