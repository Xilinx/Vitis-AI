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

#ifndef HLS_TEST
// OpenCL C API utils
#include "xclhost.hpp"
#endif

#include "xf_utils_sw/logger.hpp"

#include "x_utils.hpp"

// GQE L2
#include "xf_database/meta_table.hpp"
#include "xf_database/kernel_command.hpp"
#include "xf_database/enums.hpp"
// HLS
#include "ap_int.h"

#include "table_dt.hpp"

#include <sys/time.h>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <cstdio>
#include <unordered_map>

#define VEC_SCAN 8

const int PU_NM = 8;

#ifndef __SYNTHESIS__
extern "C" void gqeJoin(size_t _build_probe_flag,

                        // input data columns
                        ap_uint<8 * TPCH_INT_SZ * VEC_SCAN>* din_col0,
                        ap_uint<8 * TPCH_INT_SZ * VEC_SCAN>* din_col1,
                        ap_uint<8 * TPCH_INT_SZ * VEC_SCAN>* din_col2,

                        // validation buffer
                        ap_uint<64>* din_val,

                        // kernel config
                        ap_uint<512> din_krn_cfg[14],

                        // meta input buffer
                        ap_uint<512> din_meta[24],
                        // meta output buffer
                        ap_uint<512> dout_meta[24],

                        //  output data columns
                        ap_uint<8 * TPCH_INT_SZ * VEC_LEN>* dout_col0,
                        ap_uint<8 * TPCH_INT_SZ * VEC_LEN>* dout_col1,
                        ap_uint<8 * TPCH_INT_SZ * VEC_LEN>* dout_col2,
                        ap_uint<8 * TPCH_INT_SZ * VEC_LEN>* dout_col3,

                        // hbm buffers used to save build table key/payload
                        ap_uint<256>* htb_buf0,
                        ap_uint<256>* htb_buf1,
                        ap_uint<256>* htb_buf2,
                        ap_uint<256>* htb_buf3,
                        ap_uint<256>* htb_buf4,
                        ap_uint<256>* htb_buf5,
                        ap_uint<256>* htb_buf6,
                        ap_uint<256>* htb_buf7,

                        // overflow buffers
                        ap_uint<256>* stb_buf0,
                        ap_uint<256>* stb_buf1,
                        ap_uint<256>* stb_buf2,
                        ap_uint<256>* stb_buf3,
                        ap_uint<256>* stb_buf4,
                        ap_uint<256>* stb_buf5,
                        ap_uint<256>* stb_buf6,
                        ap_uint<256>* stb_buf7);

#endif

inline int tvdiff(const timeval& tv0, const timeval& tv1) {
    return (tv1.tv_sec - tv0.tv_sec) * 1000000 + (tv1.tv_usec - tv0.tv_usec);
}

inline int tvdiff(const timeval& tv0, const timeval& tv1, const char* name) {
    int exec_us = tvdiff(tv0, tv1);
    printf("%s: %d.%03d msec\n", name, (exec_us / 1000), (exec_us % 1000));
    return exec_us;
}
template <typename T>
int generate_data(T* data, int64_t range, size_t n) {
    if (!data) {
        return -1;
    }

    for (size_t i = 0; i < n; i++) {
        data[i] = (T)(rand() % range + 1);
    }

    return 0;
}

// load one col data into 1 buffer
template <typename T>
int load_dat(void* data, const std::string& name, const std::string& dir, const int sf, const size_t n) {
    if (!data) {
        return -1;
    }
    std::string fn = dir + "/dat" + std::to_string(sf) + "/" + name + ".dat";
    FILE* f = fopen(fn.c_str(), "rb");
    if (!f) {
        std::cerr << "ERROR: " << fn << " cannot be opened for binary read." << std::endl;
    }
    size_t cnt = fread(data, sizeof(T), n, f);
    fclose(f);
    if (cnt != n) {
        std::cerr << "ERROR: " << cnt << " entries read from " << fn << ", " << n << " entries required." << std::endl;
        return -1;
    }
    return 0;
}

TPCH_INT get_golden_sum(TPCH_INT o_row,
                        TPCH_INT* col_o_orderkey,
                        TPCH_INT* col_o_rowID,
                        TPCH_INT l_row,
                        TPCH_INT* col_l_orderkey,
                        TPCH_INT* col_l_rowID) {
    TPCH_INT sum = 0;
    TPCH_INT cnt = 0;

    // std::unordered_multimap<uint32_t, uint32_t> ht1;
    std::unordered_multimap<TPCH_INT, TPCH_INT> ht1;

    std::cout << "Table A: " << std::endl;
    {
        for (int64_t i = 0; i < o_row; ++i) {
            TPCH_INT key = col_o_orderkey[i];
            col_o_rowID[i] = i;
            if (i < 64) {
                std::cout << "key: " << key << ", o_rowid: " << i + 1 << std::endl;
            }
            // insert into hash table
            // if (date >= 700 && date < 800) {
            ht1.insert(std::make_pair(key, i + 1));
            //}
        }
    }
    std::cout << "-----------------------------------------" << std::endl;
    std::cout << "Table B: " << std::endl;
    for (int64_t i = 0; i < l_row; ++i) {
        TPCH_INT key = col_l_orderkey[i];
        col_l_rowID[i] = i + 1;
        if (i < 64) {
            std::cout << "key: " << key << ", l_rowid: " << i + 1 << std::endl;
        }
    }
    std::cout << "-----------------------------------------" << std::endl;

    std::cout << "Table C golden: " << std::endl;
    // read t once
    for (int64_t i = 0; i < l_row; ++i) {
        TPCH_INT key = col_l_orderkey[i];
        TPCH_INT l_rowid = col_l_rowID[i];
        // check hash table
        auto its = ht1.equal_range(key);
        for (auto it = its.first; it != its.second; ++it) {
            if (cnt < 64) {
                std::cout << "key: " << key << ", o_rowid: " << it->second << ", l_rowid : " << l_rowid << std::endl;
            }
            int64_t sum_i = key + it->second + l_rowid;

            sum += sum_i;
            ++cnt;
        }
    }

    std::cout << "INFO: CPU ref matched " << cnt << " rows, sum = " << sum << std::endl;
    return sum;
}

/* filter (19940101<= col:1 < 19950101) */
static void gen_q5simple_orders_fcfg(uint32_t cfg[]) {
    using namespace xf::database;
    int n = 0;

    // cond_1
    cfg[n++] = (uint32_t)0UL;
    cfg[n++] = (uint32_t)0UL;
    cfg[n++] = (uint32_t)0UL;
    cfg[n++] = (uint32_t)0UL;
    cfg[n++] = 0UL | (FOP_DC << FilterOpWidth) | (FOP_DC);

    // 700<= cond_1 <800
    uint64_t dat1 = (uint64_t)700;
    uint64_t dat2 = (uint64_t)800;
    cfg[n++] = dat1 & 0xffffffff;
    cfg[n++] = dat1 >> 32;
    cfg[n++] = dat2 & 0xffffffff;
    cfg[n++] = dat2 >> 32;
    cfg[n++] = 0UL | (FOP_GEU << FilterOpWidth) | (FOP_LTU);
    // cond_3
    cfg[n++] = (uint32_t)0L;
    cfg[n++] = (uint32_t)0L;
    cfg[n++] = (uint32_t)0L;
    cfg[n++] = (uint32_t)0L;
    cfg[n++] = 0UL | (FOP_DC << FilterOpWidth) | (FOP_DC);
    // cond_4
    cfg[n++] = (uint32_t)0L;
    cfg[n++] = (uint32_t)0L;
    cfg[n++] = (uint32_t)0L;
    cfg[n++] = (uint32_t)0L;
    cfg[n++] = 0UL | (FOP_DC << FilterOpWidth) | (FOP_DC);

    uint32_t r = 0;
    int sh = 0;
    // cond_1 -- cond_2
    r |= ((uint32_t)(FOP_DC << sh));
    sh += FilterOpWidth;
    // cond_1 -- cond_3
    r |= ((uint32_t)(FOP_DC << sh));
    sh += FilterOpWidth;
    // cond_1 -- cond_4
    r |= ((uint32_t)(FOP_DC << sh));
    sh += FilterOpWidth;

    // cond_2 -- cond_3
    r |= ((uint32_t)(FOP_DC << sh));
    sh += FilterOpWidth;
    // cond_2 -- cond_4
    r |= ((uint32_t)(FOP_DC << sh));
    sh += FilterOpWidth;

    // cond_3 -- cond_4
    r |= ((uint32_t)(FOP_DC << sh));
    sh += FilterOpWidth;

    cfg[n++] = r;

    // 4 true and 6 true
    cfg[n++] = (uint32_t)0UL;
    cfg[n++] = (uint32_t)0UL;
    cfg[n++] = (uint32_t)0UL;
    cfg[n++] = (uint32_t)0UL;
    cfg[n++] = (uint32_t)0UL;
    cfg[n++] = (uint32_t)0UL;
    cfg[n++] = (uint32_t)0UL;
    cfg[n++] = (uint32_t)0UL;
    cfg[n++] = (uint32_t)0UL;
    cfg[n++] = (uint32_t)0UL;
    cfg[n++] = (uint32_t)0UL;
    cfg[n++] = (uint32_t)0UL;
    cfg[n++] = (uint32_t)0UL;
    cfg[n++] = (uint32_t)0UL;
    cfg[n++] = (uint32_t)0UL;
    cfg[n++] = (uint32_t)0UL;
    cfg[n++] = (uint32_t)0UL;
    cfg[n++] = (uint32_t)0UL;
    cfg[n++] = (uint32_t)0UL;
    cfg[n++] = (uint32_t)0UL;
    cfg[n++] = (uint32_t)0UL;
    cfg[n++] = (uint32_t)0UL;
    cfg[n++] = (uint32_t)0UL;
    cfg[n++] = (uint32_t)0UL;
    cfg[n++] = (uint32_t)0UL;
    cfg[n++] = (uint32_t)0UL;
    cfg[n++] = (uint32_t)0UL;
    cfg[n++] = (uint32_t)0UL;
    cfg[n++] = (uint32_t)0UL;
    cfg[n++] = (uint32_t)0UL;
    cfg[n++] = (uint32_t)0UL;
    cfg[n++] = (uint32_t)(1UL << 31);
}

void gen_pass_fcfg(uint32_t cfg[]) {
    using namespace xf::database;
    int n = 0;

    // cond_1
    cfg[n++] = (uint32_t)0UL;
    cfg[n++] = (uint32_t)0UL;
    cfg[n++] = (uint32_t)0UL;
    cfg[n++] = (uint32_t)0UL;
    cfg[n++] = 0UL | (FOP_DC << FilterOpWidth) | (FOP_DC);
    // cond_2
    cfg[n++] = (uint32_t)0UL;
    cfg[n++] = (uint32_t)0UL;
    cfg[n++] = (uint32_t)0UL;
    cfg[n++] = (uint32_t)0UL;
    cfg[n++] = 0UL | (FOP_DC << FilterOpWidth) | (FOP_DC);
    // cond_3
    cfg[n++] = (uint32_t)0UL;
    cfg[n++] = (uint32_t)0UL;
    cfg[n++] = (uint32_t)0UL;
    cfg[n++] = (uint32_t)0UL;
    cfg[n++] = 0UL | (FOP_DC << FilterOpWidth) | (FOP_DC);
    // cond_4
    cfg[n++] = (uint32_t)0UL;
    cfg[n++] = (uint32_t)0UL;
    cfg[n++] = (uint32_t)0UL;
    cfg[n++] = (uint32_t)0UL;
    cfg[n++] = 0UL | (FOP_DC << FilterOpWidth) | (FOP_DC);

    uint32_t r = 0;
    int sh = 0;
    // cond_1 -- cond_2
    r |= ((uint32_t)(FOP_DC << sh));
    sh += FilterOpWidth;
    // cond_1 -- cond_3
    r |= ((uint32_t)(FOP_DC << sh));
    sh += FilterOpWidth;
    // cond_1 -- cond_4
    r |= ((uint32_t)(FOP_DC << sh));
    sh += FilterOpWidth;

    // cond_2 -- cond_3
    r |= ((uint32_t)(FOP_DC << sh));
    sh += FilterOpWidth;
    // cond_2 -- cond_4
    r |= ((uint32_t)(FOP_DC << sh));
    sh += FilterOpWidth;

    // cond_3 -- cond_4
    r |= ((uint32_t)(FOP_DC << sh));
    sh += FilterOpWidth;

    cfg[n++] = r;

    // 4 true and 6 true
    cfg[n++] = (uint32_t)0UL;
    cfg[n++] = (uint32_t)0UL;
    cfg[n++] = (uint32_t)0UL;
    cfg[n++] = (uint32_t)0UL;
    cfg[n++] = (uint32_t)0UL;
    cfg[n++] = (uint32_t)0UL;
    cfg[n++] = (uint32_t)0UL;
    cfg[n++] = (uint32_t)0UL;
    cfg[n++] = (uint32_t)0UL;
    cfg[n++] = (uint32_t)0UL;
    cfg[n++] = (uint32_t)0UL;
    cfg[n++] = (uint32_t)0UL;
    cfg[n++] = (uint32_t)0UL;
    cfg[n++] = (uint32_t)0UL;
    cfg[n++] = (uint32_t)0UL;
    cfg[n++] = (uint32_t)0UL;
    cfg[n++] = (uint32_t)0UL;
    cfg[n++] = (uint32_t)0UL;
    cfg[n++] = (uint32_t)0UL;
    cfg[n++] = (uint32_t)0UL;
    cfg[n++] = (uint32_t)0UL;
    cfg[n++] = (uint32_t)0UL;
    cfg[n++] = (uint32_t)0UL;
    cfg[n++] = (uint32_t)0UL;
    cfg[n++] = (uint32_t)0UL;
    cfg[n++] = (uint32_t)0UL;
    cfg[n++] = (uint32_t)0UL;
    cfg[n++] = (uint32_t)0UL;
    cfg[n++] = (uint32_t)0UL;
    cfg[n++] = (uint32_t)0UL;
    cfg[n++] = (uint32_t)0UL;
    cfg[n++] = (uint32_t)(1UL << 31);
}

int main(int argc, const char* argv[]) {
    std::cout << "\n--------- TPC-H Query 5 Simplified (1G) ---------\n";

    // cmd arg parser.
    x_utils::ArgParser parser(argc, argv);

#ifndef HLS_TEST
    std::string xclbin_path; // eg. q5kernel_VCU1525_hw.xclbin
    if (!parser.getCmdOption("-xclbin", xclbin_path)) {
        std::cout << "ERROR: xclbin path is not set!\n";
        return 1;
    }
#endif

    std::string scale;
    int sim_scale = 100;
    if (parser.getCmdOption("-scale", scale)) {
        try {
            sim_scale = std::stoi(scale);
        } catch (...) {
            sim_scale = 10000;
        }
    }
    int o_nrow = O_MAX_ROW / sim_scale;
    int l_nrow = L_MAX_ROW / sim_scale;

    std::cout << "Orders has " << o_nrow << " rows" << std::endl;
    std::cout << "Lineitem has " << l_nrow << " rows" << std::endl;

    x_utils::MM mm;

    // Number of vec in input buf. Add some extra and align.
    size_t table_l_depth = (l_nrow + VEC_LEN - 1) / VEC_LEN;
    size_t table_o_depth = (o_nrow + VEC_LEN - 1) / VEC_LEN;

    size_t table_l_size = table_l_depth * VEC_LEN * TPCH_INT_SZ;
    size_t table_o_size = table_o_depth * VEC_LEN * TPCH_INT_SZ;

    // data load from disk. will re-use in each call, but assumed to be different.
    TPCH_INT* table_o_user_0 = mm.aligned_alloc<TPCH_INT>(table_o_depth * VEC_LEN);
    TPCH_INT* table_o_user_1 = mm.aligned_alloc<TPCH_INT>(table_o_depth * VEC_LEN);
    TPCH_INT* table_o_user_2 = mm.aligned_alloc<TPCH_INT>(table_o_depth * VEC_LEN);

    int error = 0;
    error += generate_data<TPCH_INT>((TPCH_INT*)(table_o_user_0), 100000, o_nrow);
    // error += generate_data<TPCH_INT>((TPCH_INT*)(table_o_user_1), 1000, o_nrow);
    if (error) return error;
    std::cout << "Orders table data has been generated" << std::endl;

    TPCH_INT* table_l_user_0 = mm.aligned_alloc<TPCH_INT>(table_l_depth * VEC_LEN);
    TPCH_INT* table_l_user_1 = mm.aligned_alloc<TPCH_INT>(table_l_depth * VEC_LEN);
    TPCH_INT* table_l_user_2 = mm.aligned_alloc<TPCH_INT>(table_l_depth * VEC_LEN);

    error += generate_data<TPCH_INT>((TPCH_INT*)(table_l_user_0), 10000, l_nrow);
    // error += generate_data<TPCH_INT>((TPCH_INT*)(table_l_user_1), 10000, l_nrow);
    // error += generate_data<TPCH_INT>((TPCH_INT*)(table_l_user_2), 10000, l_nrow);
    if (error) return error;
    std::cout << "LineItem table data has been generated" << std::endl;

    TPCH_INT golden = get_golden_sum(o_nrow, table_o_user_0, table_o_user_1, l_nrow, table_l_user_0, table_l_user_1);

    ap_uint<64>* din_val = mm.aligned_alloc<ap_uint<64> >((l_nrow + 63) / 64);
    for (int i = 0; i < (l_nrow + 63) / 64; i++) {
        din_val[i] = 0xffffffffffffffff;
    }

    // result buff
    size_t result_nrow = (1 << 26);
    size_t table_result_depth = result_nrow / VEC_LEN; // 8 columns in one buffer
    size_t table_result_size = table_result_depth * VEC_LEN * TPCH_INT_SZ;
    ap_uint<512>* table_out_0 = mm.aligned_alloc<ap_uint<512> >(table_result_depth);
    ap_uint<512>* table_out_1 = mm.aligned_alloc<ap_uint<512> >(table_result_depth);
    ap_uint<512>* table_out_2 = mm.aligned_alloc<ap_uint<512> >(table_result_depth);
    ap_uint<512>* table_out_3 = mm.aligned_alloc<ap_uint<512> >(table_result_depth);
    memset(table_out_0, 0, table_result_size);
    memset(table_out_1, 0, table_result_size);
    memset(table_out_2, 0, table_result_size);
    memset(table_out_3, 0, table_result_size);

    // using jcmdclass = xf::database::gqe::JoinCommand;
    // jcmdclass jcmd = jcmdclass();

    // jcmd.setJoinType(xf::database::INNER_JOIN);
    // jcmd.Scan(0, {0, 1});
    // jcmd.Scan(1, {0, 1});
    // jcmd.setWriteCol({0, 1});

    //// jcmd.setEvaluation(0, "strm1*(-strm2+c2)", {0, 100});
    // jcmd.setFilter(0, "19940101<=b && b<19950101");

    //// jcmd.setShuffle0(0, {0, 1});
    //// jcmd.setShuffle0(1, {0, 1, 2});

    //// jcmd.setShuffle1(0, {0, 1});    // setJoinKeys
    //// jcmd.setShuffle1(1, {0, 1, 2}); // setJoinKeys

    //// jcmd.setShuffle2({0, 1});
    //// jcmd.setShuffle3({8});
    //// jcmd.setShuffle4({0});
    //// jcmd.setShuffle2({0, 1, 6, 12});
    //// jcmd.setShuffle3({0, 1, 2, 3});
    //// jcmd.setShuffle4({0, 1, 2, 3});
    // ap_uint<512>* krn_cfg = jcmd.getConfigBits();

    ap_uint<512>* krn_cfg = mm.aligned_alloc<ap_uint<512> >(14);
    memset(krn_cfg, 0, sizeof(ap_uint<512>) * 14);
    // join on
    krn_cfg[0].range(0, 0) = 1;
    // join type: normal hash join
    krn_cfg[0].range(4, 3) = 0;
    // tab A gen_row_id
    krn_cfg[0].range(16, 16) = 1;
    // tab A din_val_en
    krn_cfg[0].range(17, 17) = 0;
    // tab B gen_row_id
    krn_cfg[0].range(18, 18) = 1;
    // tab B din_val_en
    krn_cfg[0].range(19, 19) = 1;
    // tab A col enable
    krn_cfg[0].range(8, 6) = 1;
    // tab B col enable
    krn_cfg[0].range(11, 9) = 1;
    // append mode off
    krn_cfg[0].range(5, 5) = 0;
    // write out enable
    krn_cfg[0].range(15, 12) = 7;

    // filter for build table
    uint32_t cfg[53];
    gen_pass_fcfg(cfg);
    memcpy(&krn_cfg[6], cfg, sizeof(uint32_t) * 53);

    // 512b word * 4
    // filter b
    gen_pass_fcfg(cfg);
    memcpy(&krn_cfg[10], cfg, sizeof(uint32_t) * 53);

    // filter for join table
    // std::cout << "cfg---------------cfg" << std::endl;
    // for (int i = 0; i < 14; i++) {
    //    for (int n = 0; n < 16; n++) {
    //        std::cout << "i, n: " << i << ", " << n << ": " << krn_cfg[i].range(32 * n + 31, 32 * n) << std::endl;
    //    }
    //}
    // std::cout << "cfg--------end------cfg" << std::endl;

    //--------------- metabuffer setup -----------------
    // using col0 and col1 buffer during build
    // setup build used meta input
    xf::database::gqe::MetaTable meta_build_in;
    meta_build_in.setSecID(0);
    meta_build_in.setColNum(1);
    meta_build_in.setCol(0, 0, o_nrow);

    // setup probe used meta input
    xf::database::gqe::MetaTable meta_probe_in;
    meta_probe_in.setSecID(0);
    meta_probe_in.setColNum(1);
    meta_probe_in.setCol(0, 0, l_nrow);

    // ouput col0,1,2,3 buffers data, with order: 0 1 2 3. (When aggr is off)
    // when aggr is on, actually only using col0 is enough.
    // below example only illustrates the output buffers can be shuffled.
    // setup probe used meta output
    xf::database::gqe::MetaTable meta_probe_out;
    meta_probe_out.setColNum(3);
    meta_probe_out.setCol(0, 0, result_nrow);
    meta_probe_out.setCol(1, 1, result_nrow);
    meta_probe_out.setCol(2, 2, result_nrow);
    // meta_probe_out.setCol(3, 3, result_nrow);

    int htb_buf_depth = HT_BUFF_DEPTH;
    int stb_buf_depth = HT_BUFF_DEPTH;
    int htb_buf_size = sizeof(ap_uint<256>) * htb_buf_depth;
    int stb_buf_size = sizeof(ap_uint<256>) * stb_buf_depth;

    size_t flag_build = 0;
    size_t flag_probe = 1;
#ifdef HLS_TEST

    ap_uint<256>* htb_buf0 = mm.aligned_alloc<ap_uint<256> >(htb_buf_depth);
    ap_uint<256>* htb_buf1 = mm.aligned_alloc<ap_uint<256> >(htb_buf_depth);
    ap_uint<256>* htb_buf2 = mm.aligned_alloc<ap_uint<256> >(htb_buf_depth);
    ap_uint<256>* htb_buf3 = mm.aligned_alloc<ap_uint<256> >(htb_buf_depth);
    ap_uint<256>* htb_buf4 = mm.aligned_alloc<ap_uint<256> >(htb_buf_depth);
    ap_uint<256>* htb_buf5 = mm.aligned_alloc<ap_uint<256> >(htb_buf_depth);
    ap_uint<256>* htb_buf6 = mm.aligned_alloc<ap_uint<256> >(htb_buf_depth);
    ap_uint<256>* htb_buf7 = mm.aligned_alloc<ap_uint<256> >(htb_buf_depth);

    ap_uint<256>* stb_buf0 = mm.aligned_alloc<ap_uint<256> >(stb_buf_depth);
    ap_uint<256>* stb_buf1 = mm.aligned_alloc<ap_uint<256> >(stb_buf_depth);
    ap_uint<256>* stb_buf2 = mm.aligned_alloc<ap_uint<256> >(stb_buf_depth);
    ap_uint<256>* stb_buf3 = mm.aligned_alloc<ap_uint<256> >(stb_buf_depth);
    ap_uint<256>* stb_buf4 = mm.aligned_alloc<ap_uint<256> >(stb_buf_depth);
    ap_uint<256>* stb_buf5 = mm.aligned_alloc<ap_uint<256> >(stb_buf_depth);
    ap_uint<256>* stb_buf6 = mm.aligned_alloc<ap_uint<256> >(stb_buf_depth);
    ap_uint<256>* stb_buf7 = mm.aligned_alloc<ap_uint<256> >(stb_buf_depth);

    // build
    gqeJoin(flag_build, (ap_uint<512>*)table_o_user_0, (ap_uint<512>*)table_o_user_1, (ap_uint<512>*)table_o_user_2,
            din_val, krn_cfg, meta_build_in.meta(), meta_probe_out.meta(), table_out_0, table_out_1, table_out_2,
            table_out_3, htb_buf0, htb_buf1, htb_buf2, htb_buf3, htb_buf4, htb_buf5, htb_buf6, htb_buf7, stb_buf0,
            stb_buf1, stb_buf2, stb_buf3, stb_buf4, stb_buf5, stb_buf6, stb_buf7);

    std::cout << std::endl << "probe starts................." << std::endl;
    // probe
    gqeJoin(flag_probe, (ap_uint<512>*)table_l_user_0, (ap_uint<512>*)table_l_user_1, (ap_uint<512>*)table_l_user_2,
            din_val, krn_cfg, meta_probe_in.meta(), meta_probe_out.meta(), table_out_0, table_out_1, table_out_2,
            table_out_3, htb_buf0, htb_buf1, htb_buf2, htb_buf3, htb_buf4, htb_buf5, htb_buf6, htb_buf7, stb_buf0,
            stb_buf1, stb_buf2, stb_buf3, stb_buf4, stb_buf5, stb_buf6, stb_buf7);
    std::cout << "probe ends................." << std::endl;

#else

    using namespace xf::common::utils_sw;
    Logger logger(std::cout, std::cerr);

    // Get CL devices.
    cl_int err;
    cl_context ctx;
    cl_device_id dev_id;
    cl_command_queue cmq;
    cl_program prg;

    err = xclhost::init_hardware(&ctx, &dev_id, &cmq,
                                 CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, MSTR(XDEVICE));
    if (err != CL_SUCCESS) {
        fprintf(stderr, "ERROR: fail to init OpenCL with " MSTR(XDEVICE) "\n");
        return err;
    }

    err = xclhost::load_binary(&prg, ctx, dev_id, xclbin_path.c_str());
    if (err != CL_SUCCESS) {
        fprintf(stderr, "ERROR: fail to program PL\n");
        return err;
    }

    // build kernel
    cl_kernel bkernel;
    bkernel = clCreateKernel(prg, "gqeJoin", &err);
    // will not exit with failure by default
    logger.logCreateKernel(err);
    // probe kernel, pipeline used handle
    cl_kernel jkernel;
    jkernel = clCreateKernel(prg, "gqeJoin", &err);
    logger.logCreateKernel(err);

    std::cout << "Kernels has been created\n";

    cl_mem_ext_ptr_t mext_table_o[3], mext_table_l[3], mext_cfg5s;
    cl_mem_ext_ptr_t mext_valid_in_col;
    cl_mem_ext_ptr_t mext_table_out[4];
    cl_mem_ext_ptr_t memExt[PU_NM * 2];

    mext_table_o[0] = {1, table_o_user_0, bkernel};
    mext_table_o[1] = {2, table_o_user_1, bkernel};
    mext_table_o[2] = {3, table_o_user_2, bkernel};

    mext_valid_in_col = {4, din_val, bkernel};

    mext_cfg5s = {5, krn_cfg, bkernel};

    mext_table_l[0] = {1, table_l_user_0, jkernel};
    mext_table_l[1] = {2, table_l_user_1, jkernel};
    mext_table_l[2] = {3, table_l_user_2, jkernel};

    mext_table_out[0] = {8, table_out_0, jkernel};
    mext_table_out[1] = {9, table_out_1, jkernel};
    mext_table_out[2] = {10, table_out_2, jkernel};
    mext_table_out[3] = {11, table_out_3, jkernel};
    for (int j = 0; j < 16; j++) {
        memExt[j] = {12 + j, NULL, bkernel};
    }

    cl_mem_ext_ptr_t mext_meta_build_in, mext_meta_probe_in, mext_meta_probe_out;
    mext_meta_build_in = {6, meta_build_in.meta(), bkernel};
    mext_meta_probe_in = {6, meta_probe_in.meta(), jkernel};
    mext_meta_probe_out = {7, meta_probe_out.meta(), jkernel};

    // Map buffers
    cl_mem buf_table_o[3];
    cl_mem buf_table_l[3];
    cl_mem buf_table_out[4];

    for (int c = 0; c < 3; c++) {
        buf_table_o[c] = clCreateBuffer(ctx, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                        table_o_size, &mext_table_o[c], &err);
        buf_table_l[c] = clCreateBuffer(ctx, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                        table_l_size, &mext_table_l[c], &err);
    }
    cl_mem buf_valid_in_col = clCreateBuffer(ctx, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                             ((l_nrow + 63) / 64) * sizeof(ap_uint<64>), &mext_valid_in_col, &err);

    for (int c = 0; c < 4; c++) {
        buf_table_out[c] = clCreateBuffer(ctx, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                          table_result_size, &mext_table_out[c], &err);
    }

    cl_mem buf_cfg5s = clCreateBuffer(ctx, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                      (sizeof(ap_uint<512>) * 14), &mext_cfg5s, &err);

    // htb stb buffers
    cl_mem buf_tmp[PU_NM * 2];
    for (int j = 0; j < PU_NM; j++) {
        buf_tmp[j] = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS | CL_MEM_EXT_PTR_XILINX,
                                    (size_t)(htb_buf_size), &memExt[j], &err);
    }
    for (int j = PU_NM; j < 2 * PU_NM; j++) {
        buf_tmp[j] = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS | CL_MEM_EXT_PTR_XILINX,
                                    (size_t)(stb_buf_size), &memExt[j], &err);
    }

    cl_mem buf_meta_build_in = clCreateBuffer(ctx, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                              (sizeof(ap_uint<512>) * 8), &mext_meta_build_in, &err);

    cl_mem buf_meta_probe_in = clCreateBuffer(ctx, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                              (sizeof(ap_uint<512>) * 8), &mext_meta_probe_in, &err);
    cl_mem buf_meta_probe_out = clCreateBuffer(ctx, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                               (sizeof(ap_uint<512>) * 8), &mext_meta_probe_out, &err);

    std::cout << "buffers have been mapped.\n";

    // helper buffer sets
    std::vector<cl_mem> non_loop_bufs;
    for (int c = 0; c < 3; c++) {
        non_loop_bufs.push_back(buf_table_o[c]);
        non_loop_bufs.push_back(buf_valid_in_col);
    }
    non_loop_bufs.push_back(buf_cfg5s);
    non_loop_bufs.push_back(buf_meta_build_in);
    non_loop_bufs.push_back(buf_meta_probe_out);

    std::vector<cl_mem> loop_in_bufs;
    for (int c = 0; c < 3; c++) {
        loop_in_bufs.push_back(buf_table_l[c]);
    }
    loop_in_bufs.push_back(buf_meta_probe_in);
    loop_in_bufs.push_back(buf_meta_probe_out);

    std::vector<cl_mem> loop_out_bufs;
    for (int c = 0; c < 4; c++) {
        loop_out_bufs.push_back(buf_table_out[c]);
    }
    loop_out_bufs.push_back(buf_meta_probe_out);

    // make resident
    clEnqueueMigrateMemObjects(cmq, loop_in_bufs.size(), loop_in_bufs.data(), CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED,
                               0, nullptr, nullptr);
    clEnqueueMigrateMemObjects(cmq, loop_out_bufs.size(), loop_out_bufs.data(), CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED,
                               0, nullptr, nullptr);

    // set args for bkernel
    int j = 0;
    clSetKernelArg(bkernel, j++, sizeof(size_t), &flag_build);
    clSetKernelArg(bkernel, j++, sizeof(cl_mem), &buf_table_o[0]);
    clSetKernelArg(bkernel, j++, sizeof(cl_mem), &buf_table_o[1]);
    clSetKernelArg(bkernel, j++, sizeof(cl_mem), &buf_table_o[2]);
    clSetKernelArg(bkernel, j++, sizeof(cl_mem), &buf_valid_in_col);
    clSetKernelArg(bkernel, j++, sizeof(cl_mem), &buf_cfg5s);
    clSetKernelArg(bkernel, j++, sizeof(cl_mem), &buf_meta_build_in);
    clSetKernelArg(bkernel, j++, sizeof(cl_mem), &buf_meta_probe_out);
    clSetKernelArg(bkernel, j++, sizeof(cl_mem), &buf_table_out[0]);
    clSetKernelArg(bkernel, j++, sizeof(cl_mem), &buf_table_out[1]);
    clSetKernelArg(bkernel, j++, sizeof(cl_mem), &buf_table_out[2]);
    clSetKernelArg(bkernel, j++, sizeof(cl_mem), &buf_table_out[3]);
    for (int k = 0; k < PU_NM * 2; k++) {
        clSetKernelArg(bkernel, j++, sizeof(cl_mem), &buf_tmp[k]);
    }

    // set args for jkernel
    j = 0;
    clSetKernelArg(jkernel, j++, sizeof(size_t), &flag_probe);
    clSetKernelArg(jkernel, j++, sizeof(cl_mem), &buf_table_l[0]);
    clSetKernelArg(jkernel, j++, sizeof(cl_mem), &buf_table_l[1]);
    clSetKernelArg(jkernel, j++, sizeof(cl_mem), &buf_table_l[2]);
    clSetKernelArg(jkernel, j++, sizeof(cl_mem), &buf_valid_in_col);
    clSetKernelArg(jkernel, j++, sizeof(cl_mem), &buf_cfg5s);
    clSetKernelArg(jkernel, j++, sizeof(cl_mem), &buf_meta_probe_in);
    clSetKernelArg(jkernel, j++, sizeof(cl_mem), &buf_meta_probe_out);
    clSetKernelArg(jkernel, j++, sizeof(cl_mem), &buf_table_out[0]);
    clSetKernelArg(jkernel, j++, sizeof(cl_mem), &buf_table_out[1]);
    clSetKernelArg(jkernel, j++, sizeof(cl_mem), &buf_table_out[2]);
    clSetKernelArg(jkernel, j++, sizeof(cl_mem), &buf_table_out[3]);
    for (int k = 0; k < PU_NM * 2; k++) {
        clSetKernelArg(jkernel, j++, sizeof(cl_mem), &buf_tmp[k]);
    }

    int k_id = 0;
    std::array<cl_event, 1> evt_tb_o;
    std::array<cl_event, 1> evt_bkrn;
    std::array<cl_event, 1> evt_tb_l;
    std::array<cl_event, 1> evt_pkrn;
    std::array<cl_event, 1> evt_tb_out;

    timeval t0, t1;
    gettimeofday(&t0, 0);
    // --- build ---
    // 2) migrate order table data from host buffer to device buffer
    clEnqueueMigrateMemObjects(cmq, non_loop_bufs.size(), non_loop_bufs.data(), 0, 0, nullptr, &evt_tb_o[0]);
    clFinish(cmq);
    gettimeofday(&t1, 0);
    tvdiff(t0, t1, "build h2d: ");

    gettimeofday(&t0, 0);
    // 3) launch build kernel
    clEnqueueTask(cmq, bkernel, 1, evt_tb_o.data(), &evt_bkrn[0]);
    clWaitForEvents(1, &evt_bkrn[0]);
    clFinish(cmq);
    gettimeofday(&t1, 0);
    tvdiff(t0, t1, "build time: ");

    // --- probe ---
    // 5) migrate L table data from host buffer to device buffer
    gettimeofday(&t0, 0);
    clEnqueueMigrateMemObjects(cmq, loop_in_bufs.size(), loop_in_bufs.data(), 0, 0, nullptr, &evt_tb_l[0]);
    clFinish(cmq);
    gettimeofday(&t1, 0);
    tvdiff(t0, t1, "probe h2d: ");

    // 6) launch probe kernel
    gettimeofday(&t0, 0);
    clEnqueueTask(cmq, jkernel, 1, evt_tb_l.data(), &evt_pkrn[0]);
    clFinish(cmq);
    gettimeofday(&t1, 0);
    tvdiff(t0, t1, "probe krn time: ");

    // 7) migrate result data from device buffer to host buffer
    gettimeofday(&t0, 0);
    clEnqueueMigrateMemObjects(cmq, loop_out_bufs.size(), loop_out_bufs.data(), CL_MIGRATE_MEM_OBJECT_HOST, 1,
                               evt_pkrn.data(), &evt_tb_out[0]);
    clFinish(cmq);
    gettimeofday(&t1, 0);
    tvdiff(t0, t1, "probe d2h time: ");

    // 8) copy output data from pinned host buffer to user host buffer
    clWaitForEvents(1, &evt_tb_out[0]);
#endif

    // =========== print result ===========
    // check the probe updated meta
    int out_nrow = meta_probe_out.getColLen();
    int p_nrow = out_nrow;
    std::cout << "Output buffer has " << out_nrow << " rows." << std::endl;

    int64_t key = 0;
    int64_t o_rowid = 0;
    int64_t l_rowid = 0;
    int64_t sum = 0;
    std::cout << "------------Checking result-------------" << std::endl;
    std::cout << "FPGA joined data: " << std::endl;
    // printing only first 64 data
    std::cout << "key, o_rowid, l_rowid" << std::endl;
    int64_t cnt = 0;
    for (int n = 0; n < p_nrow / 8; n++) {
        for (int i = 0; i < 8; i++) {
            l_rowid = table_out_0[n](63 + 64 * i, 64 * i);
            o_rowid = table_out_1[n](63 + 64 * i, 64 * i);
            key = table_out_2[n](63 + 64 * i, 64 * i);
            int64_t sum_i = key + l_rowid + o_rowid;
            sum += sum_i;
            if (cnt < 64) std::cout << key << ", " << o_rowid << ", " << l_rowid << std::endl;
            cnt++;
        }
    }
    for (int i = 0; i < p_nrow % 8; i++) {
        l_rowid = table_out_0[p_nrow / 8](63 + 64 * i, 64 * i);
        o_rowid = table_out_1[p_nrow / 8](63 + 64 * i, 64 * i);
        key = table_out_2[p_nrow / 8](63 + 64 * i, 64 * i);
        int64_t sum_i = key + l_rowid + o_rowid;
        sum += sum_i;
        if (cnt < 64) std::cout << key << ", " << o_rowid << ", " << l_rowid << std::endl;
        cnt++;
    }
    std::cout << "fpga, sum = " << sum << std::endl;
    std::cout << "cpu ref, golden= " << golden << std::endl;

    // test pass/fail messages
    int has_err_in_design = (golden == sum) ? 0 : 1;
    has_err_in_design ? logger.error(Logger::Message::TEST_FAIL) : logger.info(Logger::Message::TEST_PASS);

    return has_err_in_design;
}
