/*
 * Copyright 2018 Xilinx, Inc.
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

#include "hash_aggr_kernel.hpp"
#include "xf_database/enums.hpp"

#include <sys/time.h>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <unordered_map>

std::unordered_map<TPCH_INT, TPCH_INT> map0, map1, map2;

TPCH_INT group_sum(
    // input
    TPCH_INT* key,
    TPCH_INT* pay,
    int num,
    std::unordered_map<TPCH_INT, TPCH_INT>& ref_map) {
    for (int i = 0; i < num; ++i) {
        TPCH_INT k = key[i];
        TPCH_INT p = pay[i];
        auto it = ref_map.find(k);
        if (it != ref_map.end()) {
            TPCH_INT s = it->second + p; // calculate
#if 0
            if (i < 1024) {
                std::cout << std::hex << "update: idx=" << i << " key=" << k << " i_pld=" << p
                          << " old_pld=" << it->second << " new_pld=" << s << std::endl;
            }
#endif
            ref_map[k] = s; // update
        } else {
            ref_map.insert(std::make_pair(k, p));

            if (i < 1024) {
                std::cout << std::hex << "insert: idx=" << i << " key=" << k << " i_pld=" << p << std::endl;
            }
        }
    }

    return (TPCH_INT)ref_map.size();
}

TPCH_INT group_cnt(
    // input
    TPCH_INT* key,
    TPCH_INT* pay,
    int num,
    std::unordered_map<TPCH_INT, TPCH_INT>& ref_map) {
    for (int i = 0; i < num; ++i) {
        TPCH_INT k = key[i];
        TPCH_INT p = pay[i];
        auto it = ref_map.find(k);
        if (it != ref_map.end()) {
            TPCH_INT s = it->second + 1;
            ref_map[k] = s; // update
        } else {
            ref_map.insert(std::make_pair(k, 1));
        }
    }

    return (TPCH_INT)ref_map.size();
}

TPCH_INT group_mean(
    // input
    TPCH_INT* key,
    TPCH_INT* pay,
    int num,
    std::unordered_map<TPCH_INT, TPCH_INT>& map_sum,
    std::unordered_map<TPCH_INT, TPCH_INT>& map_cnt,
    std::unordered_map<TPCH_INT, TPCH_INT>& map_mean) {
    for (int i = 0; i < num; ++i) {
        TPCH_INT k = key[i];
        TPCH_INT p = pay[i];

        TPCH_INT s;
        TPCH_INT c;
        TPCH_INT m;

        auto sum = map_sum.find(k);
        auto cnt = map_cnt.find(k);
        auto mean = map_mean.find(k);
        if (sum != map_cnt.end()) {
            s = sum->second + p;
            c = cnt->second + 1;
            m = s / c;

            map_sum[k] = s; // update
            map_cnt[k] = c;
            map_mean[k] = m;
        } else {
            map_sum.insert(std::make_pair(k, p));
            map_cnt.insert(std::make_pair(k, 1));
            map_mean.insert(std::make_pair(k, m));
        }
    }

    return (TPCH_INT)map_sum.size();
}

TPCH_INT group_max(
    // input
    TPCH_INT* key,
    TPCH_INT* pay,
    int num,
    std::unordered_map<TPCH_INT, TPCH_INT>& ref_map) {
    for (int i = 0; i < num; ++i) {
        TPCH_INT k = key[i];
        TPCH_INT p = pay[i];
        auto it = ref_map.find(k);
        if (it != ref_map.end()) {
            TPCH_INT s = it->second > p ? it->second : p;
            ref_map[k] = s; // update
        } else {
            ref_map.insert(std::make_pair(k, p));
        }
    }

    return (TPCH_INT)ref_map.size();
}

TPCH_INT group_min(
    // input
    TPCH_INT* key,
    TPCH_INT* pay,
    int num,
    std::unordered_map<TPCH_INT, TPCH_INT>& ref_map) {
    for (int i = 0; i < num; ++i) {
        TPCH_INT k = key[i];
        TPCH_INT p = pay[i];
        auto it = ref_map.find(k);
        if (it != ref_map.end()) {
            TPCH_INT s = it->second > p ? p : it->second;
            ref_map[k] = s; // update
        } else {
            ref_map.insert(std::make_pair(k, p));
        }
    }

    return (TPCH_INT)ref_map.size();
}

TPCH_INT group_cnt_nz(
    // input
    TPCH_INT* key,
    TPCH_INT* pay,
    int num,
    std::unordered_map<TPCH_INT, TPCH_INT>& ref_map) {
    for (int i = 0; i < num; ++i) {
        TPCH_INT k = key[i];
        TPCH_INT p = pay[i];
        auto it = ref_map.find(k);
        if (it != ref_map.end()) {
            TPCH_INT s = p == 0 ? it->second : (it->second + 1);
            ref_map[k] = s; // update
        } else {
            ref_map.insert(std::make_pair(k, p == 0 ? 0 : 1));
        }
    }

    return (TPCH_INT)ref_map.size();
}

int check_result(ap_uint<1024>* data,
                 int num,
                 ap_uint<4> op,
                 ap_uint<32> key_col,
                 ap_uint<32> pld_col,
                 std::unordered_map<TPCH_INT, TPCH_INT>& ref_map) {
    int nerror = 0;
    int ncorrect = 0;
    ap_uint<8 * KEY_SZ * KEY_COL + 3 * 8 * MONEY_SZ * PLD_COL> result;

    for (int i = 0; i < num; i++) {
        result = data[i](8 * KEY_SZ * KEY_COL + 3 * 8 * MONEY_SZ * PLD_COL - 1, 0);
        ap_uint<3 * 8 * MONEY_SZ> p = result(3 * 8 * MONEY_SZ - 1, 0);

        TPCH_INT key = result(8 * KEY_SZ + 3 * 8 * MONEY_SZ * PLD_COL - 1, 3 * 8 * MONEY_SZ * PLD_COL);
        TPCH_INT pld;

        if (op == xf::database::enums::AOP_MIN || op == xf::database::enums::AOP_COUNT) {
            pld = p(8 * MONEY_SZ - 1, 0);
        } else if (op == xf::database::enums::AOP_COUNTNONZEROS) {
            pld = p(3 * 8 * MONEY_SZ - 1, 2 * 8 * MONEY_SZ);
        } else if (op == xf::database::enums::AOP_SUM || op == xf::database::enums::AOP_MEAN) {
            pld = p(3 * 8 * MONEY_SZ - 1, 8 * MONEY_SZ);
        } else {
            // not supported yet
        }

        std::cout << std::hex << "Checking: idx=" << i << " key:" << key << " pld:" << pld << std::endl;

        auto it = ref_map.find(key);
        if (it != ref_map.end()) {
            TPCH_INT golden_pld = it->second;
            if (pld != golden_pld) {
                std::cout << "ERROR! key:" << key << ", pld:" << pld << ", refpld:" << golden_pld << std::endl;
                ++nerror;
            } else {
                ++ncorrect;
            }
        } else {
            std::cout << "ERROR! k:" << key << " does not exist in ref" << std::endl;
            ++nerror;
        }
    }

    if (nerror == 0) {
        std::cout << "PASS! No error found!" << std::endl;
    } else {
        std::cout << "FAIL! Found " << nerror << " errors!" << std::endl;
    }
    return nerror;
}

void generate_test_dat(TPCH_INT* key, TPCH_INT* pld, size_t n) {
    for (int i = 0; i < n; i++) {
        key[i] = rand();
        pld[i] = rand();
    }
}

int main(int argc, const char* argv[]) {
    ap_uint<4> op = xf::database::enums::AOP_SUM;
    ap_uint<32> opt_type = (op, op, op, op, op, op, op, op);

    const size_t hbm_size = 32 << 10; // maximum is 256MB = 32<<20
    const size_t hbm_depth = 4 << 10; // maximum is 4M * 512b which is 4<<20
    const size_t l_depth = L_MAX_ROW;
    TPCH_INT* col_l_orderkey = (TPCH_INT*)malloc(l_depth * sizeof(TPCH_INT));
    TPCH_INT* col_l_extendedprice = (TPCH_INT*)malloc(l_depth * sizeof(TPCH_INT));
    std::cout << "Host map Buffer has been allocated.\n";

    generate_test_dat(col_l_orderkey, col_l_extendedprice, l_depth);

    // golden reference
    TPCH_INT result_cnt;
    if (op = xf::database::enums::AOP_SUM)
        result_cnt = group_sum(col_l_orderkey, col_l_extendedprice, l_depth, map0);
    else if (op == xf::database::enums::AOP_MAX)
        result_cnt = group_max(col_l_orderkey, col_l_extendedprice, l_depth, map0);
    else if (op == xf::database::enums::AOP_MIN)
        result_cnt = group_min(col_l_orderkey, col_l_extendedprice, l_depth, map0);
    else if (op == xf::database::enums::AOP_COUNT)
        result_cnt = group_cnt(col_l_orderkey, col_l_extendedprice, l_depth, map0);
    else if (op == xf::database::enums::AOP_COUNTNONZEROS)
        result_cnt = group_cnt_nz(col_l_orderkey, col_l_extendedprice, l_depth, map0);
    else if (op == xf::database::enums::AOP_MEAN)
        result_cnt = group_mean(col_l_orderkey, col_l_extendedprice, l_depth, map1, map2, map0);

    const size_t r_depth = R_MAX_ROW;

    ap_uint<1024>* aggr_result_buf; // result
    aggr_result_buf = (ap_uint<1024>*)malloc(r_depth * sizeof(ap_uint<1024>));

    ap_uint<32>* pu_begin_status;
    ap_uint<32>* pu_end_status;
    pu_begin_status = (ap_uint<32>*)malloc(PU_STATUS_DEPTH * sizeof(ap_uint<32>));
    pu_end_status = (ap_uint<32>*)malloc(PU_STATUS_DEPTH * sizeof(ap_uint<32>));

    pu_begin_status[0] = opt_type;
    pu_begin_status[1] = KEY_COL;
    pu_begin_status[2] = PLD_COL;
    pu_begin_status[3] = 0;

    for (int i = 0; i < PU_STATUS_DEPTH; i++) {
        std::cout << std::hex << "read_config: pu_begin_status[" << i << "]=" << pu_begin_status[i] << std::endl;
        pu_end_status[i] = 0;
    }

    ap_uint<512>* ping_buf0; // ping buffer
    ap_uint<512>* ping_buf1; // ping buffer
    ap_uint<512>* ping_buf2; // ping buffer
    ap_uint<512>* ping_buf3; // ping buffer
    ping_buf0 = (ap_uint<512>*)malloc(hbm_depth * sizeof(ap_uint<512>));
    ping_buf1 = (ap_uint<512>*)malloc(hbm_depth * sizeof(ap_uint<512>));
    ping_buf2 = (ap_uint<512>*)malloc(hbm_depth * sizeof(ap_uint<512>));
    ping_buf3 = (ap_uint<512>*)malloc(hbm_depth * sizeof(ap_uint<512>));

    ap_uint<512>* pong_buf0; // pong buffer
    ap_uint<512>* pong_buf1; // pong buffer
    ap_uint<512>* pong_buf2; // pong buffer
    ap_uint<512>* pong_buf3; // pong buffer
    pong_buf0 = (ap_uint<512>*)malloc(hbm_depth * sizeof(ap_uint<512>));
    pong_buf1 = (ap_uint<512>*)malloc(hbm_depth * sizeof(ap_uint<512>));
    pong_buf2 = (ap_uint<512>*)malloc(hbm_depth * sizeof(ap_uint<512>));
    pong_buf3 = (ap_uint<512>*)malloc(hbm_depth * sizeof(ap_uint<512>));

    hash_aggr_kernel((ap_uint<8 * KEY_SZ * VEC_LEN>*)col_l_orderkey,
                     (ap_uint<8 * MONEY_SZ * VEC_LEN>*)col_l_extendedprice, l_depth,

                     pu_begin_status, pu_end_status,

                     ping_buf0, ping_buf1, ping_buf2, ping_buf3, pong_buf0, pong_buf1, pong_buf2, pong_buf3,

                     aggr_result_buf);

    int agg_result_num = pu_end_status[3];
    int nerror = 0; // result_cnt!=agg_result_num;

    nerror = check_result(aggr_result_buf, agg_result_num, op, KEY_COL, PLD_COL, map0);

    std::cout << "ref_result_num=" << result_cnt << std::endl;
    std::cout << "kernel_result_num=" << agg_result_num << std::endl;
    std::cout << "---------------------------------------------\n" << std::endl;

    free(ping_buf0);
    free(ping_buf1);
    free(ping_buf2);
    free(ping_buf3);
    free(pong_buf0);
    free(pong_buf1);
    free(pong_buf2);
    free(pong_buf3);

    return nerror;
}
