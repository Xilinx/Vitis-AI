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

#include <vector> // std::vector
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <algorithm>

#include "hls_stream.h"

#include "mjkernel.hpp"
#include "xf_database/hash_multi_join.hpp"
#include "xf_database/enums.hpp"

#define TEST_LENGTH_S 100
#define TEST_LENGTH_T 100
#define ANTI_RATE 0.9
#ifndef __SYNTHESIS__
//--------------------------------scan-----------------------------------
static void scan(ap_uint<(WKEY + WPAY) * VEC_LEN> unit[T_MAX_DEPTH],
                 int num,

                 hls::stream<ap_uint<WKEY> >& k_strms,
                 hls::stream<ap_uint<WPAY> >& p_strms,
                 hls::stream<bool>& o_e_strm) {
    ap_uint<WKEY> k_strms_temp;
    ap_uint<WPAY> p_strms_temp;

    for (int i = 0; i < num; i++) {
        for (int j = 0; j < VEC_LEN; j++) {
            k_strms_temp = unit[i]((WKEY + WPAY) * (j + 1) - 1, (WKEY + WPAY) * j + WPAY);
            p_strms_temp = unit[i]((WKEY + WPAY) * j + WPAY - 1, (WKEY + WPAY) * j);

            k_strms.write(k_strms_temp);
            p_strms.write(p_strms_temp);
            o_e_strm.write(false);
        }
    }
    o_e_strm.write(true);
}

//-------------------------generate test data----------------------------
void generate_data(
    // test hash join build kernel
    ap_uint<(WKEY + WPAY) * VEC_LEN> s_unit[T_MAX_DEPTH],
    ap_uint<(WKEY + WPAY) * VEC_LEN> t_unit[T_MAX_DEPTH],
    int num_s,
    int num_t,

    // for computing golden data
    hls::stream<ap_uint<WKEY> >& o_s_key_strm,
    hls::stream<ap_uint<WPAY> >& o_s_pld_strm,
    hls::stream<bool>& o_e0_strm,

    hls::stream<ap_uint<WKEY> >& o_t_key_strm,
    hls::stream<ap_uint<WPAY> >& o_t_pld_strm,
    hls::stream<bool>& o_e1_strm) {
    // generate s&t unit
    for (int i = 0; i < num_s; i++) {
        for (int j = 0; j < VEC_LEN; j++) {
            ap_uint<WKEY> s_key;
            s_key = rand() % (num_s / 10);
            ap_uint<WPAY> s_pld = rand();

            ap_uint<WKEY + WPAY> srow = (s_key, s_pld);

            s_unit[i]((j + 1) * (WKEY + WPAY) - 1, j * (WKEY + WPAY)) = srow;
        }
    }
    for (int i = 0; i < num_t; i++) {
        for (int j = 0; j < VEC_LEN; j++) {
            ap_uint<WKEY> t_key;
            t_key = rand() % (int(num_s / 10 / (1 - ANTI_RATE)));
            ap_uint<WPAY> t_pld = rand();

            ap_uint<WKEY + WPAY> trow = (t_key, t_pld);

            t_unit[i]((j + 1) * (WKEY + WPAY) - 1, j * (WKEY + WPAY)) = trow;
        }
    }

    // scan s-table
    scan(s_unit, num_s, o_s_key_strm, o_s_pld_strm, o_e0_strm);

    // scan t-table
    scan(t_unit, num_t, o_t_key_strm, o_t_pld_strm, o_e1_strm);
}

//-------------------------generate golden data-----------------------------
template <int test_num>
void hash_join_golden(xf::database::enums::JoinType join_flag,
                      hls::stream<ap_uint<WKEY> >& i_s_key_strm,
                      hls::stream<ap_uint<WPAY> >& i_s_pld_strm,
                      hls::stream<bool>& i_e0_strm,

                      hls::stream<ap_uint<WKEY> >& i_t_key_strm,
                      hls::stream<ap_uint<WPAY> >& i_t_pld_strm,
                      hls::stream<bool>& i_e1_strm,

                      hls::stream<ap_uint<WKEY + 2 * WPAY> >& o_j_strm,
                      hls::stream<bool>& o_e_strm) {
    bool slast, tlast;
    uint32_t cnt = 0;

    ap_uint<WKEY + WPAY> row_temp;
    ap_uint<WKEY + WPAY> srow_table[test_num * VEC_LEN];
    ap_uint<WKEY + 2 * WPAY> j_temp;

    // generate s-table
    slast = i_e0_strm.read();
    while (!slast) {
        row_temp(WKEY + WPAY - 1, WPAY) = i_s_key_strm.read();
        row_temp(WPAY - 1, 0) = i_s_pld_strm.read();
        slast = i_e0_strm.read();

        srow_table[cnt] = row_temp;
        cnt++;
    }
    int datacount = 0;
    tlast = i_e1_strm.read();
    while (!tlast) {
        ap_uint<WKEY> t_key = i_t_key_strm.read();
        ap_uint<WPAY> t_pld = i_t_pld_strm.read();
        tlast = i_e1_strm.read();
        bool flag = 0;
        for (int i = 0; i < cnt; i++) {
            ap_uint<WKEY> s_key = srow_table[i](WKEY + WPAY - 1, WPAY);
            ap_uint<WPAY> s_pld = srow_table[i](WPAY - 1, 0);

            if (s_key == t_key) {
                if (join_flag == xf::database::enums::JT_INNER) {
                    j_temp(WKEY + 2 * WPAY - 1, 2 * WPAY) = t_key;
                    j_temp(2 * WPAY - 1, WPAY) = s_pld;
                    j_temp(WPAY - 1, 0) = t_pld;
                    std::cout << std::hex << "Golden Data:" << j_temp << std::dec << " " << ++datacount << std::endl;

                    o_j_strm.write(j_temp);
                    o_e_strm.write(false);
                }
                if (join_flag == xf::database::enums::JT_SEMI && flag == 0) {
                    j_temp(WKEY + 2 * WPAY - 1, 2 * WPAY) = t_key;
                    j_temp(2 * WPAY - 1, WPAY) = 0;
                    j_temp(WPAY - 1, 0) = t_pld;
                    std::cout << std::hex << "Golden Data:" << j_temp << std::dec << " " << ++datacount << std::endl;

                    o_j_strm.write(j_temp);
                    o_e_strm.write(false);
                }
                flag = 1;
            }
        }
        if (join_flag == xf::database::enums::JT_ANTI) {
            if (flag == 0) {
                j_temp(WKEY + 2 * WPAY - 1, 2 * WPAY) = t_key;
                j_temp(2 * WPAY - 1, WPAY) = 0;
                j_temp(WPAY - 1, 0) = t_pld;
                std::cout << std::hex << "Golden Data:" << j_temp << std::dec << " " << ++datacount << std::endl;

                o_j_strm.write(j_temp);
                o_e_strm.write(false);
            }
        }
    }
    o_e_strm.write(true);
}

//-----------------------------------compare data---------------------------------
template <int test_num>
int check_data(xf::database::enums::JoinType join_type,
               hls::stream<ap_uint<WKEY + 2 * WPAY> >& o_j_strm,
               hls::stream<bool>& o_e_strm,

               ap_uint<512> j_res[J_MAX_DEPTH]) {
    int nerror;
    int error;
    ap_uint<512> j_temp;
    int datacount = 0;
    bool last = o_e_strm.read();
    while (!last) {
        j_temp = o_j_strm.read();
        last = o_e_strm.read();

        for (int i = 0; i < J_MAX_DEPTH; i++) {
            if (j_temp == j_res[i]) {
                error = 0;
                if (join_type == xf::database::enums::JT_INNER)
                    std::cout << std::hex << "Hash-Join:" << j_res[i] << " " << std::dec << ++datacount << std::endl;
                if (join_type == xf::database::enums::JT_ANTI)
                    std::cout << std::hex << "Anti-Join:" << j_res[i] << " " << std::dec << ++datacount << std::endl;
                if (join_type == xf::database::enums::JT_SEMI)
                    std::cout << std::hex << "Semi-Join:" << j_res[i] << " " << std::dec << ++datacount << std::endl;
                break;
            } else {
                error = 1;
            }
        }
        nerror += error;

        if (error) std::cout << std::hex << "Unit Not Found: " << j_temp << std::endl;
    }
    return error;
}

int main() {
    const int PU_NM = 1 << WPUHASH;
    const int nrow_s = TEST_LENGTH_S;
    const int nrow_t = TEST_LENGTH_T;

    // generate s&t table
    ap_uint<(WKEY + WPAY) * VEC_LEN>* s_unit;
    ap_uint<(WKEY + WPAY) * VEC_LEN>* t_unit;
    s_unit = (ap_uint<(WKEY + WPAY) * VEC_LEN>*)malloc(S_MAX_DEPTH * sizeof(ap_uint<(WKEY + WPAY) * VEC_LEN>));
    t_unit = (ap_uint<(WKEY + WPAY) * VEC_LEN>*)malloc(T_MAX_DEPTH * sizeof(ap_uint<(WKEY + WPAY) * VEC_LEN>));

    hls::stream<ap_uint<WKEY> > s_key_strm;
    hls::stream<ap_uint<WPAY> > s_pld_strm;
    hls::stream<bool> s_e_strm;

    hls::stream<ap_uint<WKEY> > t_key_strm;
    hls::stream<ap_uint<WPAY> > t_pld_strm;
    hls::stream<bool> t_e_strm;

    xf::database::enums::JoinType join_type = xf::database::enums::JT_INNER;

    generate_data(s_unit, t_unit, nrow_s, nrow_t, s_key_strm, s_pld_strm, s_e_strm, t_key_strm, t_pld_strm, t_e_strm);

    // allocate internal buffer
    ap_uint<256>* pu_ht[PU_NM];
    ap_uint<256>* pu_s[PU_NM];
    ap_uint<512>* j_res0;

    for (int i = 0; i < PU_NM; i++) {
        pu_ht[i] = (ap_uint<256>*)malloc(PU_HT_DEPTH * sizeof(ap_uint<256>));
        pu_s[i] = (ap_uint<256>*)malloc(PU_S_DEPTH * sizeof(ap_uint<256>));
    }

    j_res0 = (ap_uint<512>*)malloc(J_MAX_DEPTH * sizeof(ap_uint<512>));

    // status
    ap_uint<32> hj_begin_status[BUILD_CFG_DEPTH]; // status. DDR
    ap_uint<32> hj_end_status[BUILD_CFG_DEPTH];   // status. DDR

    hj_begin_status[0] = 4; // depth
    hj_begin_status[1] = 0; // join_number

    // call build
    std::cout << "------------------------kernel start--------------------------" << std::endl;

    mjkernel(
        // input
        (uint32_t)join_type,
        nrow_s, // input, number of row in s unit
        s_unit, // input, 4 row per vec. DDR
        nrow_t, t_unit,

        // output hash-table
        pu_ht[0], // PU0 hash-tables
        pu_ht[1], // PU0 hash-tables
        pu_ht[2], // PU0 hash-tables
        pu_ht[3], // PU0 hash-tables
        pu_ht[4], // PU0 hash-tables
        pu_ht[5], // PU0 hash-tables
        pu_ht[6], // PU0 hash-tables
        pu_ht[7], // PU0 hash-tables

        // output S units
        pu_s[0], // PU0 S units
        pu_s[1], // PU0 S units
        pu_s[2], // PU0 S units
        pu_s[3], // PU0 S units
        pu_s[4], // PU0 S units
        pu_s[5], // PU0 S units
        pu_s[6], // PU0 S units
        pu_s[7], // PU0 S units

        // join result
        hj_begin_status, hj_end_status, j_res0);

    // generate golden data
    hls::stream<ap_uint<WKEY + 2 * WPAY> > j_strm;
    hls::stream<bool> j_e_strm;

    hash_join_golden<nrow_s>(join_type, s_key_strm, s_pld_strm, s_e_strm, t_key_strm, t_pld_strm, t_e_strm, j_strm,
                             j_e_strm);

    // check
    int nerror;
    nerror = check_data<nrow_s>(join_type, j_strm, j_e_strm, j_res0);

    for (int i = 0; i < PU_NM; i++) {
        free(pu_ht[i]);
        free(pu_s[i]);
    }

    free(j_res0);
    free(s_unit);
    free(t_unit);

    // print result
    if (nerror) {
        std::cout << "\nFAIL: " << nerror << " errors found.\n";
    } else {
        std::cout << "\nPASS: no error found.\n";
    }
    return 0;
}
#endif
