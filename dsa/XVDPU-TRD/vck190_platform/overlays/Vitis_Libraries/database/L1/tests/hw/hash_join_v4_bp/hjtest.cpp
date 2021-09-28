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

#include "hjkernel.hpp"
#include "xf_database/hash_join_v3.hpp"

#define TEST_LENGTH 100

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
    int num,

    // for computing golden data
    hls::stream<ap_uint<WKEY> >& o_s_key_strm,
    hls::stream<ap_uint<WPAY> >& o_s_pld_strm,
    hls::stream<bool>& o_e0_strm,

    hls::stream<ap_uint<WKEY> >& o_t_key_strm,
    hls::stream<ap_uint<WPAY> >& o_t_pld_strm,
    hls::stream<bool>& o_e1_strm) {
    // generate s&t unit
    for (int i = 0; i < num; i++) {
        for (int j = 0; j < VEC_LEN; j++) {
            ap_uint<WKEY> key = rand();
            ap_uint<WPAY> s_pld = rand();
            ap_uint<WPAY> t_pld = rand();

            ap_uint<WKEY + WPAY> srow = (key, s_pld);
            ap_uint<WKEY + WPAY> trow = (key, t_pld);

            s_unit[i]((j + 1) * (WKEY + WPAY) - 1, j * (WKEY + WPAY)) = srow;
            t_unit[i]((j + 1) * (WKEY + WPAY) - 1, j * (WKEY + WPAY)) = trow;
        }
    }

    // scan s-table
    scan(s_unit, num, o_s_key_strm, o_s_pld_strm, o_e0_strm);

    // scan t-table
    scan(t_unit, num, o_t_key_strm, o_t_pld_strm, o_e1_strm);
}

//-------------------------generate golden data-----------------------------
template <int test_num>
void hash_join_golden(hls::stream<ap_uint<WKEY> >& i_s_key_strm,
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

    tlast = i_e1_strm.read();
    while (!tlast) {
        ap_uint<WKEY> t_key = i_t_key_strm.read();
        ap_uint<WPAY> t_pld = i_t_pld_strm.read();
        tlast = i_e1_strm.read();

        for (int i = 0; i < cnt; i++) {
            ap_uint<WKEY> s_key = srow_table[i](WKEY + WPAY - 1, WPAY);
            ap_uint<WPAY> s_pld = srow_table[i](WPAY - 1, 0);

            if (s_key == t_key) {
                j_temp(WKEY + 2 * WPAY - 1, 2 * WPAY) = s_key;
                j_temp(2 * WPAY - 1, WPAY) = s_pld;
                j_temp(WPAY - 1, 0) = t_pld;

                o_j_strm.write(j_temp);
                o_e_strm.write(false);

                std::cout << std::hex << "Golden Data: " << j_temp << std::endl;
            }
        }
    }
    o_e_strm.write(true);
}

//-----------------------------------compare data---------------------------------
template <int test_num>
int check_data(hls::stream<ap_uint<WKEY + 2 * WPAY> >& o_j_strm,
               hls::stream<bool>& o_e_strm,

               ap_uint<512> j_res[J_MAX_DEPTH]) {
    int nerror;
    int error;
    ap_uint<512> j_temp;

    bool last = o_e_strm.read();
    while (!last) {
        j_temp = o_j_strm.read();
        last = o_e_strm.read();

        for (int i = 0; i < J_MAX_DEPTH; i++) {
            if (j_temp == j_res[i]) {
                error = 0;
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
    const int nrow = TEST_LENGTH;

    // generate s&t table
    ap_uint<(WKEY + WPAY) * VEC_LEN>* s_unit;
    ap_uint<(WKEY + WPAY) * VEC_LEN>* t_unit;
    s_unit = (ap_uint<(WKEY + WPAY) * VEC_LEN>*)malloc(T_MAX_DEPTH * sizeof(ap_uint<(WKEY + WPAY) * VEC_LEN>));
    t_unit = (ap_uint<(WKEY + WPAY) * VEC_LEN>*)malloc(T_MAX_DEPTH * sizeof(ap_uint<(WKEY + WPAY) * VEC_LEN>));

    hls::stream<ap_uint<WKEY> > s_key_strm;
    hls::stream<ap_uint<WPAY> > s_pld_strm;
    hls::stream<bool> s_e_strm;

    hls::stream<ap_uint<WKEY> > t_key_strm;
    hls::stream<ap_uint<WPAY> > t_pld_strm;
    hls::stream<bool> t_e_strm;

    generate_data(s_unit, t_unit, nrow, s_key_strm, s_pld_strm, s_e_strm, t_key_strm, t_pld_strm, t_e_strm);

    // status
    ap_uint<32> hj_begin_status[BUILD_CFG_DEPTH]; // status. DDR
    ap_uint<32> hj_end_status[BUILD_CFG_DEPTH];   // status. DDR

    hj_begin_status[0] = 0; // build_id
    hj_begin_status[1] = 0; // probe id
    hj_begin_status[2] = 4; // fixed depth for every hash
    hj_begin_status[3] = 0; // join_num
    for (int i = 0; i < 8; i++) {
        hj_begin_status[i + 4] = 0;
    }

    // allocate internal buffer
    ap_uint<64>* pu_ht[PU_NM];
    ap_uint<64>* pu_s[PU_NM];
    ap_uint<512>* j_res0;
    ap_uint<512>* j_res1;

    for (int i = 0; i < PU_NM; i++) {
        pu_ht[i] = (ap_uint<64>*)malloc(PU_HT_DEPTH * sizeof(ap_uint<64>));
        pu_s[i] = (ap_uint<64>*)malloc(PU_S_DEPTH * sizeof(ap_uint<64>));
    }

    j_res0 = (ap_uint<512>*)malloc(J_MAX_DEPTH * sizeof(ap_uint<512>));
    j_res1 = (ap_uint<512>*)malloc(J_MAX_DEPTH * sizeof(ap_uint<512>));

    // call build
    std::cout << "------------------------build start--------------------------" << std::endl;

    hjkernel(
        // input
        false,
        nrow,   // input, number of row in s unit
        s_unit, // input, 4 row per vec. DDR

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

        // status
        hj_begin_status, // status. DDR
        hj_end_status,   // status. DDR
        j_res0);

    // call probe
    std::cout << "-----------------------probe start0-------------------------" << std::endl;

    hjkernel(
        // input
        true,
        nrow,   // input, number of row in s unit
        t_unit, // input, 4 row per vec. DDR

        // input hash-table
        pu_ht[0], // PU0 hash-tables
        pu_ht[1], // PU0 hash-tables
        pu_ht[2], // PU0 hash-tables
        pu_ht[3], // PU0 hash-tables
        pu_ht[4], // PU0 hash-tables
        pu_ht[5], // PU0 hash-tables
        pu_ht[6], // PU0 hash-tables
        pu_ht[7], // PU0 hash-tables

        // input S units
        pu_s[0], // PU0 S units
        pu_s[1], // PU0 S units
        pu_s[2], // PU0 S units
        pu_s[3], // PU0 S units
        pu_s[4], // PU0 S units
        pu_s[5], // PU0 S units
        pu_s[6], // PU0 S units
        pu_s[7], // PU0 S units

        // status
        hj_end_status, // status. DDR
        hj_begin_status, j_res0);

    std::cout << "-----------------------probe start1-------------------------" << std::endl;

    hjkernel(
        // input
        true,
        nrow,   // input, number of row in s unit
        t_unit, // input, 4 row per vec. DDR

        // input hash-table
        pu_ht[0], // PU0 hash-tables
        pu_ht[1], // PU0 hash-tables
        pu_ht[2], // PU0 hash-tables
        pu_ht[3], // PU0 hash-tables
        pu_ht[4], // PU0 hash-tables
        pu_ht[5], // PU0 hash-tables
        pu_ht[6], // PU0 hash-tables
        pu_ht[7], // PU0 hash-tables

        // input S units
        pu_s[0], // PU0 S units
        pu_s[1], // PU0 S units
        pu_s[2], // PU0 S units
        pu_s[3], // PU0 S units
        pu_s[4], // PU0 S units
        pu_s[5], // PU0 S units
        pu_s[6], // PU0 S units
        pu_s[7], // PU0 S units

        // status
        hj_end_status, // status. DDR
        hj_begin_status, j_res1);

    // generate golden data
    hls::stream<ap_uint<WKEY + 2 * WPAY> > j_strm;
    hls::stream<bool> j_e_strm;

    hash_join_golden<nrow>(s_key_strm, s_pld_strm, s_e_strm, t_key_strm, t_pld_strm, t_e_strm, j_strm, j_e_strm);

    // check
    int nerror;
    nerror = check_data<nrow>(j_strm, j_e_strm, j_res1);

    for (int i = 0; i < PU_NM; i++) {
        free(pu_ht[i]);
        free(pu_s[i]);
    }
    free(j_res0);
    free(j_res1);
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
