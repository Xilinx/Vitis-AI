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
#ifndef HJKERNEL_H
#define HJKERNEL_H

#include "xf_database/types.hpp"
#include "ap_int.h"

// for test
#define WKEY 64   // width of key
#define WPAY 64   // width of payload
#define VEC_LEN 4 // channel number
#define HJ_MODE 1 // 0 - radix, 1 - Jenkins
#define WPUHASH 3 // hash into 2^3=8 PUs.
#define WHASH 7   // 17 is suggested in hw, each pu hash table have 2^17 = 128K entries. 8 PU have totally 1M entries.

#define NPU (1 << WPUHASH)
#define PU_HT_DEPTH (30 << 10) // 30M is suggested in hardware
#define PU_S_DEPTH (30 << 10)  // 30M is suggested in hardware
#define BUILD_CFG_DEPTH (2)    // depth, join_number

#define S_MAX_DEPTH ((1 << 14) / 4) // 1M row / 4 row per vec is suggeted in hardware
#define T_MAX_DEPTH ((1 << 14) / 4) // 1M row / 4 row per vec is suggested in hardware
#define J_MAX_DEPTH ((1 << 14))     // 1M row row per vec is suggested in hardware

extern "C" void hjkernel(
    // input
    uint32_t s_nrow,                                      // input, number of row in s unit
    ap_uint<(WKEY + WPAY) * VEC_LEN> s_unit[S_MAX_DEPTH], // input, 4 row per vec. DDR
    uint32_t t_nrow,                                      // input, number of row in t unit
    ap_uint<(WKEY + WPAY) * VEC_LEN> t_unit[T_MAX_DEPTH], // input, 4 row per vec. DDR

    // input hash-table
    ap_uint<256> pu0_ht[PU_HT_DEPTH], // PU0 hash-tables
    ap_uint<256> pu1_ht[PU_HT_DEPTH], // PU0 hash-tables
    ap_uint<256> pu2_ht[PU_HT_DEPTH], // PU0 hash-tables
    ap_uint<256> pu3_ht[PU_HT_DEPTH], // PU0 hash-tables
    ap_uint<256> pu4_ht[PU_HT_DEPTH], // PU0 hash-tables
    ap_uint<256> pu5_ht[PU_HT_DEPTH], // PU0 hash-tables
    ap_uint<256> pu6_ht[PU_HT_DEPTH], // PU0 hash-tables
    ap_uint<256> pu7_ht[PU_HT_DEPTH], // PU0 hash-tables

    // input S units
    ap_uint<256> pu0_s[PU_S_DEPTH], // PU0 S units
    ap_uint<256> pu1_s[PU_S_DEPTH], // PU0 S units
    ap_uint<256> pu2_s[PU_S_DEPTH], // PU0 S units
    ap_uint<256> pu3_s[PU_S_DEPTH], // PU0 S units
    ap_uint<256> pu4_s[PU_S_DEPTH], // PU0 S units
    ap_uint<256> pu5_s[PU_S_DEPTH], // PU0 S units
    ap_uint<256> pu6_s[PU_S_DEPTH], // PU0 S units
    ap_uint<256> pu7_s[PU_S_DEPTH], // PU0 S units

    // output join result
    ap_uint<32> hj_begin_status[BUILD_CFG_DEPTH], // status. DDR
    ap_uint<32> hj_end_status[BUILD_CFG_DEPTH],   // status. DDR
    ap_uint<512> j_res[J_MAX_DEPTH]               // output. DDR
    );

#endif
