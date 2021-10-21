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

#define AP_INT_MAX_W 4096
#include <ap_int.h>

#define TPCH_INT int32_t
#define L_MAX_ROW 1000
#define R_MAX_ROW 1000
#define KEY_SZ 4
#define MONEY_SZ 4
#define VEC_LEN 4
#define HBM_DEPTH (32 << 10)
#define L_DEPTH L_MAX_ROW
#define R_DEPTH R_MAX_ROW

// PU_STATUS_DEPTH is the number of status parameter
#define PU_STATUS_DEPTH (4)
// HASH_MODE: 1->lookup3 0->radix
#define HMODE (1)
// 1<<PU_HASH is the number of PU
#define PU_HASH (2)
// 1<<WHASH is number of hash entries
#define WHASH (10) // max is 17
// KEY_COL is the column number of key
#define KEY_COL (8)
// PLD_COL is the column number of pld
#define PLD_COL (8)
// WCNT is the width of cnt for each entries, WCNT=max(KEY_COL,PLD_COL)
#define WCNT (8)

extern "C" void hash_aggr_kernel(ap_uint<8 * KEY_SZ * VEC_LEN> buf_l_orderkey[L_DEPTH],
                                 ap_uint<8 * MONEY_SZ * VEC_LEN> buf_l_extendedprice[L_DEPTH],
                                 const int l_nrow,

                                 ap_uint<32> config[PU_STATUS_DEPTH],
                                 ap_uint<32> result_info[PU_STATUS_DEPTH],

                                 ap_uint<512> ping_buf0[L_DEPTH],
                                 ap_uint<512> ping_buf1[L_DEPTH],
                                 ap_uint<512> ping_buf2[L_DEPTH],
                                 ap_uint<512> ping_buf3[L_DEPTH],
                                 ap_uint<512> pong_buf0[L_DEPTH],
                                 ap_uint<512> pong_buf1[L_DEPTH],
                                 ap_uint<512> pong_buf2[L_DEPTH],
                                 ap_uint<512> pong_buf3[L_DEPTH],

                                 ap_uint<1024> result[R_DEPTH]); // result k1;
