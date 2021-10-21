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

#include "table_dt.hpp"

#define AP_INT_MAX_W 4096
#include <ap_int.h>

#define L_DEPTH (L_MAX_ROW / VEC_LEN)
#define O_DEPTH (O_MAX_ROW / VEC_LEN)

#define Q5E2_HJ_HW_J 18
#define Q5E2_HJ_AW 19
#define Q5E2_HJ_BVW 20 // only used when enable bloomfilter
#define Q5E2_HJ_HW_P 3
#define Q5E2_HJ_PU_NM (1 << Q5E2_HJ_HW_P) // number of process unit
#define Q5E2_HJ_MODE (1)                  // 0 -radix 1 - Jenkins
#define Q5E2_HJ_CH_NM 4                   // channel number

// platform information
#define BUFF_DEPTH (O_MAX_ROW / 8 * 2)

extern "C" void q5simplified(ap_uint<8 * KEY_SZ * VEC_LEN> buf_l_orderkey[L_DEPTH],
                             ap_uint<8 * MONEY_SZ * VEC_LEN> buf_l_extendedprice[L_DEPTH],
                             ap_uint<8 * MONEY_SZ * VEC_LEN> buf_l_discount[L_DEPTH],
                             const int l_nrow,
                             ap_uint<8 * KEY_SZ * VEC_LEN> buf_o_orderkey[O_DEPTH],
                             ap_uint<8 * DATE_SZ * VEC_LEN> buf_o_orderdate[O_DEPTH],
                             const int o_nrow,
                             // temp
                             ap_uint<8 * KEY_SZ> buf0[BUFF_DEPTH],
                             ap_uint<8 * KEY_SZ> buf1[BUFF_DEPTH],
                             ap_uint<8 * KEY_SZ> buf2[BUFF_DEPTH],
                             ap_uint<8 * KEY_SZ> buf3[BUFF_DEPTH],
                             ap_uint<8 * KEY_SZ> buf4[BUFF_DEPTH],
                             ap_uint<8 * KEY_SZ> buf5[BUFF_DEPTH],
                             ap_uint<8 * KEY_SZ> buf6[BUFF_DEPTH],
                             ap_uint<8 * KEY_SZ> buf7[BUFF_DEPTH],
                             // output
                             ap_uint<8 * MONEY_SZ * 2> buf_result[1]);
