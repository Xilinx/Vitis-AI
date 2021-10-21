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

#define W_TPCH_INT (8 * sizeof(TPCH_INT))

#define AP_INT_MAX_W 4096
#include <ap_int.h>

#define L_DEPTH (L_MAX_ROW / VEC_LEN + 64)
#define O_DEPTH (O_MAX_ROW / VEC_LEN + 64)

#define HJ_HW_J (17)
// pu number is 1 << HW_P
#define HJ_HW_P (3)
// 0 - radix, 1 - Jenkins
#define HJ_MODE (1)
// channel number
#define HJ_CH_NM (4)

#define BUFF_DEPTH (O_MAX_ROW / 8 * 2)

#define PU_HT_DEPTH (30 << 20) // 30M
#define PU_S_DEPTH (30 << 20)  // 30M
#define BUILD_CFG_DEPTH (2)    // depth, join_number

extern "C" void join_kernel(const int join_flag,
                            ap_uint<W_TPCH_INT * VEC_LEN> buf_o_orderkey[O_DEPTH],
                            const int o_nrow,
                            ap_uint<W_TPCH_INT * VEC_LEN> buf_l_orderkey[L_DEPTH],
                            ap_uint<W_TPCH_INT * VEC_LEN> buf_l_extendedprice[L_DEPTH],
                            ap_uint<W_TPCH_INT * VEC_LEN> buf_l_discount[L_DEPTH],
                            const int l_nrow,
                            // tune
                            const int k_bucket,
                            // temp
                            ap_uint<256> pu0_ht[PU_HT_DEPTH], // PU0 hash-tables
                            ap_uint<256> pu1_ht[PU_HT_DEPTH], // PU0 hash-tables
                            ap_uint<256> pu2_ht[PU_HT_DEPTH], // PU0 hash-tables
                            ap_uint<256> pu3_ht[PU_HT_DEPTH], // PU0 hash-tables
                            ap_uint<256> pu4_ht[PU_HT_DEPTH], // PU0 hash-tables
                            ap_uint<256> pu5_ht[PU_HT_DEPTH], // PU0 hash-tables
                            ap_uint<256> pu6_ht[PU_HT_DEPTH], // PU0 hash-tables
                            ap_uint<256> pu7_ht[PU_HT_DEPTH], // PU0 hash-tables

                            ap_uint<256> pu0_s[PU_S_DEPTH], // PU0 S units
                            ap_uint<256> pu1_s[PU_S_DEPTH], // PU0 S units
                            ap_uint<256> pu2_s[PU_S_DEPTH], // PU0 S units
                            ap_uint<256> pu3_s[PU_S_DEPTH], // PU0 S units
                            ap_uint<256> pu4_s[PU_S_DEPTH], // PU0 S units
                            ap_uint<256> pu5_s[PU_S_DEPTH], // PU0 S units
                            ap_uint<256> pu6_s[PU_S_DEPTH], // PU0 S units
                            ap_uint<256> pu7_s[PU_S_DEPTH], // PU0 S units

                            // output
                            ap_uint<W_TPCH_INT * 2> buf_result[1]);
