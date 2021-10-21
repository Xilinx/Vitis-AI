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

#define D_DEPTH (L_MAX_ROW / VEC_LEN + 1)

#define Q5_HJ_HW_J 17
#define Q5_HJ_AW 19
#define Q5_HJ_BVW 20 // only used when enable bloomfilter

#define Q5_HJ_HW_P 3
#define Q5_HJ_HDP_J (1 << (Q5_HJ_HW_J - 2))
#define Q5_HJ_HDP_P (1 << Q5_HJ_HW_P)
#define Q5_HJ_PU_NM Q5_HJ_HDP_P // number of process unit
#define Q5_HJ_MODE (1)          // 0 -radix 1 - Jenkins
#define Q5_HJ_CH_NM (4)         // channel number

// platform information
#define BUFF_DEPTH (O_MAX_ROW / 8 * 2)

#define KEY_NUM 2
#define PLD_NUM 3
#define OUT_COL_NUM 4

extern "C" void q5_hash_join(ap_uint<8 * TPCH_INT_SZ * VEC_LEN> buf_skey1[D_DEPTH],
                             ap_uint<8 * TPCH_INT_SZ * VEC_LEN> buf_spay1[D_DEPTH],
                             //
                             ap_uint<8 * TPCH_INT_SZ * VEC_LEN> buf_bkey1[D_DEPTH],
                             ap_uint<8 * TPCH_INT_SZ * VEC_LEN> buf_bpay1[D_DEPTH],
                             ap_uint<8 * TPCH_INT_SZ * VEC_LEN> buf_bpay2[D_DEPTH],
                             ap_uint<8 * TPCH_INT_SZ * VEC_LEN> buf_bpay3[D_DEPTH],
                             //
                             ap_uint<8 * TPCH_INT_SZ * VEC_LEN> buf_okey1[D_DEPTH],
                             ap_uint<8 * TPCH_INT_SZ * VEC_LEN> buf_opay1[D_DEPTH],
                             ap_uint<8 * TPCH_INT_SZ * VEC_LEN> buf_opay2[D_DEPTH],
                             ap_uint<8 * TPCH_INT_SZ * VEC_LEN> buf_opay3[D_DEPTH],
                             //
                             ap_uint<8 * TPCH_INT_SZ * 4> buf0[BUFF_DEPTH],
                             ap_uint<8 * TPCH_INT_SZ * 4> buf1[BUFF_DEPTH],
                             ap_uint<8 * TPCH_INT_SZ * 4> buf2[BUFF_DEPTH],
                             ap_uint<8 * TPCH_INT_SZ * 4> buf3[BUFF_DEPTH],
                             ap_uint<8 * TPCH_INT_SZ * 4> buf4[BUFF_DEPTH],
                             ap_uint<8 * TPCH_INT_SZ * 4> buf5[BUFF_DEPTH],
                             ap_uint<8 * TPCH_INT_SZ * 4> buf6[BUFF_DEPTH],
                             ap_uint<8 * TPCH_INT_SZ * 4> buf7[BUFF_DEPTH],
                             const int idx,
                             const int enable_filter,
                             const int config);
