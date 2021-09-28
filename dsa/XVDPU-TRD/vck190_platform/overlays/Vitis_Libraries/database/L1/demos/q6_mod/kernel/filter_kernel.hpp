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

#ifndef FILTER_KERNEL_H
#define FILTER_KERNEL_H

#include "table_dt.hpp"

#define AP_INT_MAX_W 4096
#include <ap_int.h>

#ifdef MINI_TEST
#define L_DEPTH MINI_TEST
#else
#define L_DEPTH ((L_MAX_ROW / VEC_LEN) + 1)
#endif

#define BURST_LEN 64

extern "C" void filter_kernel(
    // config/op
    ap_uint<32> buf_filter_cfg[],
    // input, condition columns
    ap_uint<8 * KEY_SZ * VEC_LEN> buf_l_shipdate[L_DEPTH],
    ap_uint<8 * MONEY_SZ * VEC_LEN> buf_l_discount[L_DEPTH],
    ap_uint<8 * TPCH_INT_SZ * VEC_LEN> buf_l_quantity[L_DEPTH],
    ap_uint<8 * KEY_SZ * VEC_LEN> buf_l_commitdate[L_DEPTH],
    // input, payload column
    ap_uint<8 * MONEY_SZ * VEC_LEN> buf_l_extendedprice[L_DEPTH],
    // input, size of workload
    const int l_nrow,
    // output
    ap_uint<8 * MONEY_SZ * 2> buf_revenue[1]);

#include "xf_database/enums.hpp"

#endif // FILTER_KERNEL_H
