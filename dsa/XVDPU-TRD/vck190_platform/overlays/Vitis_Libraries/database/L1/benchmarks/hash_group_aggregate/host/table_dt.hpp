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

#ifndef TABLE_DT_H
#define TABLE_DT_H

// XXX inline with tpch_read_2.h
#include <stdint.h>
typedef int32_t TPCH_INT;
typedef int64_t TPCH_INT64;

typedef TPCH_INT MONEY_T;
typedef TPCH_INT DATE_T;
typedef TPCH_INT KEY_T;

typedef TPCH_INT64 MONEY_T64;
typedef TPCH_INT64 KEY_T64;

#define MONEY_SZ sizeof(MONEY_T)
#define DATE_SZ sizeof(DATE_T)
#define KEY_SZ sizeof(KEY_T)

// every cycle, 1 input rows. if VEC_LEN=16,then the AXI will be TPCH_INT*8(=512) width
// and achieve perfect transmission speed. this vec_len affact both kernel and host
#define VEC_LEN (4)
// the total distinct of key must be less than NUM_REP*((2^(1+HJ_HW_J))
#define NUM_REP (150)
// the times of the kernel launched, to calculate the average time
#define NUM_REP_HOST (3)
// NAGG is the num of agg_function:(max,min,sum,cnt)=4
#define NAGG (4)
// HJ_HW_J is width of hash by bit, which decide the scale of uram, default 14 could handle 2*16K rows in one work flow.
// when the aggragate key's distinct is in 32K ,the kernel works in high-performence(267 Mrows/s)
// when the aggragate key's distinct is over 32K ,the kernel works in functional-performence (157 Mrows/s)
#define HJ_HW_J (8)
// ensure when kernel read in vec, won't over read
#define L_MAX_ROW (6000000)
#define O_MAX_ROW (1500000)
// use 3uram will make the kernel 3 level uram to process while default is 2 level//to be public

// small distinct case is the high-performence distinct for default kernel, 6000 distinct for 3uram kernel
//#define SMALL_DISTINCT_CASE
// print result will make the host print time far more longer than the execution time, default lock it
#define PRINT_RESULT
#endif // TABLE_DT_H
