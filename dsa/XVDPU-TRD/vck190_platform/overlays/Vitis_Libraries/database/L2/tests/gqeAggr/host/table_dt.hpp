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

typedef TPCH_INT MONEY_T;
typedef TPCH_INT DATE_T;
typedef TPCH_INT KEY_T;

#define MONEY_SZ sizeof(TPCH_INT)
#define DATE_SZ sizeof(TPCH_INT)
#define KEY_SZ sizeof(TPCH_INT)
#define REGION_LEN 25

#define TPCH_INT_SZ sizeof(TPCH_INT)

// every cycle, 4 input rows.
#define VEC_LEN 16

//
// ensure when kernel read in vec, won't over read
#define C_MAX_ROW (150000)
#define S_MAX_ROW (10000)
#define L_MAX_ROW (6001215)
#define O_MAX_ROW (1500000)
#define R_MAX_ROW (5)
#define N_MAX_ROW (25)

#define BUFF_DEPTH (O_MAX_ROW / 8 * 2)

#define HT_BUFF_DEPTH (1 << 20)  // 30M
#define S_BUFF_DEPTH (1 << 20)   // 30M
#define HBM_BUFF_DEPTH (1 << 20) // 30M

#endif // TABLE_DT_H
