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
#ifndef KERNEL_HPP
#define KERNEL_HPP

#define AP_INT_MAX_W 4096
#include "ap_int.h"

#define BURST_LEN 32
#define VEC_LEN 8

#define BUF_DEPTH (1 << 8)

extern "C" {
void Test(ap_uint<64 * VEC_LEN> buf0[BUF_DEPTH], ap_uint<64 * VEC_LEN> buf1[BUF_DEPTH], int nrow, ap_uint<64>* bufo);
}

#endif
