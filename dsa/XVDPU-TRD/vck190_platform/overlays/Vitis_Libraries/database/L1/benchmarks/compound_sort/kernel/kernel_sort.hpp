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

#ifndef _KERNEL_SORT_HPP_
#define _KERNEL_SORT_HPP_

#include <ap_int.h>
#include <hls_stream.h>
#include "xf_database/compound_sort.hpp"

#define INSERT_LEN 1024
#define KEY_BW 32
#define DATA_BW 32
#define BW (KEY_BW + DATA_BW)
#define LEN (INSERT_LEN * 4 * 32) // max length support: 1024*4*512
typedef ap_uint<32> KEY_TYPE;
typedef ap_uint<32> DATA_TYPE;

extern "C" void SortKernel(int order, int keyLength, KEY_TYPE inKey[LEN], KEY_TYPE outKey[LEN]);

#endif //_KERNEL_SORT_HPP_
