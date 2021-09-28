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
#ifndef _DUT_H_
#define _DUT_H_
#include "ap_int.h"
#include "hls_stream.h"
#include "xf_utils_hw/uram_array.hpp"

// as reference uram size 4K*256
#define WDATA (64)
#define NDATA (16 << 10)
#define NCACHE (4)

#define NUM_SIZE (1 << 10)

void dut(ap_uint<WDATA> ii, hls::stream<ap_uint<WDATA> >& out_stream);

#endif // _DUT_H_
