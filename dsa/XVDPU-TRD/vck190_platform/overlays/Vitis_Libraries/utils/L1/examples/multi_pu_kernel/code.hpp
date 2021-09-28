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

#ifndef _XF_HW_UTILS_TEST_CODE_HPP
#define _XF_HW_UTILS_TEST_CODE_HPP

#include "xf_utils_hw/types.hpp"

#define W_STRM 256
#define W_PU 32
#define W_PRC 16
#define W_DSC 16
#define NPU 8

#define W_AXI (W_STRM * 2)
#define BURST_LENTH 32
#define DATA_LEN (4096 * 16)
#define W_DATA W_PU

// the type of input and output data
typedef ap_uint<W_DATA> t_data;
// the type of inner stream
typedef ap_uint<W_STRM> t_strm;

// the depth of axi port
// It must meet  DDR_DEPTH * W_AXI >= W_DATA * DATA_LEN
const int DDR_DEPTH = DATA_LEN * W_DATA / W_AXI;

ap_uint<W_PU> update_data(ap_uint<W_PU> data);

ap_uint<W_PU> calculate(ap_uint<W_PU> data);

void top_core(ap_uint<W_AXI>* in_buf, ap_uint<W_AXI>* out_buf, const int len);

#endif // _XF_HW_UTILS_TEST_CODE_HPP
