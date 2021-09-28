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

#define W_STRM 512
#define W_PU 64
#define NS (8 * 1024)
#define NPU 8

#include "xf_utils_hw/types.hpp"

void round_robin_mpu(hls::stream<ap_uint<W_STRM> >& istrm,
                     hls::stream<bool>& e_istrm,
                     hls::stream<ap_uint<W_STRM> >& ostrm,
                     hls::stream<bool>& e_ostrm);

#endif // _XF_HW_UTILS_TEST_CODE_HPP
