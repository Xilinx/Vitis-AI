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

#define W_STRM 128
#define W_PU 16
#define W_PRC 8
#define W_DSC 8
#define NS (1024 * 2 * 2)
#define NPU 8

ap_uint<W_PU> update_data(ap_uint<W_PU> data);

void test_core(hls::stream<ap_uint<W_STRM> >& istrm,
               hls::stream<bool>& e_istrm,
               hls::stream<ap_uint<W_STRM> >& ostrm,
               hls::stream<bool>& e_ostrm);

#endif // _XF_HW_UTILS_TEST_CODE_HPP
