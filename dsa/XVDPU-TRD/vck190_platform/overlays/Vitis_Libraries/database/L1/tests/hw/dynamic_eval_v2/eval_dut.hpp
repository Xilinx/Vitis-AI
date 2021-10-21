/*
 * Copyright 2020 Xilinx, Inc.
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

#ifndef FILTER_DUT_HPP
#define FILTER_DUT_HPP

#define WKEY 32
#define WPAY 32

// used modules
#include <ap_int.h>
#include <hls_stream.h>

// to have enums on host side.
#include "xf_database/dynamic_eval_v2.hpp"

void eval2_dut(hls::stream<ap_uint<32> >& cfgs,
               //
               hls::stream<int>& col0_istrm,
               hls::stream<int>& col1_istrm,
               hls::stream<int>& col2_istrm,
               hls::stream<int>& col3_istrm,
               hls::stream<bool>& e_istrm,
               //
               hls::stream<int>& ret_ostrm,
               hls::stream<bool>& e_ostrm);
#endif
