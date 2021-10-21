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

// top header
#include "eval_dut.hpp"

void eval2_dut(hls::stream<ap_uint<32> >& cfgs,
               //
               hls::stream<int>& col0_istrm,
               hls::stream<int>& col1_istrm,
               hls::stream<int>& col2_istrm,
               hls::stream<int>& col3_istrm,
               hls::stream<bool>& e_istrm,
               //
               hls::stream<int>& ret_ostrm,
               hls::stream<bool>& e_ostrm) {
    xf::database::dynamicEvalV2(cfgs, col0_istrm, col1_istrm, col2_istrm, col3_istrm, e_istrm, ret_ostrm, e_ostrm);
}
