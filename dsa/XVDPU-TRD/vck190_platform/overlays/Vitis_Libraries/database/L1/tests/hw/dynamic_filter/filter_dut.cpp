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

// top header
#include "filter_dut.hpp"

void filter_dut(hls::stream<xf::database::DynamicFilterInfo<4, 32>::cfg_type>& filter_cfg_strm,
                hls::stream<ap_uint<WKEY> > k_strms[4],
                hls::stream<ap_uint<WPAY> >& p_strm,
                hls::stream<bool>& e_strm,
                // out
                hls::stream<ap_uint<WPAY> >& f_strm,
                hls::stream<bool>& e_f_strm) {
    xf::database::dynamicFilter(filter_cfg_strm, k_strms[0], k_strms[1], k_strms[2], k_strms[3], p_strm, e_strm, //
                                f_strm, e_f_strm);
}
