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
#include "xf_database/dynamic_filter.hpp"

void filter_dut(hls::stream<xf::database::DynamicFilterInfo<4, WKEY>::cfg_type>& filter_cfg_strm,
                hls::stream<ap_uint<WKEY> > k_strms[4],
                hls::stream<ap_uint<WPAY> >& p_strm,
                hls::stream<bool>& e_strm,
                // out
                hls::stream<ap_uint<WPAY> >& f_strm,
                hls::stream<bool>& e_f_strm);

#endif
