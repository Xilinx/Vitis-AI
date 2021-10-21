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

#ifndef PART_DUT_HPP
#define PART_DUT_HPP

// used modules
#include <ap_int.h>
#include <hls_stream.h>

#define CH_NM 1
#define COL_NM 8

#define KEYW 64
#define PW 192
#define EW 32

void part_dut(
    // input
    hls::stream<int>& bit_num_strm,

    hls::stream<ap_uint<KEYW> > k0_strm_arry[CH_NM],
    hls::stream<ap_uint<PW> > p0_strm_arry[CH_NM],
    hls::stream<bool> e0_strm_arry[CH_NM],

    // output
    hls::stream<ap_uint<16> >& o_bkpu_num_strm,
    hls::stream<ap_uint<10> >& o_nm_strm,
    hls::stream<ap_uint<EW> > o_kpld_strm[COL_NM]);

#endif
