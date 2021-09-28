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

#include <hls_stream.h>

#include "xf_database/hash_partition.hpp"

#include "part_dut.hpp"

void part_dut(
    // input
    hls::stream<int>& bit_num_strm,

    hls::stream<ap_uint<64> > k0_strm_arry[CH_NM],
    hls::stream<ap_uint<192> > p0_strm_arry[CH_NM],
    hls::stream<bool> e0_strm_arry[CH_NM],

    // output
    hls::stream<ap_uint<16> >& o_bkpu_num_strm,
    hls::stream<ap_uint<10> >& o_nm_strm,
    hls::stream<ap_uint<EW> > o_kpld_strm[COL_NM]) {
    // core dut
    xf::database::hashPartition<1, 64, 192, 32, 0, 8, 18, CH_NM, COL_NM>(
        0, 512, bit_num_strm, k0_strm_arry, p0_strm_arry, e0_strm_arry, o_bkpu_num_strm, o_nm_strm, o_kpld_strm);
}
