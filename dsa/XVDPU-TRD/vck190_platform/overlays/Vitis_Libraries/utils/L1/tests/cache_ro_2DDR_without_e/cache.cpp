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
#include "xf_utils_hw/cache.hpp"

// XXX: macro with same name as template parameter is defined in this header, for easy matching up.
// so this header must be included AFTER the API header under test.
#include "cache_tb.hpp"

void syn_top(int size,
             hls::stream<ap_uint<BUSADDRWIDTH> >& raddrStrm,
             hls::stream<ap_uint<BUSDATAWIDTH> >& rdataStrm0,
             hls::stream<ap_uint<BUSDATAWIDTH> >& rdataStrm1,
             ap_uint<512>* ddrMem0,
             ap_uint<512>* ddrMem1) {
    const int ddrsize = DDRSIZEIN512;
#pragma HLS INTERFACE m_axi offset = slave latency = 64 depth = ddrsize num_write_outstanding = \
    1 num_read_outstanding = 256 max_write_burst_length = 2 max_read_burst_length = 2 bundle = gmem0_0 port = ddrMem0
#pragma HLS INTERFACE m_axi offset = slave latency = 64 depth = ddrsize num_write_outstanding = \
    1 num_read_outstanding = 256 max_write_burst_length = 2 max_read_burst_length = 2 bundle = gmem0_1 port = ddrMem1

#pragma HLS INTERFACE s_axilite port = ddrMem0 bundle = control
#pragma HLS INTERFACE s_axilite port = ddrMem1 bundle = control

    xf::common::utils_hw::cache<ap_uint<BUSDATAWIDTH>, RAMROW, GRPRAM, EACHLINE, BUSADDRWIDTH, 0, 1, 2> dut;

    dut.initDualOffChip();

    dut.readOnly(size, ddrMem0, ddrMem1, raddrStrm, rdataStrm0, rdataStrm1);
}
