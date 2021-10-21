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

#include <stdint.h>
#include "xf_datamover/store_stream_to_master.hpp"

extern "C" void s2m_x2(
    // 0
    hls::stream<ap_axiu<64, 0, 0, 0> >& s0,
    ap_uint<64>* p0,
    uint64_t sz0,

    // 1
    hls::stream<ap_axiu<32, 0, 0, 0> >& s1,
    ap_uint<32>* p1,
    uint64_t sz1

    ) {
    using namespace xf::datamover;

    ; // clang-format off
#pragma HLS interface axis port=s0
#pragma HLS interface m_axi offset=slave bundle=gmem0 port=p0 \
    max_write_burst_length=32 num_write_outstanding=4 latency=128
#pragma HLS interface s_axilite bundle=control port=p0
#pragma HLS interface s_axilite bundle=control port=sz0

#pragma HLS interface axis port=s1
#pragma HLS interface m_axi offset=slave bundle=gmem1 port=p1 \
    max_write_burst_length=32 num_write_outstanding=4 latency=128
#pragma HLS interface s_axilite bundle=control port=p1
#pragma HLS interface s_axilite bundle=control port=sz1

#pragma HLS interface s_axilite bundle=control port=return
    ; // clang-format on

#pragma HLS dataflow

    storeStreamToMaster(s0, p0, sz0);
    storeStreamToMaster(s1, p1, sz1);
}
