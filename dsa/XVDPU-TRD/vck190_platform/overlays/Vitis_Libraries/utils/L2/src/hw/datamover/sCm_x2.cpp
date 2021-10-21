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
#include "xf_datamover/check_stream_with_master.hpp"
#include "xf_datamover/write_result.hpp"

extern "C" void sCm_x2(
    // 0
    hls::stream<ap_axiu<64, 0, 0, 0> >& s0,
    ap_uint<64>* g0,
    xf::datamover::CheckResult::type* ret0,
    uint64_t sz0,

    // 1
    hls::stream<ap_axiu<32, 0, 0, 0> >& s1,
    ap_uint<32>* g1,
    xf::datamover::CheckResult::type* ret1,
    uint64_t sz1

    ) {
    using namespace xf::datamover;

    ; // clang-format off
#pragma HLS interface axis port=s0
#pragma HLS interface m_axi offset=slave bundle=gmem0 port=g0 \
    max_read_burst_length=32 num_read_outstanding=4 latency=128
#pragma HLS interface s_axilite bundle=control port=g0
#pragma HLS interface m_axi offset=slave bundle=gmemr port=ret0 \
    max_read_burst_length=32 num_read_outstanding=4 latency=128
#pragma HLS interface s_axilite bundle=control port=ret0
#pragma HLS interface s_axilite bundle=control port=sz0
#pragma HLS interface axis port=s1
#pragma HLS interface m_axi offset=slave bundle=gmem1 port=g1 \
    max_read_burst_length=32 num_read_outstanding=4 latency=128
#pragma HLS interface s_axilite bundle=control port=g1
#pragma HLS interface m_axi offset=slave bundle=gmemr port=ret1 \
    max_read_burst_length=32 num_read_outstanding=4 latency=128
#pragma HLS interface s_axilite bundle=control port=ret1
#pragma HLS interface s_axilite bundle=control port=sz1
#pragma HLS interface s_axilite bundle=control port=return
    ; // clang-format on

#pragma HLS dataflow

    hls::stream<CheckResult::type, 2> rs0;
    hls::stream<CheckResult::type, 2> rs1;

    checkStreamWithMaster(s0, g0, rs0, sz0);
    checkStreamWithMaster(s1, g1, rs1, sz1);

    writeResult(rs0, ret0, rs1, ret1);
}
