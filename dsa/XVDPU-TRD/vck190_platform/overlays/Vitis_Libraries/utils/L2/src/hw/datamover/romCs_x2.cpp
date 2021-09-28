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
#include "xf_datamover/static_rom.hpp"
#include "xf_datamover/write_result.hpp"
#include "xf_datamover/types.hpp"

template <class T0, class T1>
void romCs_x2_run(
    // 0
    T0& rom0,
    hls::stream<ap_axiu<64, 0, 0, 0> >& s0,
    xf::datamover::CheckResult::type* ret0,
    uint64_t sz0,

    // 1
    T1& rom1,
    hls::stream<ap_axiu<16, 0, 0, 0> >& s1,
    xf::datamover::CheckResult::type* ret1,
    uint64_t sz1

    ) {
    using namespace xf::datamover;

#pragma HLS dataflow

    hls::stream<CheckResult::type, 2> rs0;
    hls::stream<CheckResult::type, 2> rs1;

    rom0.checkStream(s0, rs0, sz0);
    rom1.checkStream(s1, rs1, sz1);
    writeResult(rs0, ret0, rs1, ret1);
}

extern "C" void romCs_x2(

    // 0
    hls::stream<ap_axiu<64, 0, 0, 0> >& s0,
    xf::datamover::CheckResult::type* ret0,
    uint64_t sz0,

    // 1
    hls::stream<ap_axiu<16, 0, 0, 0> >& s1,
    xf::datamover::CheckResult::type* ret1,
    uint64_t sz1

    ) {
    using namespace xf::datamover;

    ; // clang-format off
#pragma HLS interface axis port=s0
#pragma HLS interface m_axi offset=slave bundle=gmemr port=ret0 \
    max_write_burst_length=32 num_write_outstanding=4 latency=128
#pragma HLS interface s_axilite bundle=control port=ret0
#pragma HLS interface s_axilite bundle=control port=sz0
#pragma HLS interface axis port=s1
#pragma HLS interface m_axi offset=slave bundle=gmemr port=ret1 \
    max_write_burst_length=32 num_write_outstanding=4 latency=128
#pragma HLS interface s_axilite bundle=control port=ret1
#pragma HLS interface s_axilite bundle=control port=sz1
#pragma HLS interface s_axilite bundle=control port=return
    ; // clang-format on

    StaticRom<64, 512> rom0;
    const ap_uint<64> in0[] = {
#include "d_double.txt.inc"
    };
    rom0.data = in0;

    StaticRom<16, 512> rom1;
    const ap_uint<16> in1[] = {
#include "d_half.txt.inc"
    };
    rom1.data = in1;

    romCs_x2_run(rom0, s0, ret0, sz0, rom1, s1, ret1, sz1);
}
