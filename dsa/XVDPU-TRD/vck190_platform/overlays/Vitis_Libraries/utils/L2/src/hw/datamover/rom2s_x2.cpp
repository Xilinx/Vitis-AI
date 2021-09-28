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
#include "xf_datamover/types.hpp"

template <class T0, class T1>
void rom2s_x2_run(
    // 0
    T0& rom0,
    hls::stream<ap_axiu<64, 0, 0, 0> >& s0,
    uint64_t sz0,

    // 1
    T1& rom1,
    hls::stream<ap_axiu<32, 0, 0, 0> >& s1,
    uint64_t sz1

    ) {
    using namespace xf::datamover;

#pragma HLS dataflow

    rom0.toStream(s0, sz0);
    rom1.toStream(s1, sz1);
}

extern "C" void rom2s_x2(

    // 0
    hls::stream<ap_axiu<64, 0, 0, 0> >& s0,
    uint64_t sz0,

    // 1
    hls::stream<ap_axiu<32, 0, 0, 0> >& s1,
    uint64_t sz1

    ) {
    using namespace xf::datamover;

    ; // clang-format off
#pragma HLS interface axis port=s0
#pragma HLS interface s_axilite bundle=control port=sz0

#pragma HLS interface axis port=s1
#pragma HLS interface s_axilite bundle=control port=sz1

#pragma HLS interface s_axilite bundle=control port=return
    ; // clang-format on

    StaticRom<64, 512> rom0;
    const ap_uint<64> in0[] = {
#include "d_int64.txt.inc"
    };
    rom0.data = in0;

    StaticRom<32, 1024> rom1;
    const ap_uint<32> in1[] = {
#include "d_int32.txt.inc"
    };
    rom1.data = in1;

    rom2s_x2_run(rom0, s0, sz0, rom1, s1, sz1);
}
