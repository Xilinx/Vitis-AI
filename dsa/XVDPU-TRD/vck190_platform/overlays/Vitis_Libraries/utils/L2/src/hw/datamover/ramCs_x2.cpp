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
#include "xf_datamover/preloadable_ram.hpp"
#include "xf_datamover/read_const.hpp"
#include "xf_datamover/write_result.hpp"
#include "xf_datamover/types.hpp"

template <class T0, class T1>
void ramCs_x2_preload(
    // 0
    xf::datamover::ConstData::type* din0,
    T0& ram0,
    uint64_t sz0,

    // 1
    xf::datamover::ConstData::type* din1,
    T1& ram1,
    uint64_t sz1

    ) {
    using namespace xf::datamover;

#pragma HLS dataflow

    hls::stream<ConstData::type, 8> is0;
    hls::stream<ConstData::type, 8> is1;

    readConst(din0, is0, sz0, din1, is1, sz1);

    ram0.preload(is0, sz0);
    ram1.preload(is1, sz1);
}

template <class T0, class T1>
void ramCs_x2_run(
    // 0
    T0& ram0,
    hls::stream<ap_axiu<64, 0, 0, 0> >& s0,
    xf::datamover::CheckResult::type* ret0,
    uint64_t sz0,

    // 1
    T1& ram1,
    hls::stream<ap_axiu<32, 0, 0, 0> >& s1,
    xf::datamover::CheckResult::type* ret1,
    uint64_t sz1

    ) {
    using namespace xf::datamover;

#pragma HLS dataflow

    hls::stream<CheckResult::type, 2> rs0;
    hls::stream<CheckResult::type, 2> rs1;

    ram0.checkStream(s0, rs0, sz0);
    ram1.checkStream(s1, rs1, sz1);
    writeResult(rs0, ret0, rs1, ret1);
}

extern "C" void ramCs_x2(
    // common
    xf::datamover::MoverMode mode,

    // 0
    xf::datamover::ConstData::type* din0,
    hls::stream<ap_axiu<64, 0, 0, 0> >& s0,
    xf::datamover::CheckResult::type* ret0,
    uint64_t sz0,

    // 1
    xf::datamover::ConstData::type* din1,
    hls::stream<ap_axiu<32, 0, 0, 0> >& s1,
    xf::datamover::CheckResult::type* ret1,
    uint64_t sz1

    ) {
    using namespace xf::datamover;

    ; // clang-format off
#pragma HLS interface s_axilite bundle=control port=mode

#pragma HLS interface m_axi offset=slave bundle=gmem0 port=din0 \
    max_read_burst_length=32 num_read_outstanding=4 latency=128
#pragma HLS interface s_axilite bundle=control port=din0
#pragma HLS interface axis port=s0
#pragma HLS interface m_axi offset=slave bundle=gmemr port=ret0 \
    max_read_burst_length=32 num_read_outstanding=4 latency=128
#pragma HLS interface s_axilite bundle=control port=ret0
#pragma HLS interface s_axilite bundle=control port=sz0

#pragma HLS interface m_axi offset=slave bundle=gmem1 port=din1 \
    max_read_burst_length=32 num_read_outstanding=4 latency=128
#pragma HLS interface s_axilite bundle=control port=din1
#pragma HLS interface axis port=s1
#pragma HLS interface m_axi offset=slave bundle=gmemr port=ret1 \
    max_read_burst_length=32 num_read_outstanding=4 latency=128
#pragma HLS interface s_axilite bundle=control port=ret1
#pragma HLS interface s_axilite bundle=control port=sz1

#pragma HLS interface s_axilite bundle=control port=return
    ; // clang-format on

    PreloadableUram<64, 512> ram0;
    PreloadableBram<32, 1024> ram1;

    if (mode == MODE_PRELOAD) {
        ramCs_x2_preload(din0, ram0, sz0, din1, ram1, sz1);
    } else if (mode == MODE_RUN) {
        ramCs_x2_run(ram0, s0, ret0, sz0, ram1, s1, ret1, sz1);
    }
}
