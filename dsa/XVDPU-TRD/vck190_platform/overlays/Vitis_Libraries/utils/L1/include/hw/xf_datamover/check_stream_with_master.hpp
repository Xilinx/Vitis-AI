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

#ifndef DATAMOVER_CHECK_STREAM_WITH_MASTER_HPP
#define DATAMOVER_CHECK_STREAM_WITH_MASTER_HPP

#include "hls_stream.h"
#include "ap_axi_sdata.h"
#include "xf_datamover/types.hpp"

namespace xf {
namespace datamover {

/**
 * Load from AXI master and stream, validate and save result to buffer.
 *
 * It is expected that the size is aligned with (WM/8) and (WS/8),
 * otherwise out-of-boundary data would be read/written.
 *
 * @tparam WM width of AXI master.
 * @tparam WS width of AXI stream.
 * @tparam BLEN burst length when reading from AXI master.
 *
 * @param s stream port.
 * @param gm master port for golden data.
 * @param ret check result.
 * @param size size of data to be loaded, in byte.
 */
template <int WM, int WS, int BLEN = 32>
void checkStreamWithMaster(hls::stream<ap_axiu<WS, 0, 0, 0> >& s,
                           ap_uint<WM>* gm,
                           hls::stream<CheckResult::type>& ret,
                           uint64_t size) {
#ifndef __SYNTHESIS__
    static_assert(WM == WS, "WM should be equal to WS in the current implementation");
#endif
    const int bytePerData = WM / 8;
    const int nBlks = size / bytePerData + ((size % bytePerData) > 0);
    const int nBurst = nBlks / BLEN;
    int cnt = 0;
    CheckResult::type flag = 1;
    for (int i = 0; i < nBurst; i++) {
        for (int j = 0; j < BLEN; j++) {
#pragma HLS pipeline II = 1
            ap_axiu<WS, 0, 0, 0> tmp = s.read();
            if (tmp.data != gm[cnt]) {
                flag = 0;
            }
            cnt++;
        }
    }
    int leftBlks = nBlks % BLEN;
    if (leftBlks > 0) {
        for (int i = 0; i < leftBlks; i++) {
#pragma HLS pipeline II = 1
            ap_axiu<WS, 0, 0, 0> tmp = s.read();
            if (tmp.data != gm[cnt]) {
                flag = 0;
            }
            cnt++;
        }
    }
    ret.write(flag);
}

} /* datamover */
} /* xf */
#endif
