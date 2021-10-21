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

#ifndef DATAMOVER_LOAD_MASTER_TO_STREAM_HPP
#define DATAMOVER_LOAD_MASTER_TO_STREAM_HPP

#include "ap_int.h"
#include "ap_axi_sdata.h"
#include "hls_stream.h"

namespace xf {
namespace datamover {

/**
 * Load from AXI master, and send data to stream.
 *
 * It is expected that the size is aligned with (WM/8) and (WS/8),
 * otherwise out-of-boundary data would be read/written.
 *
 * @tparam WM width of AXI master.
 * @tparam WS width of AXI stream.
 * @tparam BLEN burst length when reading from AXI master.
 *
 * @param mm master port.
 * @param s stream port.
 * @param size size of data to be loaded, in byte.
 */
template <int WM, int WS, int BLEN = 32>
void loadMasterToStream(ap_uint<WM>* mm, hls::stream<ap_axiu<WS, 0, 0, 0> >& s, uint64_t size) {
#ifndef __SYNTHESIS__
    static_assert(WM == WS, "WM should be equal to WS in the current implementation");
#endif
    const int bytePerData = WM / 8;
    const int nBlks = size / bytePerData + ((size % bytePerData) > 0);
    const int nBurst = nBlks / BLEN;
    int cnt = 0;
    for (int i = 0; i < nBurst; i++) {
        for (int j = 0; j < BLEN; j++) {
#pragma HLS pipeline II = 1
            ap_axiu<WS, 0, 0, 0> tmp;
            tmp.data = mm[cnt];
            tmp.keep = -1;
            tmp.last = 0;
            s.write(tmp);
            cnt++;
        }
    }
    int leftBlks = nBlks % BLEN;
    if (leftBlks > 0) {
        for (int i = 0; i < leftBlks; i++) {
#pragma HLS pipeline II = 1
            ap_axiu<WS, 0, 0, 0> tmp;
            tmp.data = mm[cnt];
            tmp.keep = -1;
            tmp.last = 0;
            s.write(tmp);
            cnt++;
        }
    }
}

} /* datamover */
} /* xf */
#endif
