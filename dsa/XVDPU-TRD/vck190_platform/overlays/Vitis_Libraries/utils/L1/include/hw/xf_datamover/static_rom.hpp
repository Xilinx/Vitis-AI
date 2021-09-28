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

#ifndef DATAMOVER_STATIC_ROM_HPP
#define DATAMOVER_STATIC_ROM_HPP

#include "xf_datamover/types.hpp"
#include "ap_axi_sdata.h"
#include <hls_stream.h>

namespace xf {
namespace datamover {

/**
 * @class Static ROM.
 *
 * @tparam WS width of ROM, same with width of AXI stream to AIE (only 32/64/128 are allowed).
 * @tparam DM depth of ROM.
 */
template <int WS, int DM>
struct StaticRom {
    /// ROM
    const ap_uint<WS>* data;

    StaticRom() {
#pragma HLS inline
#pragma HLS resource variable = data core = ROM_2P
    }

    /**
     * @brief Send data from ROM to AXI stream.
     *
     * When more data than ROM size is required,
     * ROM data would be read from beginning again.
     *
     * @param s stream port.
     * @param size size of data to be streamed, in byte.
     */
    void toStream(hls::stream<ap_axiu<WS, 0, 0, 0> >& s, const uint64_t size) {
        const int bytePerData = WS / 8;
        int nBlks = size / bytePerData + ((size % bytePerData) > 0);
        for (int i = 0; i < nBlks; i++) {
#pragma HLS pipeline II = 1
            ap_axiu<WS, 0, 0, 0> tmp;
            tmp.keep = -1;
            tmp.last = 0;
            tmp.data = data[i % DM];
            s.write(tmp);
        }
    }

    /**
     * @brief Check data from AXI stream with ROM.
     *
     * When more data than ROM size is required,
     * ROM data would be read from beginning again.
     *
     * @param s stream port.
     * @param ret result stream.
     * @param size size of data to be streamed, in byte.
     */
    void checkStream(hls::stream<ap_axiu<WS, 0, 0, 0> >& s, hls::stream<CheckResult::type>& ret, const uint64_t size) {
        const int bytePerData = WS / 8;
        int fullBlks = size / bytePerData;
        CheckResult::type flag = 1;
        for (int i = 0; i < fullBlks; i++) {
#pragma HLS pipeline II = 1
            ap_axiu<WS, 0, 0, 0> tmp = s.read();
            if (tmp.data != data[i % DM]) {
                flag = 0;
            }
        }
        int leftBytes = size % bytePerData;
        // have partial block
        if (leftBytes > 0) {
            ap_axiu<WS, 0, 0, 0> tmp = s.read();
            if (tmp.data.range(leftBytes * 8 - 1, 0) != data[fullBlks % DM].range(leftBytes * 8 - 1, 0)) {
                flag = 0;
            }
        }
        ret.write(flag);
    }
};

} /* datamover */
} /* xf */
#endif
