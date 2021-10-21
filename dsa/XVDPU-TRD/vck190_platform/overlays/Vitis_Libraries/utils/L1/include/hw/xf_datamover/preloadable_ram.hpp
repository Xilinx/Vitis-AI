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

#ifndef DATAMOVER_PRELOADABLE_RAM_HPP
#define DATAMOVER_PRELOADABLE_RAM_HPP

#include "xf_datamover/types.hpp"
#include "ap_axi_sdata.h"
#include <hls_stream.h>

namespace xf {
namespace datamover {

/**
 * @class Preloadable BRAM.
 *
 * @tparam WS width of BRAM, same with width of AXI stream to AIE (only 32/64/128 are allowed).
 * @tparam DM depth of BRAM.
 */
template <int WS, int DM>
struct PreloadableBram {
   private:
    ap_uint<WS> data[DM];

   public:
    PreloadableBram() {
#pragma HLS inline
#pragma HLS resource variable = data core = RAM_2P_BRAM
    }

    /**
     * @brief Store constant stream to RAM.
     *
     * When size exceeds the maximum space of RAM,
     * the following data will overlap the RAM from the beginning.
     *
     * @param s constant stream.
     * @param size size of data to be stored into RAM, in byte.
     */
    void preload(hls::stream<ConstData::type>& s, const uint64_t size) {
        const int bytePerData = ConstData::Port_Width / 8;
        int nBlks = size / bytePerData + ((size % bytePerData) > 0);
        ConstData::type sIn;
        ap_uint<WS> bIn;
        // WS = 32
        if (WS / ConstData::Port_Width == 1) {
            for (int i = 0; i < nBlks; i++) {
#pragma HLS pipeline II = 1
                sIn = s.read();
                bIn = sIn;
                data[i % DM] = bIn;
            }
            // WS = 64
        } else if (WS / ConstData::Port_Width == 2) {
            for (int i = 0; i < nBlks; i++) {
#pragma HLS pipeline II = 1
                if (i % 2 == 0) {
                    bIn = 0;
                }
                sIn = s.read();
                bIn.range(WS - 1, WS - ConstData::Port_Width) = sIn;
                data[(i / 2) % DM] = bIn;
                bIn = bIn >> ConstData::Port_Width;
            }
            // last block is not full block
            if (nBlks % 2 != 0) {
                data[((nBlks - 1) / 2) % DM] = bIn;
            }
            // WS = 128
        } else {
            for (int i = 0; i < nBlks; i++) {
#pragma HLS pipeline II = 1
                if (i % 4 == 0) {
                    bIn = 0;
                }
                sIn = s.read();
                bIn.range(WS - 1, WS - ConstData::Port_Width) = sIn;
                data[(i / 4) % DM] = bIn;
                bIn = bIn >> ConstData::Port_Width;
            }
            // last block is not full block
            if (nBlks % 4 != 0) {
                bIn = bIn >> (ConstData::Port_Width * (3 - (nBlks % 4)));
                data[((nBlks - 1) / 4) % DM] = bIn;
            }
        }
    }

    /**
     * @brief Send data from RAM to AXI stream.
     *
     * When more data than RAM size is required,
     * RAM data would be read from beginning again.
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
     * @brief Check data from AXI stream with RAM.
     *
     * When more data than RAM size is required,
     * RAM data would be read from beginning again.
     *
     * @param s stream port.
     * @param ret result stream.
     * @param size size of data to be compared, in byte.
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

/**
 * @class Preloadable URAM.
 *
 * Please be noticed that when the width of AXI stream 'WS' is 32,
 * you will waste half of the URAM as the width of it is 64.
 *
 * @tparam WS width of URAM, same with width of AXI stream to AIE (only 32/64/128 are allowed).
 * @tparam DM depth of URAM.
 */
template <int WS, int DM>
struct PreloadableUram {
   private:
    ap_uint<WS> data[DM];

   public:
    PreloadableUram() {
#pragma HLS inline
#pragma HLS resource variable = data core = RAM_2P_URAM
    }

    /**
     * @brief Store constant stream to RAM.
     *
     * When size exceeds the maximum space of RAM,
     * the following data will overlap the RAM from the beginning.
     *
     * @param s constant stream.
     * @param size size of data to be stored into RAM, in byte.
     */
    void preload(hls::stream<ConstData::type>& s, const uint64_t size) {
        const int bytePerData = ConstData::Port_Width / 8;
        int nBlks = size / bytePerData + ((size % bytePerData) > 0);
        ConstData::type sIn;
        ap_uint<WS> bIn;
        // WS = 32
        if (WS / ConstData::Port_Width == 1) {
            for (int i = 0; i < nBlks; i++) {
#pragma HLS pipeline II = 1
                sIn = s.read();
                bIn = sIn;
                data[i % DM] = bIn;
            }
            // WS = 64
        } else if (WS / ConstData::Port_Width == 2) {
            for (int i = 0; i < nBlks; i++) {
#pragma HLS pipeline II = 1
                if (i % 2 == 0) {
                    bIn = 0;
                }
                sIn = s.read();
                bIn.range(WS - 1, WS - ConstData::Port_Width) = sIn;
                data[(i / 2) % DM] = bIn;
                bIn = bIn >> ConstData::Port_Width;
            }
            // last block is not full block
            if (nBlks % 2 != 0) {
                data[((nBlks - 1) / 2) % DM] = bIn;
            }
            // WS = 128
        } else {
            for (int i = 0; i < nBlks; i++) {
#pragma HLS pipeline II = 1
                if (i % 4 == 0) {
                    bIn = 0;
                }
                sIn = s.read();
                bIn.range(WS - 1, WS - ConstData::Port_Width) = sIn;
                data[(i / 4) % DM] = bIn;
                bIn = bIn >> ConstData::Port_Width;
            }
            // last block is not full block
            if (nBlks % 4 != 0) {
                bIn = bIn >> (ConstData::Port_Width * (3 - (nBlks % 4)));
                data[((nBlks - 1) / 4) % DM] = bIn;
            }
        }
    }

    /**
     * @brief Send data from RAM to AXI stream.
     *
     * When more data than RAM size is required,
     * RAM data would be read from beginning again.
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
     * @brief Check data from AXI stream with RAM.
     *
     * When more data than RAM size is required,
     * RAM data would be read from beginning again.
     *
     * @param s stream port.
     * @param ret result stream.
     * @param size size of data to be compared, in byte.
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
