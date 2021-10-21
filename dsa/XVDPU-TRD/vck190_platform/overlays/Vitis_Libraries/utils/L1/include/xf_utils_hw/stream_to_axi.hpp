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

/**
 * @file stream_to_axi.hpp
 * @brief Stream_to_axi template function implementation.
 *
 * This file is part of XF Common Utils Library.
 */

#ifndef XF_UTILS_HW_STRM_TO_AXI_H
#define XF_UTILS_HW_STRM_TO_AXI_H

#include "xf_utils_hw/common.hpp"

// Forward decl

namespace xf {
namespace common {
namespace utils_hw {

/**
 * @brief Write elements in burst to AXI master port.
 *
 * @tparam _BurstLen length of a burst, default is 32.
 * @tparam _WAxi     width of axi port.
 * @tparam _WStrm    width of input stream.
 *
 * @param wbuf    output AXI port.
 * @param istrm   input stream.
 * @param e_istrm end flag for input stream
 */
template <int _BurstLen = 32, int _WAxi, int _WStrm>
void streamToAxi(ap_uint<_WAxi>* wbuf, hls::stream<ap_uint<_WStrm> >& istrm, hls::stream<bool>& e_istrm);
} // utils_hw
} // common
} // xf

namespace xf {
namespace common {
namespace utils_hw {
namespace details {

/**
 * @brief the template of convert stream width from _WStrm to _WAxi and count
 * burst number.
 *
 * @tparam _WAxi   width of axi port.
 * @tparam _WStrm  width of input stream.
 * @tparam _BurstLen length of a burst.
 *
 * @param istrm   input stream.
 * @param e_istrm  end flag for input stream
 * @param axi_strm stream width is _WAxi
 * @param nb_strm  store burst number of each burst
 */
template <int _WAxi, int _WStrm, int _BurstLen>
void countForBurst(hls::stream<ap_uint<_WStrm> >& istrm,
                   hls::stream<bool>& e_istrm,
                   hls::stream<ap_uint<_WAxi> >& axi_strm,
                   hls::stream<ap_uint<8> >& nb_strm) {
    const int N = _WAxi / _WStrm;
    ap_uint<_WAxi> tmp;
    bool isLast;
    int nb = 0;
    int bs = 0;

    isLast = e_istrm.read();
doing_loop:
    while (!isLast) {
#pragma HLS pipeline II = 1
        isLast = e_istrm.read();
        int offset = bs * _WStrm;
        ap_uint<_WStrm> t = istrm.read();
        tmp.range(offset + _WStrm - 1, offset) = t(_WStrm - 1, 0);
        if (bs == (N - 1)) {
            axi_strm.write(tmp);
            if (nb == (_BurstLen - 1)) {
                nb_strm.write(_BurstLen);
                nb = 0;
            } else
                ++nb;
            bs = 0;
        } else
            ++bs;
    }
    // not enough one axi
    if (bs != 0) {
    doing_not_enough:
        for (; bs < N; ++bs) {
#pragma HLS unroll
            int offset = bs * _WStrm;
            tmp.range(offset + _WStrm - 1, offset) = 0;
        }
        axi_strm.write(tmp);
        ++nb;
    }
    if (nb != 0) {
        XF_UTILS_HW_ASSERT(nb <= _BurstLen);
        nb_strm.write(nb);
    }
    nb_strm.write(0);
}

/**
 * @brief the template of stream width of _WAxi burst out.
 *
 * @tparam _WAxi   width of axi port.
 * @tparam _WStrm  width of input stream.
 * @tparam _BurstLen length of a burst.
 *
 * @param wbuf AXI master port to write to.
 * @param axi_strm stream width is _WAxi
 * @param nb_strm  store burst number of each burst
 */
template <int _WAxi, int _WStrm, int _BurstLen>
void burstWrite(ap_uint<_WAxi>* wbuf, hls::stream<ap_uint<_WAxi> >& axi_strm, hls::stream<ap_uint<8> >& nb_strm) {
    // write each burst to axi
    int total = 0;
    ap_uint<_WAxi> tmp;
    int n = nb_strm.read();
doing_burst:
    while (n) {
    doing_one_burst:
        for (int i = 0; i < n; i++) {
#pragma HLS pipeline II = 1
            tmp = axi_strm.read();
            wbuf[total * _BurstLen + i] = tmp;
        }
        total++;
        n = nb_strm.read();
    }
}
} // details

template <int _BurstLen, int _WAxi, int _WStrm>
void streamToAxi(ap_uint<_WAxi>* wbuf, hls::stream<ap_uint<_WStrm> >& istrm, hls::stream<bool>& e_istrm) {
    XF_UTILS_HW_STATIC_ASSERT(_WAxi % _WStrm == 0, "AXI port width is not multiple of stream width");
    const int fifo_buf = 2 * _BurstLen;

#pragma HLS dataflow

    hls::stream<ap_uint<_WAxi> > axi_strm;
    hls::stream<ap_uint<8> > nb_strm;
#pragma HLS stream variable = nb_strm depth = 2
#pragma HLS stream variable = axi_strm depth = fifo_buf

    details::countForBurst<_WAxi, _WStrm, _BurstLen>(istrm, e_istrm, axi_strm, nb_strm);

    details::burstWrite<_WAxi, _WStrm, _BurstLen>(wbuf, axi_strm, nb_strm);
}

} // utils_hw
} // common
} // xf

#endif // XF_UTILS_HW_STRM_TO_AXI_H
