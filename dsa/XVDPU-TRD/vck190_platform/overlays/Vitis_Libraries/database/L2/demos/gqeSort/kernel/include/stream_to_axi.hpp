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

#include "ap_int.h"
#include "hls_stream.h"

namespace xf {
namespace common {
namespace utils_hw {
namespace details {

/**
 * @brief the template of convert stream width from _WStrm to _WAxi and count
 * burst number.
 *
 * @tparam _BurstLen length of a burst.
 * @tparam _WAxi   width of axi port.
 *
 * @param istrm   input stream.
 * @param axi_strm stream width is _WAxi
 * @param nb_strm  store burst number of each burst
 */
template <int _BurstLen, typename _Td, typename _Tk, int _WAxi>
void countForBurst(hls::stream<xf::database::details::Pair<_Td, _Tk> >& istrm,
                   hls::stream<ap_uint<_WAxi> >& axi_strm,
                   hls::stream<ap_uint<8> >& nb_strm) {
    enum { _Wk = sizeof(_Tk) * 8, _Wd = sizeof(_Td) * 8, _WStrm = _Wk + _Wd, N = _WAxi / _WStrm };
    ap_uint<_WAxi> tmp;
    int nb = 0;
    int bs = 0;

    typedef xf::database::details::Pair<_Td, _Tk> _Tp;
    _Tp dp = istrm.read();
    bool isLast = dp.end();
doing_loop:
    while (!isLast) {
#pragma HLS pipeline II = 1
        int offset = bs * _WStrm;
        ap_uint<_WStrm> t;
        t.range(_Wk - 1, 0) = dp.key();
        t.range(_Wk + _Wd - 1, _Wk) = dp.data();
        tmp.range(offset + _WStrm - 1, offset) = t(_WStrm - 1, 0);
        // fetch & decide
        dp = istrm.read();
        isLast = dp.end();
        //
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
        nb_strm.write(nb);
    }
    nb_strm.write(0);
}

/**
 * @brief the template of stream width of _WAxi burst out.
 *
 * @tparam _BurstLen length of a burst.
 * @tparam _WAxi   width of axi port.
 *
 * @param wbuf AXI master port to write to.
 * @param axi_strm stream width is _WAxi
 * @param nb_strm  store burst number of each burst
 */
template <int _BurstLen, int _WAxi>
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
/**
 * @brief the template of stream width of _WAxi burst out.
 *
 * @tparam _BurstLen length of a burst.
 * @tparam _WAxi   width of axi port.
 *
 * @param wbuf AXI master port to write to.
 * @param wbufoff start offset of AXI master port to write to .
 * @param axi_strm stream width is _WAxi
 * @param nb_strm  store burst number of each burst
 */
template <int _BurstLen, int _WAxi>
void burstWrite(ap_uint<_WAxi>* wbuf,
                const unsigned int wbufoff,
                hls::stream<ap_uint<_WAxi> >& axi_strm,
                hls::stream<ap_uint<8> >& nb_strm) {
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
            wbuf[total * _BurstLen + i + wbufoff] = tmp;
        }
        total++;
        n = nb_strm.read();
    }
}

/**
 * @brief the template of stream width of _WAxi burst out.
 *
 * @tparam _BurstLen length of a burst.
 * @tparam _WAxi   width of axi port.
 *
 * @param kbuf AXI master port to write keys to.
 * @param dbuf AXI master port to write data to.
 * @param axi_strm stream width is _WAxi
 * @param nb_strm  store burst number of each burst
 */
template <int _BurstLen, int _WAxi>
void burstWrite(ap_uint<_WAxi / 2>* kbuf,
                ap_uint<_WAxi / 2>* dbuf,
                hls::stream<ap_uint<_WAxi> >& axi_strm,
                hls::stream<ap_uint<8> >& nb_strm) {
    // write each burst to axi
    int total = 0;
    ap_uint<_WAxi> tmp;
    ap_uint<_WAxi / 2> biKey;
    ap_uint<_WAxi / 2> biData;
    int n = nb_strm.read();
doing_burst:
    while (n) {
    doing_one_burst:
        for (int i = 0; i < n; i++) {
#pragma HLS pipeline II = 1
            tmp = axi_strm.read();
            biKey.range(_WAxi / 4 - 1, 0) = tmp.range(_WAxi / 4 - 1, 0);
            biKey.range(_WAxi / 2 - 1, _WAxi / 4) = tmp.range(_WAxi * 3 / 4 - 1, _WAxi / 2);
            biData.range(_WAxi / 4 - 1, 0) = tmp.range(_WAxi / 2 - 1, _WAxi / 4);
            biData.range(_WAxi / 2 - 1, _WAxi / 4) = tmp.range(_WAxi - 1, _WAxi * 3 / 4);
            kbuf[total * _BurstLen + i] = biKey;
            dbuf[total * _BurstLen + i] = biData;
        }
        total++;
        n = nb_strm.read();
    }
}

} // details

template <int _BurstLen, int _WAxi, typename _Td, typename _Tk>
void streamToAxi(hls::stream<xf::database::details::Pair<_Td, _Tk> >& istrm, ap_uint<_WAxi>* wbuf) {
    enum { fifo_buf = 2 * _BurstLen };

#pragma HLS dataflow

    hls::stream<ap_uint<_WAxi> > axi_strm;
    hls::stream<ap_uint<8> > nb_strm;
#pragma HLS stream variable = nb_strm depth = 2
#pragma HLS stream variable = axi_strm depth = fifo_buf

    details::countForBurst<_BurstLen>(istrm, axi_strm, nb_strm);

    details::burstWrite<_BurstLen>(wbuf, axi_strm, nb_strm);
}

template <int _BurstLen, int _WAxi, typename _Td, typename _Tk>
void streamToAxi(hls::stream<xf::database::details::Pair<_Td, _Tk> >& istrm,
                 const unsigned int wbufoff,
                 ap_uint<_WAxi>* wbuf) {
    enum { fifo_buf = 2 * _BurstLen };

#pragma HLS dataflow

    hls::stream<ap_uint<_WAxi> > axi_strm;
    hls::stream<ap_uint<8> > nb_strm;
#pragma HLS stream variable = nb_strm depth = 2
#pragma HLS stream variable = axi_strm depth = fifo_buf

    details::countForBurst<_BurstLen>(istrm, axi_strm, nb_strm);

    details::burstWrite<_BurstLen>(wbuf, wbufoff, axi_strm, nb_strm);
}

template <int _BurstLen, int _WAxi, typename _Td, typename _Tk>
void streamToAxi(hls::stream<xf::database::details::Pair<_Td, _Tk> >& istrm,
                 ap_uint<_WAxi / 2>* kbuf,
                 ap_uint<_WAxi / 2>* dbuf) {
    enum { fifo_buf = 2 * _BurstLen };

#pragma HLS dataflow

    hls::stream<ap_uint<_WAxi> > axi_strm;
    hls::stream<ap_uint<8> > nb_strm;
#pragma HLS stream variable = nb_strm depth = 2
#pragma HLS stream variable = axi_strm depth = fifo_buf

    details::countForBurst<_BurstLen>(istrm, axi_strm, nb_strm);

    details::burstWrite<_BurstLen>(kbuf, dbuf, axi_strm, nb_strm);
}

} // utils_hw
} // common
} // xf

#endif // XF_UTILS_HW_STRM_TO_AXI_H
