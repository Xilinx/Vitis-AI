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
#ifndef XF_UTILS_HW_STREAM_SHUFFLE_H
#define XF_UTILS_HW_STREAM_SHUFFLE_H

#include "xf_utils_hw/types.hpp"
#include "xf_utils_hw/common.hpp"

/**
 * @file stream_shuffle.hpp
 * @brief Unidirectional cross-bar of streams.
 *
 * This file is part of Vitis Utility Library.
 */

// Forward decl ===============================================================

namespace xf {
namespace common {
namespace utils_hw {

/**
 * @brief Shuffle the contents from an array of streams to another.
 *
 * Suppose we have an array of 3 streams for R-G-B channels correspondingly,
 * and it is needed to shuffle B to Stream 0, R to Stream 1 and G to Stream 2.
 * This module can bridge this case with the configuration ``2, 0, 1``.
 * Here, ``2`` is the source index for data B at destination index ``0``,
 * and ``0`` is the source index for data R at destination index ``1``,
 * and ``1`` is the source index for data G at destination index ``2``.
 *
 * The configuration is load once in one invocation, and reused until the end.
 * This module supports up to 128 input streams, and works efficiently within
 * 16.
 *
 * If minus value is used as the source index, the corresponding stream will be
 * filled with zero.
 *
 * If a source index is specified twice, the behavior is undefined.
 *
 * @tparam _INStrm number of input  stream.
 * @tparam _ONstrm number of output stream.
 * @tparam _TIn input type.
 *
 * @param order_cfg the new order within the window. Each 8bit specifies the
 * source stream for the corresponding output stream, starting from the stream
 * with new order 0.
 * @param istrms input data streams.
 * @param e_istrm end flags for input.
 * @param ostrms output data streams.
 * @param e_ostrm end flag for output.
 */
template <int _INStrm, int _ONstrm, typename _TIn>
void streamShuffle(hls::stream<ap_uint<8 * _ONstrm> >& order_cfg,

                   hls::stream<_TIn> istrms[_INStrm],
                   hls::stream<bool>& e_istrm,

                   hls::stream<_TIn> ostrms[_ONstrm],
                   hls::stream<bool>& e_ostrm);

} // utils_hw
} // common
} // xf

// Implementation =============================================================
namespace xf {
namespace common {
namespace utils_hw {

template <int _INStrm, int _ONstrm, typename _TIn>
void streamShuffle(hls::stream<ap_uint<8 * _ONstrm> >& order_cfg,

                   hls::stream<_TIn> istrms[_INStrm],
                   hls::stream<bool>& e_istrm,

                   hls::stream<_TIn> ostrms[_ONstrm],
                   hls::stream<bool>& e_ostrm) {
    XF_UTILS_HW_STATIC_ASSERT(_INStrm <= 128, "stream_shuffle cannot handle more than 128 streams.");

    ap_int<8> route[_ONstrm];
#pragma HLS ARRAY_PARTITION variable = route complete

    _TIn reg_i[_INStrm];
#pragma HLS ARRAY_PARTITION variable = reg_i complete
    _TIn reg_o[_ONstrm];
#pragma HLS ARRAY_PARTITION variable = reg_o complete

    ap_uint<8 * _ONstrm> orders = order_cfg.read();
    for (int i = 0; i < _ONstrm; i++) {
#pragma HLS UNROLL
        route[i] = orders.range(8 * i + 7, 8 * i);
    }

    bool e = e_istrm.read();
    while (!e) {
#pragma HLS PIPELINE II = 1

        for (int i = 0; i < _INStrm; i++) {
#pragma HLS UNROLL
            reg_i[i] = istrms[i].read();
        }
        e = e_istrm.read();

        for (int i = 0; i < _ONstrm; i++) {
#pragma HLS UNROLL
            reg_o[i] = 0;
        }

        for (int i = 0; i < _ONstrm; i++) {
#pragma HLS UNROLL
            if (!route[i][7]) reg_o[i] = reg_i[route[i]]; // critical path
        }

        for (int i = 0; i < _ONstrm; i++) {
#pragma HLS UNROLL
            ostrms[i].write(reg_o[i]);
        }

        e_ostrm.write(false);
    }
    e_ostrm.write(true);
}

} // utils_hw
} // common
} // xf

#endif // XF_UTILS_HW_STREAM_SHUFFLE_H
