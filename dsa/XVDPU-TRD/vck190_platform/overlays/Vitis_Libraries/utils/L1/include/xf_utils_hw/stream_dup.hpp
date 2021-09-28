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
#ifndef XF_UTILS_HW_STREAM_DUP_H
#define XF_UTILS_HW_STREAM_DUP_H

#include "xf_utils_hw/types.hpp"
#include "xf_utils_hw/common.hpp"

/**
 * @file stream_dup.hpp
 * @brief duplicate stream and attach end signal to each dup.
 *
 * This file is part of Vitis Utility Library.
 */

// Forward decl ======================================================

namespace xf {
namespace common {
namespace utils_hw {

/**
 * @brief Duplicate stream.
 *
 * @tparam _TIn input stream width.
 * @tparam _NStrm number of output stream.
 *
 * @param istrm input data stream.
 * @param e_istrm end flag stream for input data.
 * @param ostrms output data streams.
 * @param e_ostrms end flag streams, one for each output data stream.
 */
template <typename _TIn, int _NStrm>
void streamDup(hls::stream<_TIn>& istrm,
               hls::stream<bool>& e_istrm,
               hls::stream<_TIn> ostrms[_NStrm],
               hls::stream<bool> e_ostrms[_NStrm]);

/**
 * @brief Duplicate stream.
 *
 * @tparam _TIn input stream width.
 * @tparam _NIStrm number of input stream.
 * @tparam _NDStrm number of streams to be duplicated. Should be smaller than
 * _NIStrm.
 * @tparam _NDCopy number of copies of duplicated streams to be generated
 *
 * @param choose option to choose which input streams will be duplicated and
 * configure the order of the out duplicated streams.
 * @param istrm input data stream.
 * @param e_istrm end flag stream for input data.
 * @param ostrms output data streams.
 * @param dstrms output duplicated streams.
 * @param e_ostrms end flag streams.
 */
template <typename _TIn, int _NIStrm, int _NDStrm, int _NDCopy>
void streamDup(const unsigned int choose[_NDStrm],
               hls::stream<_TIn> istrm[_NIStrm],
               hls::stream<bool>& e_istrm,
               hls::stream<_TIn> ostrms[_NIStrm],
               hls::stream<_TIn> dstrms[_NDCopy][_NDStrm],
               hls::stream<bool>& e_ostrms);

} // utils_hw
} // common
} // xf

// Implementation ====================================================

namespace xf {
namespace common {
namespace utils_hw {

template <typename _TIn, int _NStrm>
void streamDup(hls::stream<_TIn>& istrm,
               hls::stream<bool>& e_istrm,
               hls::stream<_TIn> ostrms[_NStrm],
               hls::stream<bool> e_ostrms[_NStrm]) {
    bool e = e_istrm.read();
    while (!e) {
#pragma HLS pipeline II = 1
        _TIn tmp;
        e = e_istrm.read();
        tmp = istrm.read();
        for (int i = 0; i < _NStrm; i++) {
#pragma HLS unroll
            ostrms[i].write(tmp);
            e_ostrms[i].write(0);
        }
    }
    for (int i = 0; i < _NStrm; i++) {
#pragma HLS unroll
        e_ostrms[i].write(1);
    }
}

template <typename _TIn, int _NIStrm, int _NDStrm, int _NDCopy>
void streamDup(const unsigned int choose[_NDStrm],
               hls::stream<_TIn> istrm[_NIStrm],
               hls::stream<bool>& e_istrm,
               hls::stream<_TIn> ostrms[_NIStrm],
               hls::stream<_TIn> dstrms[_NDCopy][_NDStrm],
               hls::stream<bool>& e_ostrms) {
    XF_UTILS_HW_STATIC_ASSERT(_NDStrm <= _NIStrm, "stream_dup cannot have more duplicated output than input.");

    bool e = e_istrm.read();
    int i;
    while (!e) {
#pragma HLS PIPELINE II = 1

        _TIn tmp[_NIStrm];
        for (i = 0; i < _NIStrm; i++) {
#pragma HLS UNROLL
            tmp[i] = istrm[i].read();
        }

        for (i = 0; i < _NDStrm; i++) {
#pragma HLS UNROLL
            for (int j = 0; j < _NDCopy; j++) {
#pragma HLS UNROLL
                dstrms[j][i].write(tmp[choose[i]]);
            }
        }

        for (i = 0; i < _NIStrm; i++) {
#pragma HLS UNROLL
            ostrms[i].write(tmp[i]);
        }
        e = e_istrm.read();
        e_ostrms.write(false);
    }
    e_ostrms.write(true);
}

} // utils_hw
} // common
} // xf

#endif // XF_UTILS_HW_STREAM_DUP_H
