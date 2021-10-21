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
#ifndef XF_UTILS_HW_STREAM_SPLIT_H
#define XF_UTILS_HW_STREAM_SPLIT_H

#include "xf_utils_hw/types.hpp"
#include "xf_utils_hw/enums.hpp"

/**
 * @file stream_split.hpp
 * @brief split one streams into multiple narrow ones.
 *
 * This file is part of Vitis Utility Library.
 */

// Forward decl ======================================================

namespace xf {
namespace common {
namespace utils_hw {

/**
 * @brief split one wide stream into multiple streams, start from the LSB.
 *
 * If ``_WIn > _WOut * _NStrm``, the extra bits will be discarded.
 *
 * @tparam _WIn input stream width, should be no less than _WOut * _NStrm.
 * @tparam _WOut output stream width.
 * @tparam _NStrm number of output stream.
 *
 * @param istrm input data stream.
 * @param e_istrm end flag for the input.
 * @param ostrms output data streams.
 * @param e_ostrm end flag streams for all outputs.
 * @param alg algorithm selector for this function.
 */
template <int _WIn, int _WOut, int _NStrm>
void streamSplit(hls::stream<ap_uint<_WIn> >& istrm,
                 hls::stream<bool>& e_istrm,
                 hls::stream<ap_uint<_WOut> > ostrms[_NStrm],
                 hls::stream<bool>& e_ostrm,
                 LSBSideT alg);

/**
 * @brief split one wide stream into multiple streams, start from the MSB.
 *
 * If ``_WIn > _WOut * _NStrm``, the extra bits will be discarded.
 *
 * @tparam _WIn input stream width, should be no less than _WOut * _NStrm.
 * @tparam _WOut output stream width.
 * @tparam _NStrm number of output stream.
 *
 * @param istrm input data stream.
 * @param e_istrm end flag for the input.
 * @param ostrms output data streams.
 * @param e_ostrm end flag streams for all outputs.
 * @param alg algorithm selector for this function.
 */

template <int _WIn, int _WOut, int _NStrm>
void streamSplit(hls::stream<ap_uint<_WIn> >& istrm,
                 hls::stream<bool>& e_istrm,
                 hls::stream<ap_uint<_WOut> > ostrms[_NStrm],
                 hls::stream<bool>& e_ostrm,
                 MSBSideT alg);

} // utils_hw
} // common
} // xf

// Implementation ====================================================

namespace xf {
namespace common {
namespace utils_hw {

template <int _WIn, int _WOut, int _NStrm>
void streamSplit(hls::stream<ap_uint<_WIn> >& istrm,
                 hls::stream<bool>& e_istrm,
                 hls::stream<ap_uint<_WOut> > ostrms[_NStrm],
                 hls::stream<bool>& e_ostrm,
                 LSBSideT alg) {
    /*
     * for example, _WIn=20, _WOut=4, _NStrm=4
     * input a data  0x82356 (hex)
     * split to 4 streams:
     *              lsb                msb
     *         ostrms[0] =  0x6    ostrms[0] =  0x8
     *         ostrms[1] =  0x5    ostrms[1] =  0x2
     *         ostrms[2] =  0x3    ostrms[2] =  0x3
     *         ostrms[3] =  0x2    ostrms[3] =  0x5
     * discard   highest =  0x8      lowest  =  0x6
     *
     * this primitive implement split based on lsb.
     */
    const int max = _WIn > _WOut * _NStrm ? _WIn : _WOut * _NStrm;
    bool last = e_istrm.read();
    while (!last) {
#pragma HLS pipeline II = 1
        last = e_istrm.read();
        ap_uint<max> data = istrm.read();
        // ap_uint<_WIn>  data = istrm.read();  // out of the range if _WIn<
        // _WOut*_NStrm
        for (int i = 0; i < _NStrm; ++i) {
#pragma HLS unroll
            ap_uint<_WOut> d = data.range((i + 1) * _WOut - 1, i * _WOut);
            ostrms[i].write(d);
        } // for
        e_ostrm.write(false);
    } // while
    e_ostrm.write(true);
}

template <int _WIn, int _WOut, int _NStrm>
void streamSplit(hls::stream<ap_uint<_WIn> >& istrm,
                 hls::stream<bool>& e_istrm,
                 hls::stream<ap_uint<_WOut> > ostrms[_NStrm],
                 hls::stream<bool>& e_ostrm,
                 MSBSideT alg) {
    /*
     * for example, _WIn=20, _WOut=4, _NStrm=4
     * input a data  0x82356 (hex)
     * split to 4 streams:
     *              lsb                msb
     *         ostrms[0] =  0x6    ostrms[0] =  0x8
     *         ostrms[1] =  0x5    ostrms[1] =  0x2
     *         ostrms[2] =  0x3    ostrms[2] =  0x3
     *         ostrms[3] =  0x2    ostrms[3] =  0x5
     * discard   highest =  0x8      lowest  =  0x6
     *
     * this primitive implement split based on msb.
     */
    const int nout = _WOut * _NStrm;
    const int max = _WIn > nout ? _WIn : nout;
    const int df = max - nout;
    bool last = e_istrm.read();
    while (!last) {
#pragma HLS pipeline II = 1
        last = e_istrm.read();
        // ap_uint<_WIn>  data = istrm.read(); // out of the range if _WIn <
        // _WOut*_NStrm
        ap_uint<max> data = istrm.read();
        ap_uint<nout> nd = _WIn >= nout ? (data >> df) : (data << df); // keep MSB
        for (int i = 0, j = _NStrm - 1; i < _NStrm; ++i, --j) {
#pragma HLS unroll
            ap_uint<_WOut> d = nd.range((i + 1) * _WOut - 1, i * _WOut);
            ostrms[j].write(d);
        } // for
        e_ostrm.write(false);
    } // while
    e_ostrm.write(true);
}

} // utils_hw
} // common
} // xf

#endif // XF_UTILS_HW_STREAM_SPLIT_H
