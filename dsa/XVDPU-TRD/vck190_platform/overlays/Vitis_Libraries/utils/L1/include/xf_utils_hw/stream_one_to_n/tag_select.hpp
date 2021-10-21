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
#ifndef XF_UTILS_HW_STREAM_1N_TAG_H
#define XF_UTILS_HW_STREAM_1N_TAG_H

/**
 * @file stream_one_to_n/tag_select.hpp
 * @brief header for distribute in round-roubin order.
 */

#include "xf_utils_hw/types.hpp"
#include "xf_utils_hw/enums.hpp"
#include "xf_utils_hw/common.hpp"
// Forward decl

namespace xf {
namespace common {
namespace utils_hw {

/**
 * @brief This function send element from one stream to multiple streams based
 * on tags.
 *
 * In this primitive, the tag is the index of ouput streams.
 * The input data in data_istrms is distributed to  the data_ostrm whose index
 * is tag. Each tag is the index of output streams,
 * and data_istrm and tag_istrm are synchronous.
 *
 * @tparam _WInStrm  the width of input data
 * @tparam _WTagStrm the width of tag,  pow(2, _WTagStrm) is the number of ouput
 * streams.
 *
 * @param data_istrm the input stream.
 * @param e_data_istrm the end signal of input stream.
 * @param tag_istrm  the  tag stream.
 * @param e_tag_istrm the end signal stream of tags.
 * @param data_ostrms the output stream.
 * @param e_data_ostrms the end signals of data_ostrms.
 * @param alg   algorithm selector
 */
template <int _WInStrm, int _WTagStrm>
void streamOneToN(hls::stream<ap_uint<_WInStrm> >& data_istrm,
                  hls::stream<bool>& e_data_istrm,
                  hls::stream<ap_uint<_WTagStrm> >& tag_istrm,
                  hls::stream<bool>& e_tag_istrm,
                  hls::stream<ap_uint<_WInStrm> > data_ostrms[PowerOf2<_WTagStrm>::value],
                  hls::stream<bool> e_data_ostrms[PowerOf2<_WTagStrm>::value],
                  TagSelectT alg);

/**
 * @brief This function send element from one stream to multiple streams based
 * on tags.
 *
 * In this primitive, the tag is the index of ouput streams.
 * The input data in data_istrms is distributed to  the data_ostrm whose index is
 * tag. Each tag is the index of output streams,
 * and data_istrm and tag_istrm are synchronous.
 *
 * @tparam _TIn  the type of input & output data.
 * @tparam _WTagStrm the width of tag,  pow(2, _WTagStrm) is the number of ouput
 * streams.
 *
 * @param data_istrm the input stream.
 * @param e_data_istrm the end signal of input stream.
 * @param tag_istrm  the tag stream.
 * @param e_tag_istrm the end signal of tag stream.
 * @param data_ostrms the output stream.
 * @param e_data_ostrms the end signals of data_ostrms.
 * @param alg   algorithm selector.
 */
template <typename _TIn, int _WTagStrm>
void streamOneToN(hls::stream<_TIn>& data_istrm,
                  hls::stream<bool>& e_data_istrm,
                  hls::stream<ap_uint<_WTagStrm> >& tag_istrm,
                  hls::stream<bool>& e_tag_istrm,
                  hls::stream<_TIn> data_ostrms[PowerOf2<_WTagStrm>::value],
                  hls::stream<bool> e_data_ostrms[PowerOf2<_WTagStrm>::value],
                  TagSelectT alg);

} // utils_hw
} // common
} // xf

////////////////////////////////////////////////////////////////////////////

// Implementation

namespace xf {
namespace common {
namespace utils_hw {
namespace details {

template <int _WInStrm, int _WTagStrm>
void stream_one_to_n_tag_select(hls::stream<ap_uint<_WInStrm> >& data_istrm,
                                hls::stream<bool>& e_data_istrm,
                                hls::stream<ap_uint<_WTagStrm> >& tag_istrm,
                                hls::stream<bool>& e_tag_istrm,
                                hls::stream<ap_uint<_WInStrm> > data_ostrms[PowerOf2<_WTagStrm>::value],
                                hls::stream<bool> e_data_ostrms[PowerOf2<_WTagStrm>::value]) {
    bool last_tag = e_tag_istrm.read();
    bool last_istrm = e_data_istrm.read();
    while (!last_tag && !last_istrm) {
#pragma HLS pipeline II = 1
        ap_uint<_WInStrm> data = data_istrm.read();
        ap_uint<_WTagStrm> tag = tag_istrm.read();
        XF_UTILS_HW_ASSERT(tag >= 0);
        XF_UTILS_HW_ASSERT(tag < PowerOf2<_WTagStrm>::value);
        data_ostrms[tag].write(data);
        e_data_ostrms[tag].write(false);
        last_tag = e_tag_istrm.read();
        last_istrm = e_data_istrm.read();
    }
    // drop
    while (!last_istrm) {
        last_istrm = e_data_istrm.read();
        ap_uint<_WInStrm> data = data_istrm.read();
    }

    XF_UTILS_HW_ASSERT(last_tag);

    const unsigned int nstrm = PowerOf2<_WTagStrm>::value;
    for (unsigned int i = 0; i < nstrm; ++i) {
#pragma HLS unroll
        e_data_ostrms[i].write(true);
    }
}

} // details

template <int _WInStrm, int _WTagStrm>
void streamOneToN(hls::stream<ap_uint<_WInStrm> >& data_istrm,
                  hls::stream<bool>& e_data_istrm,
                  hls::stream<ap_uint<_WTagStrm> >& tag_istrm,
                  hls::stream<bool>& e_tag_istrm,
                  hls::stream<ap_uint<_WInStrm> > data_ostrms[PowerOf2<_WTagStrm>::value],
                  hls::stream<bool> e_data_ostrms[PowerOf2<_WTagStrm>::value],
                  TagSelectT alg) {
    details::stream_one_to_n_tag_select<_WInStrm, _WTagStrm>(data_istrm, e_data_istrm, tag_istrm, e_tag_istrm,
                                                             data_ostrms, e_data_ostrms);
}
//--------------------------------------------------------------------//

namespace details {

template <typename _TIn, int _WTagStrm>
void stream_one_to_n_tag_select_type(hls::stream<_TIn>& data_istrm,
                                     hls::stream<bool>& e_data_istrm,
                                     hls::stream<ap_uint<_WTagStrm> >& tag_istrm,
                                     hls::stream<bool>& e_tag_istrm,
                                     hls::stream<_TIn> data_ostrms[PowerOf2<_WTagStrm>::value],
                                     hls::stream<bool> e_data_ostrms[PowerOf2<_WTagStrm>::value]) {
    bool last_tag = e_tag_istrm.read();
    bool last_istrm = e_data_istrm.read();
    while (!last_tag && !last_istrm) {
#pragma HLS pipeline II = 1
        _TIn data = data_istrm.read();
        ap_uint<_WTagStrm> tag = tag_istrm.read();
        XF_UTILS_HW_ASSERT(tag >= 0);
        XF_UTILS_HW_ASSERT(tag < PowerOf2<_WTagStrm>::value);
        data_ostrms[tag].write(data);
        e_data_ostrms[tag].write(false);
        last_tag = e_tag_istrm.read();
        last_istrm = e_data_istrm.read();
    }
    // drop
    while (!last_istrm) {
        last_istrm = e_data_istrm.read();
        _TIn data = data_istrm.read();
    }

    XF_UTILS_HW_ASSERT(last_tag);

    const unsigned int nstrm = PowerOf2<_WTagStrm>::value;
    for (unsigned int i = 0; i < nstrm; ++i) {
#pragma HLS unroll
        e_data_ostrms[i].write(true);
    }
}

} // details

template <typename _TIn, int _WTagStrm>
void streamOneToN(hls::stream<_TIn>& data_istrm,
                  hls::stream<bool>& e_data_istrm,
                  hls::stream<ap_uint<_WTagStrm> >& tag_istrm,
                  hls::stream<bool>& e_tag_istrm,
                  hls::stream<_TIn> data_ostrms[PowerOf2<_WTagStrm>::value],
                  hls::stream<bool> e_data_ostrms[PowerOf2<_WTagStrm>::value],
                  TagSelectT alg) {
    details::stream_one_to_n_tag_select_type<_TIn, _WTagStrm>(data_istrm, e_data_istrm, tag_istrm, e_tag_istrm,
                                                              data_ostrms, e_data_ostrms);
}

} // utils_hw
} // common
} // xf

#endif // XF_UTILS_HW_STREAM_1N_TAG_H
