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

#ifndef _XF_INRANGE_HPP_
#define _XF_INRANGE_HPP_

#ifndef __cplusplus
#error C++ is needed to include this header
#endif

typedef unsigned short uint16_t;
typedef unsigned char uchar;

#include "common/xf_common.hpp"
#include "common/xf_utility.hpp"
#include "hls_stream.h"

namespace xf {
namespace cv {

template <int WORDWIDTH_SRC, int WORDWIDTH_DST, int DEPTH_SRC, int DEPTH_DST, int SRC_T, int NPC>
void inrangeproc(XF_SNAME(WORDWIDTH_SRC) & val_src,
                 XF_SNAME(WORDWIDTH_DST) & tmp_val,
                 XF_PTNAME(DEPTH_DST) channel_out[XF_CHANNELS(SRC_T, NPC)],
                 XF_PTNAME(DEPTH_DST) _lower_thresh[XF_CHANNELS(SRC_T, NPC)],
                 XF_PTNAME(DEPTH_DST) _upper_thresh[XF_CHANNELS(SRC_T, NPC)]) {
    XF_PTNAME(DEPTH_SRC) p;

    ap_uint<8> tmp_val1 = 0;

    for (ap_uint<13> k = 0; k < (1 << XF_BITSHIFT(NPC)); k++) {
// clang-format off
    #pragma HLS unroll
        // clang-format on
        p = val_src.range(k * XF_PIXELDEPTH(DEPTH_SRC) + (XF_PIXELDEPTH(DEPTH_SRC) - 1), k * XF_PIXELDEPTH(DEPTH_SRC));

        for (ap_uint<13> ch = 0, idx = 0; ch < XF_CHANNELS(SRC_T, NPC); ch++, idx += XF_DTPIXELDEPTH(SRC_T, NPC)) {
// clang-format off
      #pragma HLS unroll
            // clang-format on
            tmp_val1 = p.range(idx + (XF_DTPIXELDEPTH(SRC_T, NPC) - 1), idx);
            channel_out[ch] =
                ((tmp_val1 >= _lower_thresh[ch]) && (tmp_val1 <= _upper_thresh[ch])) ? (ap_uint<8>)255 : (ap_uint<8>)0;
        }
        if (XF_CHANNELS(SRC_T, NPC) != 1) {
            tmp_val.range(k * XF_PIXELDEPTH(DEPTH_DST) + (XF_PIXELDEPTH(DEPTH_DST) - 1), k * XF_PIXELDEPTH(DEPTH_DST)) =
                (channel_out[0] & channel_out[1] & channel_out[2]);
        } else {
            tmp_val.range(k * XF_PIXELDEPTH(DEPTH_DST) + (XF_PIXELDEPTH(DEPTH_DST) - 1), k * XF_PIXELDEPTH(DEPTH_DST)) =
                channel_out[0];
        }
    }
}

/**
 * xFinRangeKernel: Thresholds an input image and produces an output boolean
 * image depending upon the type of thresholding.
 * Input   : _src_mat, lower_thresh and upper_thresh
 * Output  : _dst_mat
 **/
template <int SRC_T,
          int DST_T,
          int ROWS,
          int COLS,
          int DEPTH_SRC,
          int DEPTH_DST,
          int NPC,
          int WORDWIDTH_SRC,
          int WORDWIDTH_DST,
          int COLS_TRIP>
void xFinRangeKernel(xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _src_mat,
                     xf::cv::Mat<DST_T, ROWS, COLS, NPC>& _dst_mat,
                     unsigned char lower_thresh[XF_CHANNELS(SRC_T, NPC)],
                     unsigned char upper_thresh[XF_CHANNELS(SRC_T, NPC)],
                     unsigned short height,
                     unsigned short width) {
    XF_SNAME(WORDWIDTH_SRC) val_src;
    XF_SNAME(WORDWIDTH_DST) val_dst, tmp_val;
    //,out;
    XF_PTNAME(DEPTH_DST)
    _lower_thresh[XF_CHANNELS(SRC_T, NPC)]; //=(XF_PTNAME(DEPTH))lower_thresh;
    XF_PTNAME(DEPTH_DST)
    _upper_thresh[XF_CHANNELS(SRC_T, NPC)]; //=(XF_PTNAME(DEPTH))upper_thresh;
    XF_PTNAME(DEPTH_DST) channel_out[XF_CHANNELS(SRC_T, NPC)];

    for (int i = 0; i < XF_CHANNELS(SRC_T, NPC); i++) {
        _lower_thresh[i] = (XF_PTNAME(DEPTH_SRC))lower_thresh[i];
        _upper_thresh[i] = (XF_PTNAME(DEPTH_SRC))upper_thresh[i];
    }
    ap_uint<13> i, j, k, c;
rowLoop:
    for (i = 0; i < height; i++) {
// clang-format off
    #pragma HLS LOOP_TRIPCOUNT min=ROWS max=ROWS
    #pragma HLS LOOP_FLATTEN off
    // clang-format on

    colLoop:
        for (j = 0; j < width; j++) {
// clang-format off
      #pragma HLS LOOP_TRIPCOUNT min=COLS_TRIP max=COLS_TRIP
      #pragma HLS pipeline
            // clang-format on

            val_src = (XF_SNAME(WORDWIDTH_SRC))(_src_mat.read(i * width + j));

            inrangeproc<WORDWIDTH_SRC, WORDWIDTH_DST, DEPTH_SRC, DEPTH_DST, SRC_T, NPC>(val_src, tmp_val, channel_out,
                                                                                        _lower_thresh, _upper_thresh);

            _dst_mat.write(i * width + j, tmp_val); // writing the val_dst into output stream _dst
        }
    }
}

template <int SRC_T, int DST_T, int ROWS, int COLS, int NPC = 1>
void inRange(xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& src,
             unsigned char lower_thresh[XF_CHANNELS(SRC_T, NPC)],
             unsigned char upper_thresh[XF_CHANNELS(SRC_T, NPC)],
             xf::cv::Mat<DST_T, ROWS, COLS, NPC>& dst) {
    unsigned short width = src.cols >> XF_BITSHIFT(NPC);
    unsigned short height = src.rows;

#ifndef __SYNTHESIS__
    assert(((SRC_T == XF_8UC1) || (SRC_T == XF_8UC3)) && "Type must be XF_8UC1 or XF_8UC3");
    assert(((NPC == XF_NPPC1) || (NPC == XF_NPPC8)) && "NPC must be XF_NPPC1, XF_NPPC8");

    assert(((lower_thresh[0] >= 0) && (lower_thresh[0] <= 255)) && "lower_thresh must be with the range of 0 to 255");

    assert(((upper_thresh[0] >= 0) && (upper_thresh[0] <= 255)) && "lower_thresh must be with the range of 0 to 255");

    assert(((height <= ROWS) && (width <= COLS)) && "ROWS and COLS should be greater than input image");
#endif
// clang-format off
  #pragma HLS INLINE OFF
    // clang-format on

    xFinRangeKernel<SRC_T, DST_T, ROWS, COLS, XF_DEPTH(SRC_T, NPC), XF_DEPTH(DST_T, NPC), NPC, XF_WORDWIDTH(SRC_T, NPC),
                    XF_WORDWIDTH(DST_T, NPC), (COLS >> XF_BITSHIFT(NPC))>(src, dst, lower_thresh, upper_thresh, height,
                                                                          width);
}
} // namespace cv
} // namespace xf

#endif
