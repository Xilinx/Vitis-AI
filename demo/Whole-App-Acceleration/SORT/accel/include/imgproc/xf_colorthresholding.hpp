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

#ifndef _XF_COLORTHRESHOLDING_HPP_
#define _XF_COLORTHRESHOLDING_HPP_

#include "common/xf_common.hpp"
#include "common/xf_utility.hpp"
#include "hls_stream.h"
#include "imgproc/xf_inrange.hpp"

typedef unsigned short uint16_t;

typedef unsigned int uint32_t;

namespace xf {
namespace cv {

template <int MAXCOLOR>
void apply_threshold(unsigned char low_thresh[MAXCOLOR][3],
                     unsigned char high_thresh[MAXCOLOR][3],
                     ap_uint<8>& outpix,
                     ap_uint<8>& h,
                     ap_uint<8>& s,
                     ap_uint<8>& v) {
// clang-format off
  #pragma HLS inline off
    // clang-format on

    ap_uint<8> tmp_val = 0;

    ap_uint<8> tmp_val1 = 0;

    for (int k = 0; k < MAXCOLOR; k++) {
        ap_uint<8> t1, t2, t3;
        t1 = 0;
        t2 = 0;
        t3 = 0;

        if ((low_thresh[k][0] <= h) && (h <= high_thresh[k][0])) t1 = 255;
        if ((low_thresh[k][1] <= s) && (s <= high_thresh[k][1])) t2 = 255;
        if ((low_thresh[k][2] <= v) && (v <= high_thresh[k][2])) t3 = 255;

        tmp_val = tmp_val | (t1 & t2 & t3);
    }

    outpix = tmp_val;
}

template <int SRC_T,
          int DST_T,
          int ROWS,
          int COLS,
          int DEPTH_SRC,
          int DEPTH_DST,
          int NPC,
          int WORDWIDTH_SRC,
          int WORDWIDTH_DST,
          int MAXCOLOR>
void xFInRange(xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _src_mat,
               xf::cv::Mat<DST_T, ROWS, COLS, NPC>& _dst_mat,
               unsigned char low_thresh[MAXCOLOR][3],
               unsigned char high_thresh[MAXCOLOR][3],
               uint16_t img_height,
               uint16_t img_width) {
    XF_PTNAME(DEPTH_SRC) in_pix;
    XF_PTNAME(DEPTH_DST) out_pix;
    ap_uint<8> h, s, v;

    XF_SNAME(WORDWIDTH_SRC) val_src;
    XF_PTNAME(DEPTH_DST) channel_out[XF_CHANNELS(SRC_T, NPC)];
    XF_SNAME(WORDWIDTH_DST) val_dst;

    XF_PTNAME(DEPTH_DST)
    _lower_thresh[XF_CHANNELS(SRC_T, NPC)]; //=(XF_PTNAME(DEPTH))lower_thresh;
    XF_PTNAME(DEPTH_DST)
    _upper_thresh[XF_CHANNELS(SRC_T, NPC)]; //=(XF_PTNAME(DEPTH))upper_thresh;

    for (uint16_t row = 0; row < img_height; row++) {
// clang-format off
    #pragma HLS LOOP_TRIPCOUNT max = ROWS
        // clang-format on
        for (uint16_t col = 0; col < img_width; col++) {
// clang-format off
      #pragma HLS PIPELINE
      #pragma HLS LOOP_TRIPCOUNT max = COLS
            // clang-format on
            XF_SNAME(WORDWIDTH_DST) tempval = 0;

            val_src = _src_mat.read(row * img_width + col);
            XF_SNAME(WORDWIDTH_DST) tmp_val = 0;

            for (int k = 0; k < MAXCOLOR; k++) {
                for (int i = 0; i < XF_CHANNELS(SRC_T, NPC); i++) {
                    _lower_thresh[i] = (XF_PTNAME(DEPTH_DST))low_thresh[k][i];
                    _upper_thresh[i] = (XF_PTNAME(DEPTH_DST))high_thresh[k][i];
                }
                inrangeproc<WORDWIDTH_SRC, WORDWIDTH_DST, DEPTH_SRC, DEPTH_DST, SRC_T, NPC>(
                    val_src, tmp_val, channel_out, _lower_thresh, _upper_thresh);

                tempval = tempval | tmp_val;
            }

            _dst_mat.write(row * img_width + col, tempval);
        }
    }
}

template <int SRC_T, int DST_T, int MAXCOLORS, int ROWS, int COLS, int NPC>
void colorthresholding(xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _src_mat,
                       xf::cv::Mat<DST_T, ROWS, COLS, NPC>& _dst_mat,
                       unsigned char low_thresh[MAXCOLORS * 3],
                       unsigned char high_thresh[MAXCOLORS * 3]) {
// clang-format off
  #pragma HLS INLINE OFF
  #pragma HLS DATAFLOW
    // clang-format on

    unsigned char low_th[MAXCOLORS][3], high_th[MAXCOLORS][3];
// clang-format off
  #pragma HLS ARRAY_PARTITION variable = low_th dim = 1 complete
  #pragma HLS ARRAY_PARTITION variable = high_th dim = 1 complete
    // clang-format on
    uint16_t j = 0;
    for (uint16_t i = 0; i < (MAXCOLORS); i++) {
// clang-format off
    #pragma HLS PIPELINE
        // clang-format on

        low_th[i][0] = low_thresh[j];
        low_th[i][1] = low_thresh[j + 1];
        low_th[i][2] = low_thresh[j + 2];
        high_th[i][0] = high_thresh[j];
        high_th[i][1] = high_thresh[j + 1];
        high_th[i][2] = high_thresh[j + 2];
        j = j + 3;
    }

    uint16_t img_height = _src_mat.rows;
    uint16_t img_width = _src_mat.cols;

    xFInRange<SRC_T, DST_T, ROWS, COLS, XF_DEPTH(SRC_T, NPC), XF_DEPTH(DST_T, NPC), NPC, XF_WORDWIDTH(SRC_T, NPC),
              XF_WORDWIDTH(DST_T, NPC), MAXCOLORS>(_src_mat, _dst_mat, low_th, high_th, img_height, img_width);
}
} // namespace cv
} // namespace xf

#endif
