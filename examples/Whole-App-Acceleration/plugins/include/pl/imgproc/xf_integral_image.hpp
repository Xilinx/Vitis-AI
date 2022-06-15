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

#ifndef _XF_INTEGRAL_IMAGE_HPP_
#define _XF_INTEGRAL_IMAGE_HPP_

typedef unsigned short uint16_t;

#include "hls_stream.h"
#include "common/xf_common.hpp"
#include "common/xf_utility.hpp"

namespace xf {
namespace cv {

template <int SRC_TYPE, int DST_TYPE, int ROWS, int COLS, int NPC, int WORDWIDTH_SRC, int WORDWIDTH_DST, int TC>
void XFIntegralImageKernel(xf::cv::Mat<SRC_TYPE, ROWS, COLS, NPC>& _src_mat,
                           xf::cv::Mat<DST_TYPE, ROWS, COLS, NPC>& _dst_mat,
                           uint16_t height,
                           uint16_t width) {
// clang-format off
    #pragma HLS inline
    // clang-format on
    XF_SNAME(XF_32UW) linebuff[COLS];

    uint32_t cur_sum;
    ap_uint<22> prev;
    ap_uint<13> i, j;
RowLoop:
    for (i = 0; i < height; i++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=ROWS max=ROWS
        // clang-format on

        XF_SNAME(XF_8UW) val;
        cur_sum = 0;
        prev = 0;

    ColLoop:
        for (j = 0; j < width; j++) {
// clang-format off
            #pragma HLS LOOP_TRIPCOUNT min=COLS max=COLS
            #pragma HLS LOOP_FLATTEN OFF
            #pragma HLS PIPELINE
            // clang-format on

            val = (XF_SNAME(XF_8UW))_src_mat.read(i * width + j);

            prev = prev + val;

            if (i == 0) {
                cur_sum = prev;
            } else {
                cur_sum = (prev + linebuff[j]);
            }

            linebuff[j] = cur_sum;
            _dst_mat.write(i * width + j, cur_sum);
        } // ColLoop

    } // rowLoop
}

// XFIntegralImage

template <int SRC_TYPE, int DST_TYPE, int ROWS, int COLS, int NPC>
void integral(xf::cv::Mat<SRC_TYPE, ROWS, COLS, NPC>& _src_mat, xf::cv::Mat<DST_TYPE, ROWS, COLS, NPC>& _dst_mat) {
// clang-format off
    #pragma HLS INLINE OFF
// clang-format on
#ifndef __SYNTHESIS__
    assert(((NPC == XF_NPPC1)) && "NPC must be XF_NPPC1");
    assert(((_src_mat.rows <= ROWS) && (_dst_mat.cols <= COLS)) &&
           "ROWS and COLS should be greater than or equal to input image size");
#endif
    XFIntegralImageKernel<SRC_TYPE, DST_TYPE, ROWS, COLS, NPC, XF_WORDWIDTH(SRC_TYPE, NPC), XF_WORDWIDTH(DST_TYPE, NPC),
                          (COLS >> XF_BITSHIFT(NPC))>(_src_mat, _dst_mat, _src_mat.rows, _src_mat.cols);
}
} // namespace cv
} // namespace xf

#endif //_XF_INTEGRAL_IMAGE_HPP_
