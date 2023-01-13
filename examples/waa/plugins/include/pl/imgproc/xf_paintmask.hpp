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

#ifndef _XF_THRESHOLD_HPP_
#define _XF_THRESHOLD_HPP_

#ifndef __cplusplus
#error C++ is needed to include this header
#endif

typedef unsigned short uint16_t;
typedef unsigned char uchar;

#include "hls_stream.h"
#include "common/xf_common.hpp"
#include "common/xf_utility.hpp"

namespace xf {
namespace cv {

/*Paint mask function masks certain area of image depends on input mask */
template <int SRC_T,
          int MASK_T,
          int ROWS,
          int COLS,
          int DEPTH,
          int NPC,
          int WORDWIDTH_SRC,
          int WORDWIDTH_DST,
          int WORDWIDTH_MASK,
          int COLS_TRIP>
void xFpaintmaskKernel(xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _src_mat,
                       xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _in_mask,
                       xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _dst_mat,
                       xf::cv::Scalar<XF_CHANNELS(SRC_T, NPC), unsigned char>& color,
                       unsigned short height,
                       unsigned short width) {
    XF_SNAME(WORDWIDTH_SRC) val_src;
    XF_SNAME(WORDWIDTH_MASK) in_mask;
    XF_SNAME(WORDWIDTH_DST) val_dst;
    XF_PTNAME(DEPTH) p, mask;
    short int depth = XF_DTPIXELDEPTH(SRC_T, NPC) / XF_CHANNELS(SRC_T, NPC);
    XF_PTNAME(DEPTH) arr_color[XF_CHANNELS(SRC_T, NPC)];
// clang-format off
    #pragma HLS ARRAY_PARTITION variable=arr_color dim=1 complete
    // clang-format on
    for (int i = 0; i < (XF_CHANNELS(SRC_T, NPC)); i++) {
        arr_color[i] = color.val[i];
    }
    ap_uint<13> i, j, k, planes;
rowLoop:
    for (i = 0; i < height; i++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=ROWS max=ROWS
        #pragma HLS LOOP_FLATTEN off
        // clang-format on
        ap_uint<8> channels = XF_CHANNELS(SRC_T, NPC);
    colLoop:
        for (j = 0; j < width; j++) {
// clang-format off
            #pragma HLS LOOP_TRIPCOUNT min=COLS_TRIP max=COLS_TRIP
            #pragma HLS pipeline
            // clang-format on

            val_src =
                (XF_SNAME(WORDWIDTH_SRC))(_src_mat.read(i * width + j)); // reading the source stream _src into val_src
            in_mask = (XF_SNAME(WORDWIDTH_MASK))(
                _in_mask.read(i * width + j)); // reading the input mask stream _in_mask into in_mask
            for (k = 0, planes = 0; k < (XF_WORDDEPTH(WORDWIDTH_SRC)); k += depth, planes++) {
// clang-format off
                #pragma HLS unroll
                // clang-format on
                p = val_src.range(k + (depth - 1), k);
                mask = in_mask.range(k + (depth - 1), k);
                if (mask != 0) {
                    if (NPC != 1)
                        val_dst.range(k + (depth - 1), k) = arr_color[0];
                    else
                        val_dst.range(k + (depth - 1), k) = arr_color[planes];
                } else {
                    val_dst.range(k + (depth - 1), k) = p;
                }
            }

            _dst_mat.write(i * width + j, val_dst); // writing the val_dst into output stream _dst
        }
    }
}

/* Paint mask API call*/
template <int SRC_T, int MASK_T, int ROWS, int COLS, int NPC = 1>
void paintmask(xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _src_mat,
               xf::cv::Mat<MASK_T, ROWS, COLS, NPC>& in_mask,
               xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _dst_mat,
               unsigned char _color[XF_CHANNELS(SRC_T, NPC)]) {
    unsigned short width = _src_mat.cols >> XF_BITSHIFT(NPC);
    unsigned short height = _src_mat.rows;
    xf::cv::Scalar<XF_CHANNELS(SRC_T, NPC), unsigned char> color;
    for (int i = 0; i < XF_CHANNELS(SRC_T, NPC); i++) {
        color.val[i] = _color[i];
    }
#ifndef __SYNTHESIS__
    assert((SRC_T == XF_8UC1) && "Type must be XF_8UC1");
    assert(((NPC == XF_NPPC1) || (NPC == XF_NPPC8)) && "NPC must be XF_NPPC1, XF_NPPC8");
    assert(((height <= ROWS) && (width <= COLS)) && "ROWS and COLS should be greater than input image");
#endif
// clang-format off
    #pragma HLS INLINE OFF
    // clang-format on

    xFpaintmaskKernel<SRC_T, MASK_T, ROWS, COLS, XF_DEPTH(SRC_T, NPC), NPC, XF_WORDWIDTH(SRC_T, NPC),
                      XF_WORDWIDTH(SRC_T, NPC), XF_WORDWIDTH(MASK_T, NPC), (COLS >> XF_BITSHIFT(NPC))>(
        _src_mat, in_mask, _dst_mat, color, height, width);
}
} // namespace cv
} // namespace xf

#endif
