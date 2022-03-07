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

#ifndef _XF_BOUNDINGBOX_HPP_
#define _XF_BOUNDINGBOX_HPP_

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
// template<int NPC>
// bool IsOnBoundary(unsigned int i, unsigned int j, unsigned int r, unsigned int c, unsigned int r_new, unsigned int
// c_new)
//{
//#pragma HLS INLINE OFF
//
//	if((((i==r) || (i==(r_new-1))) &&  j<c_new && j>=c) || (((j==c) || (j==(c_new-1))) && i<r_new && i>=r))
//	{
//
//		return 1;
//
//	}
//	else
//	{
//		return 0;
//
//	}
//
//
//}
/**
 * CROP kernel: crops the ROI of an input image and produces an output.
 * Input   : _src_mat, roi
 * Output  : _dst_mat
 */
template <int SRC_T,
          int ROWS,
          int COLS,
          int MAX_BOXES,
          int DEPTH,
          int NPC,
          int WORDWIDTH_SRC,
          int WORDWIDTH_DST,
          int COLS_TRIP>
void xFboundingboxkernel(xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _src_mat,
                         xf::cv::Rect_<int>* roi,
                         xf::cv::Scalar<4, unsigned char>* color,
                         int num_box,
                         unsigned short height,
                         unsigned short width) {
// clang-format off
    #pragma HLS INLINE
    // clang-format on
    XF_SNAME(WORDWIDTH_SRC) val_src = 0, val_dst = 0;
    ap_uint<13> r[MAX_BOXES], c[MAX_BOXES], r_new[MAX_BOXES], c_new[MAX_BOXES];
    XF_TNAME(SRC_T, NPC) color_box[MAX_BOXES];
// clang-format off
    #pragma HLS ARRAY_PARTITION variable=r complete
    #pragma HLS ARRAY_PARTITION variable=c complete
    #pragma HLS ARRAY_PARTITION variable=r_new complete
    #pragma HLS ARRAY_PARTITION variable=r_new complete
    #pragma HLS ARRAY_PARTITION variable=color_box complete
    // clang-format on

    ap_uint<2> found = 0;
    ap_uint<2> modify_pix = 0;
    int color_idx = 0;
    ap_uint<13> r_idx = 0, r_newidx = 0, c_idx = 0, c_newidx = 0;

    for (ap_uint<13> i = 0; i < num_box; i++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=MAX_BOXES max=MAX_BOXES
        #pragma HLS UNROLL
        // clang-format on
        /*	r[i]=roi[i].x;
                r_new[i]=roi[i].x+roi[i].height;
                c[i]=roi[i].y ;
                c_new[i]=(roi[i].y+roi[i].width);*/

        c[i] = roi[i].x;
        c_new[i] = roi[i].x + roi[i].width;
        r[i] = roi[i].y;
        r_new[i] = (roi[i].y + roi[i].height);
    }
    for (int i = 0; i < (num_box); i++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=MAX_BOXES max=MAX_BOXES
        #pragma HLS PIPELINE
        // clang-format on
        for (int j = 0, k = 0; j < (XF_CHANNELS(SRC_T, NPC)); j++, k += XF_DTPIXELDEPTH(SRC_T, NPC)) {
// clang-format off
            #pragma HLS UNROLL
            // clang-format on
            color_box[i].range(k + (XF_DTPIXELDEPTH(SRC_T, NPC) - 1), k) = color[i].val[j];
        }
    }

    for (ap_uint<13> b = 0; b < num_box; b++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=MAX_BOXES max=MAX_BOXES
    // clang-format on
    colLoop:
        for (ap_uint<13> j = c[b]; j < (c_new[b]); j++) {
// clang-format off
            #pragma HLS LOOP_TRIPCOUNT min=COLS max=COLS
            #pragma HLS pipeline
            // clang-format on
            _src_mat.write(r[b] * width + j, color_box[b]);
        }
        for (ap_uint<13> j = c[b]; j < (c_new[b]); j++) {
// clang-format off
            #pragma HLS LOOP_TRIPCOUNT min=COLS max=COLS
            #pragma HLS pipeline
            // clang-format on

            _src_mat.write((r_new[b] - 1) * width + j, color_box[b]);
        }

    rowLoop1:
        for (ap_uint<13> i = (r[b] + 1); i < (r_new[b] - 1); i++) {
// clang-format off
            #pragma HLS LOOP_TRIPCOUNT min=ROWS max=ROWS
            #pragma HLS pipeline
            // clang-format on
            //	#pragma HLS LOOP_FLATTEN

            _src_mat.write(i * width + c[b], color_box[b]);
            _src_mat.write(i * width + (c_new[b] - 1), color_box[b]);
        }
    }
}

template <int SRC_T, int ROWS, int COLS, int MAX_BOXES = 1, int NPC = 1>
void boundingbox(xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _src_mat,
                 xf::cv::Rect_<int>* roi,
                 xf::cv::Scalar<4, unsigned char>* color,
                 int num_box) {
    unsigned short width = _src_mat.cols;
    unsigned short height = _src_mat.rows;
#ifndef __SYNTHESIS__
    assert((SRC_T == XF_8UC1) || (SRC_T == XF_8UC4) && "Type must be XF_8UC1 or XF_8UC4");
    assert((NPC == XF_NPPC1) && "NPC must be 1, Multipixel parallelism is not supported");

    assert(((height <= ROWS) && (width <= COLS)) && "ROWS and COLS should be greater than input image");

    for (int i = 0; i < num_box; i++) {
        assert(((roi[i].height <= height) && (roi[i].width <= width)) &&
               "ROI dimensions should be smaller or equal to the input image");
        assert(((roi[i].height > 0) && (roi[i].width > 0)) && "ROI  dimensions should be greater than 0");
        assert(((roi[i].height + roi[i].y <= height) && (roi[i].width + roi[i].x <= width)) &&
               "ROI area exceeds the input image area");
    }
#endif
// clang-format off
    #pragma HLS INLINE
    // clang-format on

    xFboundingboxkernel<SRC_T, ROWS, COLS, MAX_BOXES, XF_DEPTH(SRC_T, NPC), NPC, XF_WORDWIDTH(SRC_T, NPC),
                        XF_WORDWIDTH(SRC_T, NPC), (COLS >> XF_BITSHIFT(NPC))>(_src_mat, roi, color, num_box, height,
                                                                              width);
}
} // namespace cv
} // namespace xf

#endif
