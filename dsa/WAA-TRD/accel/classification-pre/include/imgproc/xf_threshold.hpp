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

/**
 * xFThresholdKernel: Thresholds an input image and produces an output boolean image depending
 * 		upon the type of thresholding.
 * Input   : _src_mat, _thresh_type, _binary_thresh_val,  _upper_range and _lower_range
 * Output  : _dst_mat
 */
template <int SRC_T, int ROWS, int COLS, int DEPTH, int NPC, int WORDWIDTH_SRC, int WORDWIDTH_DST, int COLS_TRIP>
void xFThresholdKernel(xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _src_mat,
                       xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _dst_mat,
                       ap_uint<8> _thresh_type,
                       short int _thresh,
                       short int maxval,
                       unsigned short height,
                       unsigned short width) {
    XF_SNAME(WORDWIDTH_SRC) val_src;
    XF_SNAME(WORDWIDTH_DST) val_dst;
    XF_PTNAME(DEPTH) p; //,out;
    XF_PTNAME(DEPTH) thresh = (XF_PTNAME(DEPTH))_thresh;

    ap_uint<13> i, j, k;
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

            val_src =
                (XF_SNAME(WORDWIDTH_SRC))(_src_mat.read(i * width + j)); // reading the source stream _src into val_src

            for (k = 0; k < (XF_WORDDEPTH(WORDWIDTH_SRC)); k += XF_PIXELDEPTH(DEPTH)) {
// clang-format off
                #pragma HLS unroll
                // clang-format on
                p = val_src.range(k + (XF_PIXELDEPTH(DEPTH) - 1), k);

                switch (_thresh_type) {
                    case XF_THRESHOLD_TYPE_BINARY:
                        val_dst.range(k + (XF_PIXELDEPTH(DEPTH) - 1), k) =
                            (p > thresh) ? (XF_PTNAME(DEPTH))maxval : (ap_uint<8>)0;
                        break;
                    case XF_THRESHOLD_TYPE_BINARY_INV:
                        val_dst.range(k + (XF_PIXELDEPTH(DEPTH) - 1), k) =
                            (p > thresh) ? (ap_uint<8>)0 : (XF_PTNAME(DEPTH))maxval;
                        break;
                    case XF_THRESHOLD_TYPE_TRUNC:
                        val_dst.range(k + (XF_PIXELDEPTH(DEPTH) - 1), k) = (p > thresh) ? (XF_PTNAME(DEPTH))thresh : p;
                        break;
                    case XF_THRESHOLD_TYPE_TOZERO:
                        val_dst.range(k + (XF_PIXELDEPTH(DEPTH) - 1), k) = (p > thresh) ? p : (ap_uint<8>)0;
                        break;
                    case XF_THRESHOLD_TYPE_TOZERO_INV:
                        val_dst.range(k + (XF_PIXELDEPTH(DEPTH) - 1), k) = (p > thresh) ? (ap_uint<8>)0 : p;
                        break;
                    default:
                        val_dst.range(k + (XF_PIXELDEPTH(DEPTH) - 1), k) = p;
                }
            }

            _dst_mat.write(i * width + j, (val_dst)); // writing the val_dst into output stream _dst
        }
    }
}

template <int THRESHOLD_TYPE, int SRC_T, int ROWS, int COLS, int NPC = 1>
void Threshold(xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _src_mat,
               xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _dst_mat,
               short int thresh,
               short int maxval) {
    unsigned short width = _src_mat.cols >> XF_BITSHIFT(NPC);
    unsigned short height = _src_mat.rows;
#ifndef __SYNTHESIS__
    assert((SRC_T == XF_8UC1) && "Type must be XF_8UC1");
    assert(((NPC == XF_NPPC1) || (NPC == XF_NPPC8)) && "NPC must be XF_NPPC1, XF_NPPC8");

    assert(((THRESHOLD_TYPE == XF_THRESHOLD_TYPE_BINARY) || (THRESHOLD_TYPE == XF_THRESHOLD_TYPE_BINARY_INV) ||
            (THRESHOLD_TYPE == XF_THRESHOLD_TYPE_TRUNC) || (THRESHOLD_TYPE == XF_THRESHOLD_TYPE_TOZERO) ||
            (THRESHOLD_TYPE == XF_THRESHOLD_TYPE_TOZERO_INV)) &&
           "_thresh_type must be either XF_THRESHOLD_TYPE_BINARY or XF_THRESHOLD_TYPE_BINARY or "
           "XF_THRESHOLD_TYPE_BINARY_INV or XF_THRESHOLD_TYPE_TRUNC or XF_THRESHOLD_TYPE_TOZERO or "
           "XF_THRESHOLD_TYPE_TOZERO_INV");
    assert(((thresh >= 0) && (thresh <= 255)) && "_binary_thresh_val must be with the range of 0 to 255");

    assert(((height <= ROWS) && (width <= COLS)) && "ROWS and COLS should be greater than input image");
#endif
// clang-format off
    #pragma HLS INLINE OFF
    // clang-format on

    xFThresholdKernel<SRC_T, ROWS, COLS, XF_DEPTH(SRC_T, NPC), NPC, XF_WORDWIDTH(SRC_T, NPC), XF_WORDWIDTH(SRC_T, NPC),
                      (COLS >> XF_BITSHIFT(NPC))>(_src_mat, _dst_mat, THRESHOLD_TYPE, thresh, maxval, height, width);
}
} // namespace cv
} // namespace xf

#endif
