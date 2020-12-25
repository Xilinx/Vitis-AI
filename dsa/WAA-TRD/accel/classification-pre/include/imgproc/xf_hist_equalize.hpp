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

#ifndef _XF_HIST_EQUALIZE_HPP_
#define _XF_HIST_EQUALIZE_HPP_

#ifndef __cplusplus
#error C++ is needed to include this header
#endif

#include "hls_stream.h"
#include "common/xf_common.hpp"
#include "common/xf_utility.hpp"
#include "imgproc/xf_histogram.hpp"

/**
 *  xfEqualize : Computes the histogram and performs
 *               Histogram Equalization
 *  _src1	: Input image
 *  _dst_mat	: Output image
 */
namespace xf {
namespace cv {

template <int SRC_T, int ROWS, int COLS, int DEPTH, int NPC, int WORDWIDTH, int SRC_TC>
void xFEqualize(xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _src1,
                uint32_t hist_stream[0][256],
                xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _dst_mat,
                uint16_t img_height,
                uint16_t img_width) {
    XF_SNAME(WORDWIDTH)
    in_buf, temp_buf;
    // Array to hold the values after cumulative distribution
    ap_uint<8> cum_hist[256];
// clang-format off
    #pragma HLS ARRAY_PARTITION variable=cum_hist complete dim=1
    // clang-format on
    // Temporary array to hold data
    ap_uint<8> tmp_cum_hist[(1 << XF_BITSHIFT(NPC))][256];
// clang-format off
    #pragma HLS ARRAY_PARTITION variable=tmp_cum_hist complete dim=1
    // clang-format on
    // Array which holds histogram of the image

    /*	Normalization	*/
    uint32_t temp_val = (uint32_t)(img_height * (img_width << XF_BITSHIFT(NPC)));
    uint32_t init_val = (uint32_t)(temp_val - hist_stream[0][0]);
    uint32_t scale;
    if (init_val == 0) {
        scale = 0;
    } else {
        scale = (uint32_t)(((1 << 31)) / init_val);
    }

    ap_uint<40> scale1 = (ap_uint<40>)((ap_uint<40>)255 * (ap_uint<40>)scale);
    ap_uint32_t temp_sum = 0;

    cum_hist[0] = 0;
Normalize_Loop:
    for (ap_uint<9> i = 1; i < 256; i++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=256 max=256
        #pragma HLS PIPELINE
        // clang-format on
        temp_sum = (uint32_t)temp_sum + (uint32_t)hist_stream[0][i];
        uint64_t sum = (uint64_t)((uint64_t)temp_sum * (uint64_t)scale1);
        sum = (uint64_t)(sum + 0x40000000);
        cum_hist[i] = sum >> 31;
    }

    for (ap_uint<9> i = 0; i < 256; i++) {
// clang-format off
        #pragma HLS PIPELINE
        // clang-format on
        for (ap_uint<5> j = 0; j < (1 << XF_BITSHIFT(NPC)); j++) {
// clang-format off
            #pragma HLS UNROLL
            // clang-format on
            ap_uint<8> tmpval = cum_hist[i];
            tmp_cum_hist[j][i] = tmpval;
        }
    }

NORMALISE_ROW_LOOP:
    for (ap_uint<13> row = 0; row < img_height; row++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=ROWS max=ROWS
    // clang-format on
    NORMALISE_COL_LOOP:
        for (ap_uint<13> col = 0; col < img_width; col++) {
// clang-format off
            #pragma HLS LOOP_TRIPCOUNT min=SRC_TC max=SRC_TC
            #pragma HLS PIPELINE
            #pragma HLS LOOP_FLATTEN OFF
            // clang-format on
            in_buf = _src1.read(row * img_width + col);
        Normalise_Extract:
            for (ap_uint<9> i = 0, j = 0; i < (8 << XF_BITSHIFT(NPC)); j++, i += 8) {
// clang-format off
                #pragma HLS DEPENDENCE variable=tmp_cum_hist array intra false
                #pragma HLS unroll
                // clang-format on
                XF_PTNAME(DEPTH)
                val;
                val = in_buf.range(i + 7, i);
                temp_buf(i + 7, i) = tmp_cum_hist[j][val];
            }
            _dst_mat.write(row * img_width + col, temp_buf);
        }
    }
}

/****************************************************************
 * equalizeHist : Wrapper function which calls the main kernel
 ****************************************************************/

template <int SRC_T, int ROWS, int COLS, int NPC = 1>
void equalizeHist(xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _src,
                  xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _src1,
                  xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _dst) {
// clang-format off
    #pragma HLS inline off
    // clang-format on

    uint16_t img_height = _src1.rows;
    uint16_t img_width = _src1.cols;
#ifndef __SYNTHESIS__
    assert(((img_height <= ROWS) && (img_width <= COLS)) && "ROWS and COLS should be greater than input image");

    assert((SRC_T == XF_8UC1) && "Type must be of XF_8UC1");
    assert(((NPC == XF_NPPC1) || (NPC == XF_NPPC8)) && " NPC must be XF_NPPC1, XF_NPPC8");
#endif
    uint32_t histogram[1][256];

    img_width = img_width >> XF_BITSHIFT(NPC);
    xFHistogramKernel<SRC_T, ROWS, COLS, XF_DEPTH(SRC_T, NPC), NPC, XF_WORDWIDTH(SRC_T, NPC),
                      ((COLS >> (XF_BITSHIFT(NPC))) >> 1), XF_CHANNELS(SRC_T, NPC)>(_src, histogram, img_height,
                                                                                    img_width);

    xFEqualize<SRC_T, ROWS, COLS, XF_DEPTH(SRC_T, NPC), NPC, XF_WORDWIDTH(SRC_T, NPC), (COLS >> XF_BITSHIFT(NPC))>(
        _src1, histogram, _dst, img_height, img_width);
}
} // namespace cv
} // namespace xf
#endif // _XF_HIST_EQUALIZE_H_
