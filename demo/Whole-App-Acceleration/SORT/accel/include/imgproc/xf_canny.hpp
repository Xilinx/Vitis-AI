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

#ifndef _XF_CANNY_HPP_
#define _XF_CANNY_HPP_

#ifndef __cplusplus
#error C++ is needed to use this file!
#endif

typedef unsigned short uint16_t;
typedef unsigned char uchar;

#include "hls_stream.h"
#include "common/xf_common.hpp"
#include "common/xf_utility.hpp"

#include "core/xf_math.h"
//#include "imgproc/xf_sobel.hpp"
#include "xf_canny_sobel.hpp"
#include "xf_averagegaussianmask.hpp"
#include "xf_magnitude.hpp"
#include "xf_canny_utils.hpp"

namespace xf {
namespace cv {
/**
 *  xFDuplicate_rows
 */
template <int IN_T, int ROWS, int COLS, int DEPTH, int NPC, int WORDWIDTH, int TC>
void xFDuplicate_rows(xf::cv::Mat<IN_T, ROWS, COLS, NPC>& _src_mat, // hls::stream< XF_SNAME(WORDWIDTH) > &_src_mat,
                      xf::cv::Mat<IN_T, ROWS, COLS, NPC>& _src_mat1,
                      xf::cv::Mat<IN_T, ROWS, COLS, NPC>& _dst1_mat,
                      xf::cv::Mat<IN_T, ROWS, COLS, NPC>& _dst2_mat,
                      xf::cv::Mat<IN_T, ROWS, COLS, NPC>& _dst1_out_mat,
                      xf::cv::Mat<IN_T, ROWS, COLS, NPC>& _dst2_out_mat,
                      uint16_t img_height,
                      uint16_t img_width) {
    img_width = img_width >> XF_BITSHIFT(NPC);

    ap_uint<13> row, col;
Row_Loop:
    for (row = 0; row < img_height; row++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=ROWS max=ROWS
        #pragma HLS LOOP_FLATTEN off
    // clang-format on
    Col_Loop:
        for (col = 0; col < img_width; col++) {
// clang-format off
            #pragma HLS LOOP_TRIPCOUNT min=TC max=TC
            #pragma HLS pipeline
            // clang-format on
            XF_SNAME(WORDWIDTH) tmp_src, tmp_src1;
            tmp_src1 = _src_mat1.read(row * img_width + col);
            tmp_src = _src_mat.read(row * img_width + col);
            _dst1_mat.write(row * img_width + col, tmp_src);
            _dst2_mat.write(row * img_width + col, tmp_src);
            _dst1_out_mat.write(row * img_width + col, tmp_src1);
            _dst2_out_mat.write(row * img_width + col, tmp_src1);
        }
    }
}

template <int SRC_T,
          int DST_T,
          int ROWS,
          int COLS,
          int DEPTH_SRC,
          int DEPTH_DST,
          int NPC,
          int NPC1,
          int WORDWIDTH_SRC,
          int WORDWIDTH_DST>
void xFPackNMS(xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _src_mat,  // hls::stream< XF_SNAME(WORDWIDTH_SRC) >& _src_mat,
               xf::cv::Mat<DST_T, ROWS, COLS, NPC1>& _dst_mat, // hls::stream< XF_SNAME(WORDWIDTH_DST)>& _dst_mat,
               uint16_t imgheight,
               uint16_t imgwidth) {
    const int num_clks_32pix = 32 / NPC;
    int col_loop_count = (imgwidth / NPC);
    ap_uint<64> val;
    int read_ind = 0, write_ind = 0;
rowLoop:
    for (int i = 0; i < (imgheight); i++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=ROWS max=ROWS
        #pragma HLS LOOP_FLATTEN off
    // clang-format on

    colLoop:
        for (int j = 0; j < col_loop_count; j = j + (num_clks_32pix)) {
// clang-format off
            #pragma HLS LOOP_TRIPCOUNT min=COLS/32 max=COLS/32
            #pragma HLS pipeline
            // clang-format on

            for (int k = 0; k < num_clks_32pix; k++) {
// clang-format off
                #pragma HLS UNROLL
                // clang-format on
                val.range(k * 2 * NPC + (NPC * 2 - 1), k * 2 * NPC) = _src_mat.read(read_ind++);
            }
            _dst_mat.write(write_ind++, val);
        }
    }
}

// xFDuplicate_rows

template <int SRC_T,
          int DST_T,
          int NORM_TYPE,
          int ROWS,
          int COLS,
          int DEPTH_IN,
          int DEPTH_OUT,
          int NPC,
          int NPC1,
          int WORDWIDTH_SRC,
          int WORDWIDTH_DST,
          int TC,
          int TC1,
          int TC2,
          int FILTER_TYPE,
          bool USE_URAM>
void xFCannyKernel(xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _src_mat,
                   xf::cv::Mat<DST_T, ROWS, COLS, NPC1>& _dst_mat,
                   unsigned char _lowthreshold,
                   unsigned char _highthreshold,
                   uint16_t img_height,
                   uint16_t img_width) {
// clang-format off
    #pragma HLS INLINE OFF
// clang-format on

// clang-format off
        #pragma HLS DATAFLOW
    // clang-format on

    if (NPC == 8) {
        xf::cv::Mat<XF_8UC1, ROWS, COLS, NPC> gaussian_mat(img_height, img_width);
        xf::cv::Mat<XF_16SC1, ROWS, COLS, NPC> gradx_mat(img_height, img_width);
        xf::cv::Mat<XF_16SC1, ROWS, COLS, NPC> gradx1_mat(img_height, img_width);
        xf::cv::Mat<XF_16SC1, ROWS, COLS, NPC> gradx2_mat(img_height, img_width);
        xf::cv::Mat<XF_16SC1, ROWS, COLS, NPC> grady_mat(img_height, img_width);
        xf::cv::Mat<XF_16SC1, ROWS, COLS, NPC> grady1_mat(img_height, img_width);
        xf::cv::Mat<XF_16SC1, ROWS, COLS, NPC> grady2_mat(img_height, img_width);
        xf::cv::Mat<XF_16SC1, ROWS, COLS, NPC, TC1> magnitude_mat(img_height, img_width);
        xf::cv::Mat<XF_8UC1, ROWS, COLS, NPC, TC2> phase_mat(img_height, img_width);
        xf::cv::Mat<XF_2UC1, ROWS, COLS, NPC> nms_mat(img_height, img_width);

        xFAverageGaussianMask3x3<SRC_T, SRC_T, ROWS, COLS, DEPTH_IN, NPC, WORDWIDTH_SRC, (COLS >> XF_BITSHIFT(NPC))>(
            _src_mat, gaussian_mat, img_height, img_width);
        xFSobel<SRC_T, XF_16SC1, ROWS, COLS, DEPTH_IN, XF_16SP, NPC, WORDWIDTH_SRC, XF_128UW, FILTER_TYPE, USE_URAM>(
            gaussian_mat, gradx_mat, grady_mat, XF_BORDER_REPLICATE, img_height, img_width);
        xFDuplicate_rows<XF_16SC1, ROWS, COLS, XF_16SP, NPC, XF_128UW, TC>(
            gradx_mat, grady_mat, gradx1_mat, gradx2_mat, grady1_mat, grady2_mat, img_height, img_width);
        magnitude<NORM_TYPE, XF_16SC1, XF_16SC1, ROWS, COLS, NPC, TC1>(gradx1_mat, grady1_mat, magnitude_mat);
        xFAngle<XF_16SC1, XF_8UC1, ROWS, COLS, XF_16SP, XF_8UP, NPC, XF_128UW, XF_64UW, TC2>(
            gradx2_mat, grady2_mat, phase_mat, img_height, img_width);
        xFSuppression3x3<XF_16SC1, XF_8UC1, XF_2UC1, ROWS, COLS, XF_16SP, XF_8UP, DEPTH_OUT, NPC, XF_128UW, XF_64UW,
                         XF_16UW, (COLS >> XF_BITSHIFT(NPC)), TC2, TC1>(
            magnitude_mat, phase_mat, nms_mat, _lowthreshold, _highthreshold, img_height, img_width);
        xFPackNMS<XF_2UC1, DST_T, ROWS, COLS, XF_2UP, DEPTH_OUT, NPC, NPC1, XF_16UW, WORDWIDTH_DST>(
            nms_mat, _dst_mat, img_height, img_width);
    }

    if (NPC == 1) {
        xf::cv::Mat<XF_8UC1, ROWS, COLS, NPC> gaussian_mat(img_height, img_width);
        xf::cv::Mat<XF_16SC1, ROWS, COLS, NPC> gradx_mat(img_height, img_width);
        xf::cv::Mat<XF_16SC1, ROWS, COLS, NPC> gradx1_mat(img_height, img_width);
        xf::cv::Mat<XF_16SC1, ROWS, COLS, NPC> gradx2_mat(img_height, img_width);
        xf::cv::Mat<XF_16SC1, ROWS, COLS, NPC> grady_mat(img_height, img_width);
        xf::cv::Mat<XF_16SC1, ROWS, COLS, NPC> grady1_mat(img_height, img_width);
        xf::cv::Mat<XF_16SC1, ROWS, COLS, NPC> grady2_mat(img_height, img_width);
        xf::cv::Mat<XF_16SC1, ROWS, COLS, NPC, TC1> magnitude_mat(img_height, img_width);
        xf::cv::Mat<XF_8UC1, ROWS, COLS, NPC, TC2> phase_mat(img_height, img_width);
        xf::cv::Mat<XF_2UC1, ROWS, COLS, NPC> nms_mat(img_height, img_width);

        xFAverageGaussianMask3x3<SRC_T, SRC_T, ROWS, COLS, DEPTH_IN, NPC, WORDWIDTH_SRC, (COLS >> XF_BITSHIFT(NPC))>(
            _src_mat, gaussian_mat, img_height, img_width);
        xFSobel<SRC_T, XF_16SC1, ROWS, COLS, DEPTH_IN, XF_16SP, NPC, WORDWIDTH_SRC, XF_16UW, FILTER_TYPE, USE_URAM>(
            gaussian_mat, gradx_mat, grady_mat, XF_BORDER_REPLICATE, img_height, img_width);
        xFDuplicate_rows<XF_16SC1, ROWS, COLS, XF_16SP, NPC, XF_16UW, TC>(
            gradx_mat, grady_mat, gradx1_mat, gradx2_mat, grady1_mat, grady2_mat, img_height, img_width);
        magnitude<NORM_TYPE, XF_16SC1, XF_16SC1, ROWS, COLS, NPC, TC1>(gradx1_mat, grady1_mat, magnitude_mat);
        xFAngle<XF_16SC1, XF_8UC1, ROWS, COLS, XF_16SP, XF_8UP, NPC, XF_16UW, XF_8UW, TC2>(
            gradx2_mat, grady2_mat, phase_mat, img_height, img_width);
        xFSuppression3x3<XF_16SC1, XF_8UC1, XF_2UC1, ROWS, COLS, XF_16SP, XF_8UP, XF_2UP, NPC, XF_16UW, XF_8UW, XF_2UW,
                         (COLS >> XF_BITSHIFT(NPC)), TC2, TC1>(magnitude_mat, phase_mat, nms_mat, _lowthreshold,
                                                               _highthreshold, img_height, img_width);
        xFPackNMS<XF_2UC1, DST_T, ROWS, COLS, XF_2UP, DEPTH_OUT, NPC, NPC1, XF_2UW, WORDWIDTH_DST>(
            nms_mat, _dst_mat, img_height, img_width);
    }
}

/**********************************************************************
 * xFCanny :  Calls the Main Function depends on requirements
 **********************************************************************/
template <int ROWS,
          int COLS,
          int DEPTH_IN,
          int DEPTH_OUT,
          int NPC,
          int WORDWIDTH_SRC,
          int WORDWIDTH_DST,
          int FILTER_TYPE,
          bool USE_URAM>
void xFCannyEdgeDetector(hls::stream<XF_SNAME(WORDWIDTH_SRC)>& _src_mat,
                         hls::stream<XF_SNAME(WORDWIDTH_DST)>& out_strm,
                         unsigned char _lowthreshold,
                         unsigned char _highthreshold,
                         int _norm_type,
                         uint16_t imgheight,
                         uint16_t imgwidth) {
#ifndef __SYNTHESIS__
    assert(((_norm_type == XF_L1NORM) || (_norm_type == XF_L2NORM)) &&
           "The _norm_type must be 'XF_L1NORM' or'XF_L2NORM'");
#endif
    xFCannyKernel<ROWS, COLS, DEPTH_IN, DEPTH_OUT, NPC, WORDWIDTH_SRC, WORDWIDTH_DST, (COLS >> XF_BITSHIFT(NPC)),
                  ((COLS >> XF_BITSHIFT(NPC)) * 3), FILTER_TYPE, USE_URAM>(
        _src_mat, out_strm, _lowthreshold, _highthreshold, _norm_type, imgheight, imgwidth);
}

template <int FILTER_TYPE,
          int NORM_TYPE,
          int SRC_T,
          int DST_T,
          int ROWS,
          int COLS,
          int NPC,
          int NPC1,
          bool USE_URAM = false>
void Canny(xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _src_mat,
           xf::cv::Mat<DST_T, ROWS, COLS, NPC1>& _dst_mat,
           unsigned char _lowthreshold,
           unsigned char _highthreshold) {
// clang-format off
    #pragma HLS INLINE OFF
// clang-format on
#ifndef __SYNTHESIS__
    assert(((NORM_TYPE == XF_L1NORM) || (NORM_TYPE == XF_L2NORM)) &&
           "The _norm_type must be 'XF_L1NORM' or'XF_L2NORM'");
#endif

    if (NORM_TYPE == 1) {
        xFCannyKernel<SRC_T, DST_T, NORM_TYPE, ROWS, COLS, XF_DEPTH(SRC_T, NPC), XF_DEPTH(DST_T, NPC1), NPC, NPC1,
                      XF_WORDWIDTH(SRC_T, NPC), XF_WORDWIDTH(DST_T, NPC1), (COLS >> XF_BITSHIFT(NPC)), 2,
                      ((COLS >> XF_BITSHIFT(NPC)) * 3), FILTER_TYPE, USE_URAM>(
            _src_mat, _dst_mat, _lowthreshold, _highthreshold, _src_mat.rows, _src_mat.cols);
    } else {
        xFCannyKernel<SRC_T, DST_T, NORM_TYPE, ROWS, COLS, XF_DEPTH(SRC_T, NPC), XF_DEPTH(DST_T, NPC1), NPC, NPC1,
                      XF_WORDWIDTH(SRC_T, NPC), XF_WORDWIDTH(DST_T, NPC1), (COLS >> XF_BITSHIFT(NPC)), 2,
                      ((COLS >> XF_BITSHIFT(NPC)) * 3), FILTER_TYPE, USE_URAM>(
            _src_mat, _dst_mat, _lowthreshold, _highthreshold, _src_mat.rows, _src_mat.cols);
    }
}
} // namespace cv
} // namespace xf

#endif
