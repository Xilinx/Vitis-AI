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

#ifndef _XF_MAGNITUDE_HPP_
#define _XF_MAGNITUDE_HPP_

#ifndef __cplusplus
#error C++ is needed to include this header
#endif

typedef unsigned short uint16_t;

#include "hls_stream.h"
#include "ap_int.h"
#include "common/xf_common.hpp"
#include "common/xf_utility.hpp"
#include "core/xf_math.h"

namespace xf {
namespace cv {

/**
 *  xFMagnitudeKernel : The Gradient Magnitude Computation Kernel.
 *  This kernel takes two gradients in AU_16SP format and computes the
 *  AU_16SP normalized magnitude.
 *  The Input arguments are src1, src2 and Norm.
 *  src1 --> Gradient X image from the output of sobel of depth AU_16SP.
 *  src2 --> Gradient Y image from the output of sobel of depth AU_16SP.
 *  _norm_type  --> Either AU_L1Norm or AU_L2Norm which are o and 1 respectively.
 *  _dst --> Magnitude computed image of depth AU_16SP.
 *  Depending on NPC, 16 or 8 pixels are read and gradient values are
 *  calculated.
 */
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
void xFMagnitudeKernel(xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _src1,
                       xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _src2,
                       xf::cv::Mat<DST_T, ROWS, COLS, NPC>& _dst_mat,
                       int _norm_type,
                       uint16_t& imgheight,
                       uint16_t& imgwidth) {
    XF_SNAME(WORDWIDTH_SRC) val_src1, val_src2;
    XF_SNAME(WORDWIDTH_DST) val_dst;

    int tempgx, tempgy, result_temp = 0;
    int16_t p, q;
    int16_t result;

rowLoop:
    for (int i = 0; i < (imgheight); i++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=ROWS max=ROWS
        #pragma HLS LOOP_FLATTEN off
    // clang-format on

    colLoop:
        for (int j = 0; j < (imgwidth); j++) {
// clang-format off
            #pragma HLS LOOP_TRIPCOUNT min=COLS_TRIP max=COLS_TRIP
            #pragma HLS pipeline
            // clang-format on

            val_src1 = (XF_SNAME(WORDWIDTH_SRC))(_src1.read(i * imgwidth + j));
            val_src2 = (XF_SNAME(WORDWIDTH_SRC))(_src2.read(i * imgwidth + j));

            int proc_loop = XF_WORDDEPTH(WORDWIDTH_DST), step = XF_PIXELDEPTH(DEPTH_DST);

        procLoop:
            for (int k = 0; k < proc_loop; k += step) {
// clang-format off
                #pragma HLS unroll
                // clang-format on

                p = val_src1.range(k + (step - 1), k); // Get bits from certain range of positions.
                q = val_src2.range(k + (step - 1), k); // Get bits from certain range of positions.
                p = __ABS(p);
                q = __ABS(q);

                if (_norm_type == XF_L1NORM) {
                    int16_t tmp = p + q;
                    result = tmp;
                } else if (_norm_type == XF_L2NORM) {
                    tempgx = p * p;
                    tempgy = q * q;
                    result_temp = tempgx + tempgy;
                    int tmp1 = xf::cv::Sqrt(result_temp); // Square root of the gradient images
                    result = (int16_t)tmp1;
                }
                val_dst.range(k + (step - 1), k) = result;
            }
            _dst_mat.write(i * imgwidth + j, (val_dst)); // writing into the output stream
        }
    }
}

template <int NORM_TYPE, int SRC_T, int DST_T, int ROWS, int COLS, int NPC>
void magnitude(xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _src_matx,
               xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _src_maty,
               xf::cv::Mat<DST_T, ROWS, COLS, NPC>& _dst_mat) {
#ifndef __SYNTHESIS__
    assert(((_src_matx.rows <= ROWS) && (_src_matx.cols <= COLS)) &&
           "ROWS and COLS should be greater than input image");
    assert(((_src_maty.rows <= ROWS) && (_src_maty.cols <= COLS)) &&
           "ROWS and COLS should be greater than input image");
    assert(((_src_matx.rows == _src_maty.rows) && (_src_matx.cols == _src_maty.cols)) &&
           "Both input images should have same size");
    assert(((_src_matx.rows == _dst_mat.rows) && (_src_matx.cols == _dst_mat.cols)) &&
           "Input and output image should be of same size");
    assert(((NPC == XF_NPPC1) || (NPC == XF_NPPC8)) && "NPC must be XF_NPPC1, XF_NPPC8 ");
#endif

// clang-format off
    #pragma HLS inline
    // clang-format on

    uint16_t imgwidth = _src_matx.cols >> XF_BITSHIFT(NPC);
    uint16_t height = _src_matx.rows;

    xFMagnitudeKernel<SRC_T, DST_T, ROWS, COLS, XF_DEPTH(SRC_T, NPC), XF_DEPTH(DST_T, NPC), NPC,
                      XF_WORDWIDTH(SRC_T, NPC), XF_WORDWIDTH(DST_T, NPC), (COLS >> XF_BITSHIFT(NPC))>(
        _src_matx, _src_maty, _dst_mat, NORM_TYPE, height, imgwidth);
}
} // namespace cv
} // namespace xf

#endif
