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

#ifndef _XF_PHASE_HPP_
#define _XF_PHASE_HPP_

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

// to convert the radians value to degrees
#define XF_NORM_FACTOR 58671 // (180/PI)  in  Q6.10

/* xfPhaseKernel : The Gradient Phase Computation Kernel. This kernel takes
 * two gradients in AU_16SP depth and computes the angles for each pixel and
 * store this in a AU_16SP image.
 *  The Input arguments are _src1, _src2.
 *  _src1 --> Gradient X image from the output of sobel of depth AU_16SP.
 *  _src2 --> Gradient Y image from the output of sobel of depth AU_16SP.
 *  _dst  --> phase computed image of depth AU_16SP.
 *  Depending on NPC, 16 or 8 pixels are read and gradient values are calculated.
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
void xfPhaseKernel(xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _src1,
                   xf::cv::Mat<DST_T, ROWS, COLS, NPC>& _src2,
                   xf::cv::Mat<DST_T, ROWS, COLS, NPC>& _dst_mat,
                   int _out_format,
                   uint16_t& imgheight,
                   uint16_t& imgwidth) {
    int M1, N1, M2, N2; // Fixed point format of x and y, x = QM1.N1, y = QM2.N2
    M1 = 1;
    N1 = (XF_PIXELDEPTH(DEPTH_SRC)) - M1;
    M2 = M1;
    N2 = (XF_PIXELDEPTH(DEPTH_SRC)) - M2;

    XF_SNAME(WORDWIDTH_SRC) val_src1, val_src2;
    XF_SNAME(WORDWIDTH_DST) val_dst;

    int16_t p, q, ret = 0;
    int16_t result;
    int result_temp = 0;

rowLoop:
    for (ap_uint<13> i = 0; i < (imgheight); i++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=ROWS max=ROWS
        #pragma HLS LOOP_FLATTEN off
    // clang-format on

    colLoop:
        for (ap_uint<13> j = 0; j < (imgwidth); j++) {
// clang-format off
            #pragma HLS LOOP_TRIPCOUNT min=COLS_TRIP max=COLS_TRIP
            #pragma HLS pipeline
            // clang-format on

            val_src1 = (XF_SNAME(WORDWIDTH_SRC))(_src1.read(i * imgwidth + j));
            val_src2 = (XF_SNAME(WORDWIDTH_SRC))(_src2.read(i * imgwidth + j));

            int proc_loop = XF_WORDDEPTH(WORDWIDTH_DST), step = XF_PIXELDEPTH(DEPTH_DST);

        procLoop:
            for (ap_uint<9> k = 0; k < proc_loop; k += step) {
// clang-format off
                #pragma HLS unroll
                // clang-format on
                p = val_src1.range(k + (step - 1), k); // Get bits from certain range of positions.
                q = val_src2.range(k + (step - 1), k); // Get bits from certain range of positions.

                ret = xf::cv::Atan2LookupFP(p, q, M1, N1, M2, N2);

                if (ret < 0) {
                    result_temp = ret + XF_PI_FIXED + XF_PI_FIXED;
                } else if (ret == 0 && q < 0) {
                    result_temp = ret + XF_PI_FIXED + XF_PI_FIXED;
                } else {
                    result_temp = ret;
                }
                if (_out_format == XF_DEGREES) {
                    // result_temp = result_temp + 0x40;
                    // result = (XF_NORM_FACTOR * result_temp)>>16;
                    result = (XF_NORM_FACTOR * result_temp + 0x8000) >> 16;
                } else if (_out_format == XF_RADIANS) {
                    result = result_temp;
                }
                val_dst.range(k + (step - 1), k) = result; // set the values in val_dst.
            }                                              // end of proc loop
            _dst_mat.write(i * imgwidth + j, (val_dst));
        } // end of col loop
    }     // end of row loop
}

/**
 * xfPhase: This function acts as a wrapper function and
 * calls the Kernel function.
 */
template <int ROWS, int COLS, int DEPTH_SRC, int DEPTH_DST, int NPC, int WORDWIDTH_SRC, int WORDWIDTH_DST>
void xFPhaseComputation(hls::stream<XF_SNAME(WORDWIDTH_SRC)>& _src1,
                        hls::stream<XF_SNAME(WORDWIDTH_SRC)>& _src2,
                        hls::stream<XF_SNAME(WORDWIDTH_DST)>& _dst,
                        int _out_format,
                        uint16_t imgheight,
                        uint16_t imgwidth) {
    imgwidth = imgwidth >> XF_BITSHIFT(NPC);

    xfPhaseKernel<ROWS, COLS, DEPTH_SRC, DEPTH_DST, NPC, WORDWIDTH_SRC, WORDWIDTH_DST, (COLS >> XF_BITSHIFT(NPC))>(
        _src1, _src2, _dst, _out_format, imgheight, imgwidth);
}

template <int RET_TYPE, int SRC_T, int DST_T, int ROWS, int COLS, int NPC>
void phase(xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _src_matx,
           xf::cv::Mat<DST_T, ROWS, COLS, NPC>& _src_maty,
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

    uint16_t imgwidth = _src_matx.cols >> XF_BITSHIFT(NPC);
    uint16_t imgheight = _src_matx.rows;

// clang-format off
    #pragma HLS INLINE OFF
    // clang-format on

    xfPhaseKernel<SRC_T, DST_T, ROWS, COLS, XF_DEPTH(SRC_T, NPC), XF_DEPTH(DST_T, NPC), NPC, XF_WORDWIDTH(SRC_T, NPC),
                  XF_WORDWIDTH(DST_T, NPC), (COLS >> XF_BITSHIFT(NPC))>(_src_matx, _src_maty, _dst_mat, RET_TYPE,
                                                                        imgheight, imgwidth);
}
} // namespace cv
} // namespace xf

#endif
