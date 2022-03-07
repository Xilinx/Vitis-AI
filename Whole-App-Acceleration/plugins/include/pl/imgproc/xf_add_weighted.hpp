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

#ifndef _XF_ADD_WEIGHTED_HPP_
#define _XF_ADD_WEIGHTED_HPP_

#include "hls_stream.h"
#include "common/xf_common.hpp"

#ifndef XF_IN_STEP
#define XF_IN_STEP 8
#endif
#ifndef XF_OUT_STEP
#define XF_OUT_STEP 8
#endif
/* calculates the weighted sum of 2 inut images */
namespace xf {
namespace cv {
template <int SRC_T,
          int DST_T,
          int ROWS,
          int COLS,
          int NPC,
          int PLANES,
          int DEPTH_SRC,
          int DEPTH_DST,
          int WORDWIDTH_SRC,
          int WORDWIDTH_DST,
          int TC>
int AddWeightedKernel(xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& src1,
                      float alpha,
                      xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& src2,
                      float beta,
                      float gama,
                      xf::cv::Mat<DST_T, ROWS, COLS, NPC>& dst,
                      uint16_t height,
                      uint16_t width) {
    ap_uint<13> i, j, k, l;

    ap_fixed<16, 8, AP_RND> temp = alpha;
    ap_fixed<16, 8, AP_RND> temp1 = beta;
    ap_fixed<16, 8, AP_RND> temp2 = gama;

    int STEP = XF_PIXELWIDTH(SRC_T, NPC) / PLANES;

    XF_SNAME(WORDWIDTH_DST) pxl_pack_out;
    XF_SNAME(WORDWIDTH_SRC) pxl_pack1, pxl_pack2;
RowLoop:
    for (i = 0; i < height; i++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=ROWS max=ROWS
        #pragma HLS LOOP_FLATTEN OFF
    // clang-format on
    ColLoop:
        for (j = 0; j < width; j++) {
// clang-format off
            #pragma HLS LOOP_TRIPCOUNT min=TC max=TC
            #pragma HLS pipeline
            // clang-format on

            pxl_pack1 = (XF_SNAME(WORDWIDTH_SRC))(src1.read(i * width + j)); // reading from 1st input stream
            pxl_pack2 = (XF_SNAME(WORDWIDTH_SRC))(src2.read(i * width + j)); // reading from 2nd input stream
        ProcLoop:
            for (k = 0, l = 0; k < ((8 << XF_BITSHIFT(NPC)) * PLANES); k += XF_IN_STEP, l += XF_OUT_STEP) {
                XF_PTNAME(DEPTH_SRC) pxl1 = pxl_pack1.range(k + 7, k); // extracting each pixel in case of 8-pixel mode
                XF_PTNAME(DEPTH_SRC) pxl2 = pxl_pack2.range(k + 7, k);

                ap_int<24> firstcmp = pxl1 * temp;
                ap_int<24> secondcmp = pxl2 * temp1;

                ap_int<16> t = (firstcmp + secondcmp + temp2); // >> 23;

                if (t > 255) {
                    t = 255;
                } else if (t < 0) {
                    t = 0;
                }

                pxl_pack_out.range(l + XF_OUT_STEP - 1, l) = (unsigned char)t;
            }

            dst.write(i * width + j, (XF_SNAME(WORDWIDTH_DST))pxl_pack_out); // writing into output stream
        }
    }
    return 0;
}

template <int SRC_T, int DST_T, int ROWS, int COLS, int NPC = 1>
void addWeighted(xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& src1,
                 float alpha,
                 xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& src2,
                 float beta,
                 float gama,
                 xf::cv::Mat<DST_T, ROWS, COLS, NPC>& dst) {
#ifndef __SYNTHESIS__
    assert(((SRC_T == XF_8UC1) || (SRC_T == XF_8UC3)) &&
           "Input TYPE must be XF_8UC1 for 1-channel, XF_8UC3 for 3-channel");
    assert(((DST_T == XF_8UC1) || (DST_T == XF_8UC3)) &&
           "Output TYPE must be XF_8UC1 for 1-channel,XF_8UC3 for 3-channel ");
    assert(((src1.rows == src2.rows) && (src1.cols == src2.cols)) && "Both input images should have same size");
    assert(((src1.rows == dst.rows) && (src1.cols == dst.cols)) && "Input and output image should be of same size");
    assert(((src1.rows <= ROWS) && (src1.cols <= COLS)) && "ROWS and COLS should be greater than input image");
    assert(((NPC == XF_NPPC1) || (NPC == XF_NPPC2) || (NPC == XF_NPPC4) || (NPC == XF_NPPC8)) &&
           "NPC must be XF_NPPC1,XF_NPPC2, XF_NPPC4,XF_NPPC8 ");
#endif
    short width = src1.cols >> XF_BITSHIFT(NPC);

    AddWeightedKernel<SRC_T, DST_T, ROWS, COLS, NPC, XF_CHANNELS(SRC_T, NPC), XF_DEPTH(SRC_T, NPC),
                      XF_DEPTH(DST_T, NPC), XF_WORDWIDTH(SRC_T, NPC), XF_WORDWIDTH(DST_T, NPC),
                      (COLS >> XF_BITSHIFT(NPC))>(src1, alpha, src2, beta, gama, dst, src1.rows, width);
}
} // namespace cv
} // namespace xf
#endif //_XF_ADD_WEIGHTED_HPP_
