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

#ifndef _XF_ACCUMULATE_WEIGHTED_HPP_
#define _XF_ACCUMULATE_WEIGHTED_HPP_

#include "hls_stream.h"
#include "common/xf_common.hpp"

#ifndef XF_IN_STEP
#define XF_IN_STEP 8
#endif
#ifndef XF_OUT_STEP
#define XF_OUT_STEP 16
#endif
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
int AccumulateWeightedKernel(xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& src1,
                             xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& src2,
                             xf::cv::Mat<DST_T, ROWS, COLS, NPC>& dst,
                             float alpha,
                             uint16_t height,
                             uint16_t width) {
    ap_uint<13> i, j, k, l;
    ap_uint<24> temp = (alpha * ((1 << 23) - 1));
    ap_uint<24> temp1 = ((1 << 23) - 1) - temp + 1;

    XF_TNAME(DST_T, NPC) pxl_pack_out;
    XF_TNAME(SRC_T, NPC) pxl_pack1, pxl_pack2;
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

            pxl_pack1 = (XF_TNAME(SRC_T, NPC))src1.read(i * width + j);
            pxl_pack2 = (XF_TNAME(SRC_T, NPC))src2.read(i * width + j);
        ProcLoop:
            for (k = 0, l = 0; k < ((8 << XF_BITSHIFT(NPC)) * PLANES); k += XF_IN_STEP, l += XF_OUT_STEP) {
                XF_CTUNAME(SRC_T, NPC) pxl1 = pxl_pack1.range(k + 7, k);
                XF_CTUNAME(SRC_T, NPC) pxl2 = pxl_pack2.range(k + 7, k);

                ap_uint<40> firstcmp = pxl1 * temp;
                ap_uint<40> secondcmp = pxl2 * temp1;

                XF_CTUNAME(DST_T, NPC) t = (firstcmp + secondcmp) >> 23;

                pxl_pack_out.range(l + XF_OUT_STEP - 1, l) = t;
            }

            dst.write(i * width + j, (XF_TNAME(DST_T, NPC))pxl_pack_out);
        }
    }
    return 0;
}

template <int SRC_T, int DST_T, int ROWS, int COLS, int NPC = 1>
void accumulateWeighted(xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& src1,
                        xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& src2,
                        xf::cv::Mat<DST_T, ROWS, COLS, NPC>& dst,
                        float alpha) {
#ifndef __SYNTHESIS__
    assert(((SRC_T == XF_8UC1) || (SRC_T == XF_8UC3)) &&
           "Input TYPE must be XF_8UC1 for 1-channel and XF_8UC3 for 3-channel image");
    assert(((DST_T == XF_16UC1) || (DST_T == XF_16UC3)) &&
           "Output TYPE must be XF_16UC1 for 1-channel and XF_16UC3 for 3-channel image");
    assert(((src1.rows == src2.rows) && (src1.cols == src2.cols)) && "Both input images should have same size");
    assert(((src1.rows == dst.rows) && (src1.cols == dst.cols)) && "Input and output image should be of same size");
    assert(((src1.rows <= ROWS) && (src1.cols <= COLS)) && "ROWS and COLS should be greater than input image");
    assert(((NPC == XF_NPPC1) || (NPC == XF_NPPC8)) && "NPC must be XF_NPPC1, XF_NPPC8 ");
#endif
    short width = src1.cols >> XF_BITSHIFT(NPC);

    AccumulateWeightedKernel<SRC_T, DST_T, ROWS, COLS, NPC, XF_CHANNELS(SRC_T, NPC), XF_DEPTH(SRC_T, NPC),
                             XF_DEPTH(DST_T, NPC), XF_WORDWIDTH(SRC_T, NPC), XF_WORDWIDTH(DST_T, NPC),
                             (COLS >> XF_BITSHIFT(NPC))>(src1, src2, dst, alpha, src1.rows, width);
}
} // namespace cv
} // namespace xf
#endif //_XF_ACCUMULATE_WEIGHTED_HPP_
