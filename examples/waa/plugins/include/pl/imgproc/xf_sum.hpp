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
template <int SRC_T, int ROWS, int COLS, int NPC, int PLANES, int DEPTH_SRC, int WORDWIDTH_SRC, int TC>
int sumKernel(xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& src1,
              xf::cv::Scalar<XF_CHANNELS(SRC_T, NPC), double>& scl,
              uint16_t height,
              uint16_t width) {
    ap_uint<13> i, j, k, l, c;

    ap_uint<64> internal_sum[PLANES];
    for (int i = 0; i < PLANES; i++) {
// clang-format off
        #pragma HLS unroll
        // clang-format on
        internal_sum[i] = 0;
    }

    int STEP = XF_PIXELDEPTH(DEPTH_SRC) / PLANES;

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

            pxl_pack1 = (XF_SNAME(WORDWIDTH_SRC))(src1.read(i * width + j));
        ProcLoop:
            for (k = 0, c = 0; k < ((8 << XF_BITSHIFT(NPC)) * PLANES); k += XF_IN_STEP, c++) {
                XF_PTNAME(DEPTH_SRC) pxl1 = pxl_pack1.range(k + 7, k);

                if (PLANES == 1) {
                    internal_sum[0] = internal_sum[0] + pxl1;
                } else {
                    internal_sum[c] = internal_sum[c] + pxl1;
                }
            }
        }
    }
    if (PLANES == 1) {
        scl.val[0] = (ap_uint<64>)internal_sum[0];
    } else {
        scl.val[0] = (ap_uint<64>)internal_sum[0];
        scl.val[1] = (ap_uint<64>)internal_sum[1];
        scl.val[2] = (ap_uint<64>)internal_sum[2];
    }
    return 0;
}

template <int SRC_T, int ROWS, int COLS, int NPC = 1>
void sum(xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& src1, double sum[XF_CHANNELS(SRC_T, NPC)]) {
#ifndef __SYNTHESIS__
    assert(((SRC_T == XF_8UC1)) && "Input TYPE must be XF_8UC1 for 1-channel image");
    assert(((src1.rows <= ROWS) && (src1.cols <= COLS)) && "ROWS and COLS should be greater than input image");
    assert(((NPC == XF_NPPC1) || (NPC == XF_NPPC8)) && "NPC must be XF_NPPC1, XF_NPPC8 ");
#endif
    short width = src1.cols >> XF_BITSHIFT(NPC);
    xf::cv::Scalar<XF_CHANNELS(SRC_T, NPC), double> scl;

    sumKernel<SRC_T, ROWS, COLS, NPC, XF_CHANNELS(SRC_T, NPC), XF_DEPTH(SRC_T, NPC), XF_WORDWIDTH(SRC_T, NPC),
              (COLS >> XF_BITSHIFT(NPC))>(src1, scl, src1.rows, width);
    for (int i = 0; i < XF_CHANNELS(SRC_T, NPC); i++) {
        sum[i] = scl.val[i];
    }
}
} // namespace cv
} // namespace xf
#endif //_XF_SUM_HPP_
