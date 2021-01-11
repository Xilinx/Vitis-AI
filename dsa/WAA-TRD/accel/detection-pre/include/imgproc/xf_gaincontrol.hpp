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

#ifndef _XF_GC_HPP_
#define _XF_GC_HPP_

#include "hls_stream.h"
#include "common/xf_common.hpp"

#ifndef XF_IN_STEP
#define XF_IN_STEP 8
#endif
#ifndef XF_OUT_STEP
#define XF_OUT_STEP 8
#endif

#define R_GAIN 140
#define B_GAIN 140

namespace xf {

namespace cv {

template <int BFORMAT,
          int SRC_T,
          int ROWS,
          int COLS,
          int NPC,
          int PLANES,
          int DEPTH_SRC,
          int DEPTH_DST,
          int WORDWIDTH_SRC,
          int WORDWIDTH_DST,
          int TC>
void gaincontrolkernel(xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& src1,
                       xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& dst,
                       uint16_t height,
                       uint16_t width) {
    ap_uint<13> i, j, k, l;

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

        ProcLoop:
            for (k = 0, l = 0; k < ((8 << XF_BITSHIFT(NPC)) * PLANES); k += XF_IN_STEP, l++) {
                XF_PTNAME(DEPTH_SRC) pxl1 = pxl_pack1.range(k + 7, k); // extracting each pixel in case of 8-pixel mode
                XF_PTNAME(DEPTH_SRC) t;
                bool cond1 = 0, cond2 = 0;

                if (NPC == XF_NPPC1) {
                    cond1 = (j % 2 == 0);
                    cond2 = (j % 2 != 0);
                } else {
                    cond1 = ((l % 2) == 0);
                    cond2 = ((l % 2) != 0);
                }

                if (BFORMAT == XF_BAYER_RG) {
                    if (i % 2 == 0 && cond1) {
                        XF_PTNAME(DEPTH_SRC) v1 = pxl1;
                        short v2 = (short)((v1 * R_GAIN) >> 7);
                        t = (v2 > 255) ? 255 : v2;
                    } else if (i % 2 != 0 && cond2) {
                        XF_PTNAME(DEPTH_SRC) v1 = pxl1;
                        short v2 = (short)((v1 * B_GAIN) >> 7);
                        t = (v2 > 255) ? 255 : v2;
                    } else {
                        t = pxl1;
                    }
                }
                if (BFORMAT == XF_BAYER_GR) {
                    if (i % 2 == 0 && cond2) {
                        XF_PTNAME(DEPTH_SRC) v1 = pxl1;
                        short v2 = (short)((v1 * R_GAIN) >> 7);
                        t = (v2 > 255) ? 255 : v2;
                    } else if (i % 2 != 0 && cond1) {
                        XF_PTNAME(DEPTH_SRC) v1 = pxl1;
                        short v2 = (short)((v1 * B_GAIN) >> 7);
                        t = (v2 > 255) ? 255 : v2;
                    } else {
                        t = pxl1;
                    }
                }
                if (BFORMAT == XF_BAYER_BG) {
                    if (i % 2 == 0 && cond1) {
                        XF_PTNAME(DEPTH_SRC) v1 = pxl1;
                        short v2 = (short)((v1 * B_GAIN) >> 7);
                        t = (v2 > 255) ? 255 : v2;
                    } else if (i % 2 == 0 && cond2) {
                        XF_PTNAME(DEPTH_SRC) v1 = pxl1;
                        short v2 = (short)((v1 * R_GAIN) >> 7);
                        t = (v2 > 255) ? 255 : v2;
                    } else {
                        t = pxl1;
                    }
                }
                if (BFORMAT == XF_BAYER_GB) {
                    if (i % 2 == 0 && cond2) {
                        XF_PTNAME(DEPTH_SRC) v1 = pxl1;
                        short v2 = (short)((v1 * B_GAIN) >> 7);
                        t = (v2 > 255) ? 255 : v2;
                    } else if (i % 2 != 0 && cond1) {
                        XF_PTNAME(DEPTH_SRC) v1 = pxl1;
                        short v2 = (short)((v1 * R_GAIN) >> 7);
                        t = (v2 > 255) ? 255 : v2;
                    } else {
                        t = pxl1;
                    }
                }

                pxl_pack_out.range(k + XF_OUT_STEP - 1, k) = t;
            }

            dst.write(i * width + j, (XF_SNAME(WORDWIDTH_DST))pxl_pack_out); // writing into ouput stream
        }
    }
}

template <int BFORMAT, int SRC_T, int ROWS, int COLS, int NPC = 1>
void gaincontrol(xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& src1, xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& dst) {
// clang-format off
#pragma HLS INLINE OFF
// clang-format on
#ifndef __SYNTHESIS__
    assert(((src1.rows == dst.rows) && (src1.cols == dst.cols)) && "Input and output image should be of same size");
    assert(((src1.rows <= ROWS) && (src1.cols <= COLS)) && "ROWS and COLS should be greater than input image");
#endif
    short width = src1.cols >> XF_BITSHIFT(NPC);

    gaincontrolkernel<BFORMAT, SRC_T, ROWS, COLS, NPC, XF_CHANNELS(SRC_T, NPC), XF_DEPTH(SRC_T, NPC),
                      XF_DEPTH(SRC_T, NPC), XF_WORDWIDTH(SRC_T, NPC), XF_WORDWIDTH(SRC_T, NPC),
                      (COLS >> XF_BITSHIFT(NPC))>(src1, dst, src1.rows, width);
}
}
}

#endif //_XF_GC_HPP_
