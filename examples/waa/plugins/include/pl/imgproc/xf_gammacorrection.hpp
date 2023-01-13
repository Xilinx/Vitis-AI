/*
 * Copyright 2021 Xilinx, Inc.
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

#ifndef _XF_GAMMACORRECT_HPP_
#define _XF_GAMMACORRECT_HPP_

#include "hls_stream.h"
#include "common/xf_common.hpp"
#include "xf_lut.hpp"
#include "hls_math.h"

namespace xf {
namespace cv {
template <int SRC_T,
          int ROWS,
          int COLS,
          int PLANES,
          int DEPTH,
          int NPC,
          int WORDWIDTH_SRC,
          int WORDWIDTH_DST,
          int COLS_TRIP>
void xFGAMMAKernel(xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _src,
                   xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _dst,
                   unsigned char lut_table[256 * PLANES],
                   uint16_t height,
                   uint16_t width) {
    // width = width >> XF_BITSHIFT(NPC);
    ap_uint<13> i, j, k;
    const int STEP = XF_DTPIXELDEPTH(SRC_T, NPC);

    unsigned char lut_p[XF_NPIXPERCYCLE(NPC) * PLANES]
                       [256]; //, lut_1[XF_NPIXPERCYCLE(NPC)][256], lut_2[XF_NPIXPERCYCLE(NPC)][256];

// clang-format off
        #pragma HLS ARRAY_PARTITION variable=lut_p complete dim=1
    // clang-format on

    // creating a temporary buffers for Resource Optimization and Performance optimization

    for (i = 0; i < (XF_NPIXPERCYCLE(NPC)); i++) {
        for (k = 0; k < PLANES; k++) {
            for (j = 0; j < 256; j++) {
                lut_p[i * PLANES + k][j] = lut_table[256 * k + j];
            }
        }
    }

    XF_TNAME(SRC_T, NPC) val_src;
    XF_TNAME(SRC_T, NPC) val_dst;

    printf("gamma correction width:%d height%d\n", width, height);

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

            val_src = _src.read(i * width + j);

            for (int p = 0, b = 0; p < XF_NPIXPERCYCLE(NPC) * PLANES; p++, b = b + 3) {
// clang-format off
#pragma HLS unroll
                // clang-format on
                XF_CTUNAME(SRC_T, NPC) val1 = val_src.range(p * STEP + STEP - 1, p * STEP);

                XF_CTUNAME(SRC_T, NPC) outval1 = lut_p[p][val1];

                val_dst.range(p * STEP + STEP - 1, p * STEP) = outval1;
            }

            _dst.write(i * width + j, val_dst);
        }
    }
}
template <int SRC_T, int DST_T, int ROWS, int COLS, int NPC = 1>
void gammacorrection(xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& src,
                     xf::cv::Mat<DST_T, ROWS, COLS, NPC>& dst,
                     unsigned char lut_table[256 * XF_CHANNELS(SRC_T, NPC)]) {
// clang-format off
#pragma HLS INLINE OFF
    // clang-format on

    unsigned short height = src.rows;
    unsigned short width = src.cols >> XF_BITSHIFT(NPC);

    xFGAMMAKernel<SRC_T, ROWS, COLS, XF_CHANNELS(SRC_T, NPC), XF_DEPTH(SRC_T, NPC), NPC, XF_WORDWIDTH(SRC_T, NPC),
                  XF_WORDWIDTH(SRC_T, NPC), (COLS >> XF_BITSHIFT(NPC))>(src, dst, lut_table, height, width);
}
}
}

#endif
