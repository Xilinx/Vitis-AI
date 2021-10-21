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

#ifndef _XF_PRE_PROCESS_
#define _XF_PRE_PROCESS_

#include "ap_int.h"
#include "hls_stream.h"

typedef unsigned short uint16_t;

//#include "common/xf_common.hpp"

namespace xf {
namespace cv {

template <int IN_TYPE,
          int OUT_TYPE,
          int HEIGHT,
          int WIDTH,
          int NPC,
          int WIDTH_A,
          int IBITS_A,
          int WIDTH_B,
          int IBITS_B,
          int WIDTH_OUT,
          int IBITS_OUT>
void xFpreProcessKernel(xf::cv::Mat<IN_TYPE, HEIGHT, WIDTH, NPC>& in_mat,
                        xf::cv::Mat<OUT_TYPE, HEIGHT, WIDTH, NPC>& out_mat,
                        ap_ufixed<WIDTH_A, IBITS_A, AP_RND> alpha_reg[XF_CHANNELS(IN_TYPE, NPC)],
                        ap_fixed<WIDTH_B, IBITS_B, AP_RND> beta_reg[XF_CHANNELS(IN_TYPE, NPC)],
                        int loop_count) {
    XF_CTUNAME(IN_TYPE, NPC) x_1pix;

    XF_TNAME(OUT_TYPE, NPC) out_pack;
    for (int k = 0; k < loop_count; k++) {
// clang-format off
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=608*608
        // clang-format on
        XF_TNAME(IN_TYPE, NPC) x_pack = in_mat.read(k);

        for (int i = 0; i < XF_NPIXPERCYCLE(NPC); i++) {
// clang-format off
#pragma HLS UNROLL
            // clang-format on
            for (int j = 0; j < XF_CHANNELS(IN_TYPE, NPC); j++) {
// clang-format off
#pragma HLS UNROLL
                // clang-format on
                x_1pix = x_pack.range((j * XF_DTPIXELDEPTH(IN_TYPE, NPC)) + (XF_DTPIXELDEPTH(IN_TYPE, NPC) - 1) +
                                          (i * XF_CHANNELS(IN_TYPE, NPC) * XF_DTPIXELDEPTH(IN_TYPE, NPC)),
                                      (j * XF_DTPIXELDEPTH(IN_TYPE, NPC)) +
                                          (i * XF_CHANNELS(IN_TYPE, NPC) * XF_DTPIXELDEPTH(IN_TYPE, NPC)));

                ap_ufixed<WIDTH_A, IBITS_A, AP_RND> a = alpha_reg[j];
                ap_fixed<WIDTH_B, IBITS_B, AP_RND> b = beta_reg[j];

                ap_fixed<WIDTH_OUT, IBITS_OUT, AP_RND> out_1pix;

                out_1pix = (x_1pix - a) * b;

                ap_uint<XF_DTPIXELDEPTH(OUT_TYPE, NPC)>* out_val;

                out_val = (ap_uint<XF_DTPIXELDEPTH(OUT_TYPE, NPC)>*)&out_1pix;

                out_pack.range((j * XF_DTPIXELDEPTH(OUT_TYPE, NPC)) + (XF_DTPIXELDEPTH(OUT_TYPE, NPC) - 1) +
                                   (i * XF_CHANNELS(OUT_TYPE, NPC) * XF_DTPIXELDEPTH(OUT_TYPE, NPC)),
                               (j * XF_DTPIXELDEPTH(OUT_TYPE, NPC)) +
                                   (i * XF_CHANNELS(OUT_TYPE, NPC) * XF_DTPIXELDEPTH(OUT_TYPE, NPC))) = *out_val;
            }
        }
        out_mat.write(k, out_pack);
    }
}

template <int IN_TYPE,
          int OUT_TYPE,
          int HEIGHT,
          int WIDTH,
          int NPC,
          int WIDTH_A,
          int IBITS_A,
          int WIDTH_B,
          int IBITS_B,
          int WIDTH_OUT,
          int IBITS_OUT>
void preProcess(xf::cv::Mat<IN_TYPE, HEIGHT, WIDTH, NPC>& in_mat,
                xf::cv::Mat<OUT_TYPE, HEIGHT, WIDTH, NPC>& out_mat,
                float params[2 * XF_CHANNELS(IN_TYPE, NPC)]) {
#pragma HLS INLINE OFF

    ap_ufixed<WIDTH_A, IBITS_A, AP_RND> alpha_reg[XF_CHANNELS(IN_TYPE, NPC)];
    ap_fixed<WIDTH_B, IBITS_B, AP_RND> beta_reg[XF_CHANNELS(IN_TYPE, NPC)];

// clang-format off
#pragma HLS ARRAY_PARTITION variable=alpha_reg dim=0 complete
#pragma HLS ARRAY_PARTITION variable=beta_reg dim=0 complete

    // clang-format on
    int channels = XF_CHANNELS(IN_TYPE, NPC);
    for (int i = 0; i < 2 * XF_CHANNELS(IN_TYPE, NPC); i++) {
// clang-format off
#pragma HLS LOOP_TRIPCOUNT min=1 max=12
#pragma HLS PIPELINE II=1
        // clang-format on
        float temp = params[i];
        if (i < 3)
            alpha_reg[i] = temp;
        else
            beta_reg[i - XF_CHANNELS(IN_TYPE, NPC)] = temp;
    }

    // clang-format off
//#pragma HLS DATAFLOW
    // clang-format on

    uint16_t width = in_mat.cols >> XF_BITSHIFT(NPC);
    uint16_t height = in_mat.rows;

    int loop_count = width * height;

    xFpreProcessKernel<IN_TYPE, OUT_TYPE, HEIGHT, WIDTH, NPC, WIDTH_A, IBITS_A, WIDTH_B, IBITS_B, WIDTH_OUT, IBITS_OUT>(
        in_mat, out_mat, alpha_reg, beta_reg, loop_count);
}
}
}
#endif
