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

#include "xf_gtm_config.h"

static constexpr int __XF_DEPTH_IN = (HEIGHT * WIDTH * XF_PIXELWIDTH(IN_TYPE, NPC1)) / INPUT_PTR_WIDTH;
static constexpr int __XF_DEPTH_OUT = (HEIGHT * WIDTH * XF_PIXELWIDTH(OUT_TYPE, NPC1)) / OUTPUT_PTR_WIDTH;

static ap_ufixed<16, 4> mean1 = 0;
static ap_ufixed<16, 4> mean2 = 0;
static ap_ufixed<16, 4> L_max1 = 0.1;
static ap_ufixed<16, 4> L_max2 = 0.1;
static ap_ufixed<16, 4> L_min1 = 1;
static ap_ufixed<16, 4> L_min2 = 1;

static bool flag = 0;

void gtm_kernel(ap_uint<INPUT_PTR_WIDTH>* src,
                ap_uint<OUTPUT_PTR_WIDTH>* dst,
                int height,
                int width,
                ap_ufixed<16, 4>& mean1,
                ap_ufixed<16, 4>& mean2,
                ap_ufixed<16, 4>& L_max1,
                ap_ufixed<16, 4>& L_max2,
                ap_ufixed<16, 4>& L_min1,
                ap_ufixed<16, 4>& L_min2,
                float c1,
                float c2) {
// clang-format off
#pragma HLS INLINE OFF
    // clang-format on

    xf::cv::Mat<IN_TYPE, HEIGHT, WIDTH, NPC1> imgInput(height, width);
    xf::cv::Mat<OUT_TYPE, HEIGHT, WIDTH, NPC1> imgOutput(height, width);

// clang-format off
#pragma HLS DATAFLOW
    // clang-format on

    xf::cv::Array2xfMat<INPUT_PTR_WIDTH, IN_TYPE, HEIGHT, WIDTH, NPC1>(src, imgInput);

    gtm<IN_TYPE, OUT_TYPE, SIN_CHANNEL_IN_TYPE, SIN_CHANNEL_OUT_TYPE, HEIGHT, WIDTH, NPC1>(
        imgInput, imgOutput, mean1, mean2, L_max1, L_max2, L_min1, L_min2, c1, c2);

    xf::cv::xfMat2Array<OUTPUT_PTR_WIDTH, OUT_TYPE, HEIGHT, WIDTH, NPC1>(imgOutput, dst);
}

void gtm_accel(
    ap_uint<INPUT_PTR_WIDTH>* in_ptr, ap_uint<OUTPUT_PTR_WIDTH>* out_ptr, float c1, float c2, int height, int width) {
// clang-format off
#pragma HLS INTERFACE m_axi     port=in_ptr  offset=slave bundle=gmem0 depth=__XF_DEPTH_IN
#pragma HLS INTERFACE m_axi     port=out_ptr offset=slave bundle=gmem1 depth=__XF_DEPTH_OUT
#pragma HLS INTERFACE s_axilite port=c1
#pragma HLS INTERFACE s_axilite port=c2
#pragma HLS INTERFACE s_axilite port=height
#pragma HLS INTERFACE s_axilite port=width
#pragma HLS INTERFACE s_axilite port=return
    // clang-format on

    if (!flag) {
        gtm_kernel(in_ptr, out_ptr, height, width, mean1, mean2, L_max1, L_max2, L_min1, L_min2, c1, c2);
        flag = 1;
    } else {
        gtm_kernel(in_ptr, out_ptr, height, width, mean2, mean1, L_max2, L_max1, L_min2, L_min1, c1, c2);
        flag = 0;
    }

    return;
} // End of kernel
