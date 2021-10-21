/*
 * Copyright 2020 Xilinx, Inc.
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

#include "xf_aec_config.h"

static bool flag = 0;

static uint32_t histogram1[1][256] = {0};
static uint32_t histogram2[1][256] = {0};

static constexpr int __XF_DEPTH = (HEIGHT * WIDTH * (XF_PIXELWIDTH(IN_TYPE, NPIX)) / 8) / (INPUT_PTR_WIDTH / 8);
static constexpr int __XF_DEPTH_OUT = (HEIGHT * WIDTH * (XF_PIXELWIDTH(OUT_TYPE, NPIX)) / 8) / (OUTPUT_PTR_WIDTH / 8);

void aec_kernel(ap_uint<INPUT_PTR_WIDTH>* src,
                ap_uint<OUTPUT_PTR_WIDTH>* dst,
                int rows,
                int cols,
                uint32_t hist0[1][256],
                uint32_t hist1[1][256]) {
#pragma HLS INLINE OFF
    xf::cv::Mat<IN_TYPE, HEIGHT, WIDTH, NPIX> imgInput(rows, cols);
    xf::cv::Mat<OUT_TYPE, HEIGHT, WIDTH, NPIX> imgOutput(rows, cols);

// clang-format off
#pragma HLS DATAFLOW
    // clang-format on

    xf::cv::Array2xfMat<INPUT_PTR_WIDTH, IN_TYPE, HEIGHT, WIDTH, NPIX>(src, imgInput);

    xf::cv::autoexposurecorrection<IN_TYPE, IN_TYPE, SIN_CHANNEL_TYPE, HEIGHT, WIDTH, NPIX>(imgInput, imgOutput, hist0,
                                                                                            hist1);
    xf::cv::xfMat2Array<OUTPUT_PTR_WIDTH, OUT_TYPE, HEIGHT, WIDTH, NPIX>(imgOutput, dst);
}
void aec_accel(ap_uint<INPUT_PTR_WIDTH>* src, ap_uint<OUTPUT_PTR_WIDTH>* dst, int rows, int cols) {
// clang-format off
#pragma HLS INTERFACE m_axi      port=src        offset=slave  bundle=gmem0 depth=__XF_DEPTH
#pragma HLS INTERFACE m_axi      port=dst       offset=slave  bundle=gmem1 depth=__XF_DEPTH_OUT
#pragma HLS INTERFACE s_axilite  port=return bundle=control
    // clang-format on

    if (!flag) {
        aec_kernel(src, dst, rows, cols, histogram1, histogram2);
        flag = 1;
    } else {
        aec_kernel(src, dst, rows, cols, histogram2, histogram1);
        flag = 0;
    }
}
