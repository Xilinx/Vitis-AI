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

#include "xf_mean_stddev_config.h"

static constexpr int __XF_DEPTH = (HEIGHT * WIDTH * (XF_PIXELWIDTH(TYPE, __NPPC)) / 8) / (PTR_WIDTH / 8);
static constexpr int __XF_DEPTH_MS = XF_CHANNELS(TYPE, __NPPC);

void mean_stddev_accel(
    ap_uint<PTR_WIDTH>* img_in, unsigned short* mean, unsigned short* stddev, int height, int width) {
// clang-format off
    #pragma HLS INTERFACE m_axi      port=img_in        offset=slave  bundle=gmem0 depth=__XF_DEPTH
    #pragma HLS INTERFACE m_axi      port=mean          offset=slave  bundle=gmem1 depth=__XF_DEPTH_MS
    #pragma HLS INTERFACE m_axi      port=stddev        offset=slave  bundle=gmem2 depth=__XF_DEPTH_MS
    #pragma HLS INTERFACE s_axilite  port=height 		      
    #pragma HLS INTERFACE s_axilite  port=width 		      
    #pragma HLS INTERFACE s_axilite  port=return
    // clang-format on

    xf::cv::Mat<TYPE, HEIGHT, WIDTH, __NPPC> imgInput(height, width);

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on

    // Retrieve xf::cv::Mat objects from img_in data:
    xf::cv::Array2xfMat<PTR_WIDTH, TYPE, HEIGHT, WIDTH, __NPPC>(img_in, imgInput);

    // Run xfOpenCV kernel:
    xf::cv::meanStdDev<TYPE, HEIGHT, WIDTH, __NPPC>(imgInput, mean, stddev);

    return;
} // End of kernel
