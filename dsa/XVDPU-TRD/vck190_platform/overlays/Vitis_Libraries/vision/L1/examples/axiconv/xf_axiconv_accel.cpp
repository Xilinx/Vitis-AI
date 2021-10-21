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

#include "xf_axiconv_config.h"

void axiconv_accel(hls::stream<ap_axiu<8, 1, 1, 1> >& src, hls::stream<ap_axiu<8, 1, 1, 1> >& dst, int rows, int cols) {
// clang-format off
    #pragma HLS INTERFACE axis port=src
    #pragma HLS INTERFACE axis port=dst
    #pragma HLS INTERFACE s_axilite port=rows               
    #pragma HLS INTERFACE s_axilite port=cols               
    #pragma HLS INTERFACE s_axilite port=return
    // clang-format on

    xf::cv::Mat<XF_8UC1, XF_HEIGHT, XF_WIDTH, XF_NPPC1> src_mat(rows, cols);
// clang-format off
    #pragma HLS dataflow
    // clang-format on

    xf::cv::AXIvideo2xfMat(src, src_mat);
    xf::cv::xfMat2AXIvideo(src_mat, dst);

    return;
}
