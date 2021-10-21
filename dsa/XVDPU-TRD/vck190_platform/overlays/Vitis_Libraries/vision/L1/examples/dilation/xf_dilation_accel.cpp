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

#include "xf_dilation_config.h"

static constexpr int __XF_DEPTH = (HEIGHT * WIDTH * (XF_PIXELWIDTH(TYPE, NPC_T)) / 8) / (INPUT_PTR_WIDTH / 8);

void dilation_accel(ap_uint<INPUT_PTR_WIDTH>* img_inp,
                    ap_uint<OUTPUT_PTR_WIDTH>* img_out,
                    unsigned char* kernel,
                    int height,
                    int width) {
// clang-format off
    #pragma HLS INTERFACE m_axi     port=img_inp  offset=slave bundle=gmem1 depth=__XF_DEPTH
    #pragma HLS INTERFACE m_axi     port=img_out  offset=slave bundle=gmem2 depth=__XF_DEPTH
    #pragma HLS INTERFACE m_axi     port=kernel  offset=slave bundle=gmem3  depth=9
    #pragma HLS INTERFACE s_axilite port=return
    // clang-format on

    xf::cv::Mat<TYPE, HEIGHT, WIDTH, NPC_T> in_mat(height, width);
    xf::cv::Mat<TYPE, HEIGHT, WIDTH, NPC_T> out_mat(height, width);

    unsigned char locKernel[FILTER_SIZE * FILTER_SIZE];
    for (int ki = 0; ki < (FILTER_SIZE * FILTER_SIZE); ki++) {
// clang-format off
        #pragma HLS PIPELINE
        // clang-format on
        locKernel[ki] = kernel[ki];
    }

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on

    xf::cv::Array2xfMat<INPUT_PTR_WIDTH, TYPE, HEIGHT, WIDTH, NPC_T>(img_inp, in_mat);
    xf::cv::dilate<XF_BORDER_CONSTANT, TYPE, HEIGHT, WIDTH, KERNEL_SHAPE, FILTER_SIZE, FILTER_SIZE, ITERATIONS, NPC_T>(
        in_mat, out_mat, kernel);
    xf::cv::xfMat2Array<OUTPUT_PTR_WIDTH, TYPE, HEIGHT, WIDTH, NPC_T>(out_mat, img_out);
}
