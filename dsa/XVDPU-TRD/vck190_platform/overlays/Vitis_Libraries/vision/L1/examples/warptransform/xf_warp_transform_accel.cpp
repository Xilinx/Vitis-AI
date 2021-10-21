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

#include "xf_warp_transform_config.h"

static constexpr int __XF_DEPTH = (HEIGHT * WIDTH * (XF_PIXELWIDTH(TYPE, NPC1)) / 8) / (PTR_WIDTH / 8);

// extern "C" {

void warp_transform_accel(
    ap_uint<PTR_WIDTH>* img_in, float* transform, ap_uint<PTR_WIDTH>* img_out, int rows, int cols) {
// clang-format off
    #pragma HLS INTERFACE m_axi      port=img_in        offset=slave  bundle=gmem0 depth=__XF_DEPTH
    #pragma HLS INTERFACE m_axi      port=transform     offset=slave  bundle=gmem1 depth=9
    #pragma HLS INTERFACE m_axi      port=img_out       offset=slave  bundle=gmem2 depth=__XF_DEPTH
    #pragma HLS INTERFACE s_axilite  port=rows                    bundle=control
    #pragma HLS INTERFACE s_axilite  port=cols                    bundle=control
    #pragma HLS INTERFACE s_axilite  port=return                    bundle=control
    // clang-format on

    xf::cv::Mat<TYPE, HEIGHT, WIDTH, NPC1> imgInput(rows, cols);
    xf::cv::Mat<TYPE, HEIGHT, WIDTH, NPC1> imgOutput(rows, cols);

// clang-format off
// clang-format on

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on

    // Copy transform data from global memory to local memory:
    float transform_matrix[9];

    for (unsigned int i = 0; i < 9; ++i) {
// clang-format off
        #pragma HLS PIPELINE
        // clang-format on
        transform_matrix[i] = transform[i];
    }

    // Retrieve xf::Mat objects from img_in data:
    xf::cv::Array2xfMat<PTR_WIDTH, TYPE, HEIGHT, WIDTH, NPC1>(img_in, imgInput);

    // Run xfOpenCV kernel:
    xf::cv::warpTransform<NUM_STORE_ROWS, START_PROC, TRANSFORM_TYPE, INTERPOLATION, TYPE, HEIGHT, WIDTH, NPC1,
                          XF_USE_URAM>(imgInput, imgOutput, transform_matrix);

    // Convert _dst xf::Mat object to output array:
    xf::cv::xfMat2Array<PTR_WIDTH, TYPE, HEIGHT, WIDTH, NPC1>(imgOutput, img_out);

    return;
} // End of kernel

//} // End of extern C
