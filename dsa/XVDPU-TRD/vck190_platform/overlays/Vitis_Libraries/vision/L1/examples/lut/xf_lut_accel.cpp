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

#include "xf_lut_config.h"

static constexpr int __XF_DEPTH = (HEIGHT * WIDTH * (XF_PIXELWIDTH(TYPE, NPC1)) / 8) / (PTR_WIDTH / 8);
static constexpr int __XF_DEPTH_PTR = (XF_CHANNELS(TYPE, NPC1)) * 256;

void lut_accel(ap_uint<PTR_WIDTH>* img_in, unsigned char* lut_ptr, ap_uint<PTR_WIDTH>* img_out, int height, int width) {
// clang-format off
    #pragma HLS INTERFACE m_axi      port=img_in        offset=slave  bundle=gmem0 depth=__XF_DEPTH
    #pragma HLS INTERFACE m_axi      port=lut_ptr        offset=slave  bundle=gmem1 depth=__XF_DEPTH_PTR
    #pragma HLS INTERFACE m_axi      port=img_out       offset=slave  bundle=gmem2 depth=__XF_DEPTH
    #pragma HLS INTERFACE s_axilite  port=return 			          bundle=control
    // clang-format on

    xf::cv::Mat<TYPE, HEIGHT, WIDTH, NPC1> imgInput(height, width);
    xf::cv::Mat<TYPE, HEIGHT, WIDTH, NPC1> imgOutput(height, width);

    unsigned char lut_buf[256];

    for (int i = 0; i < 256; i++) {
        lut_buf[i] = lut_ptr[i];
    }

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on

    // Retrieve xf::cv::Mat objects from img_in data:
    xf::cv::Array2xfMat<PTR_WIDTH, TYPE, HEIGHT, WIDTH, NPC1>(img_in, imgInput);

    // Run xfOpenCV kernel:
    xf::cv::LUT<TYPE, HEIGHT, WIDTH, NPC1>(imgInput, imgOutput, lut_buf);

    // Convert _dst xf::cv::Mat object to output array:
    xf::cv::xfMat2Array<PTR_WIDTH, TYPE, HEIGHT, WIDTH, NPC1>(imgOutput, img_out);

    return;
} // End of kernel
