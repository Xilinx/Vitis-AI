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

#include "xf_inrange_config.h"

static constexpr int __XF_DEPTH = (HEIGHT * WIDTH * (XF_PIXELWIDTH(IN_TYPE, NPC1)) / 8) / (INPUT_PTR_WIDTH / 8);

void inrange_accel(ap_uint<INPUT_PTR_WIDTH>* img_in,
                   unsigned char lower_thresh,
                   unsigned char upper_thresh,
                   ap_uint<OUTPUT_PTR_WIDTH>* img_out,
                   int height,
                   int width) {
// clang-format off
    #pragma HLS INTERFACE m_axi      port=img_in        offset=slave  bundle=gmem0	depth=__XF_DEPTH
    #pragma HLS INTERFACE m_axi      port=img_out       offset=slave  bundle=gmem1	depth=__XF_DEPTH
     #pragma HLS INTERFACE s_axilite  port=return 		      bundle=control
    // clang-format on

    xf::cv::Mat<IN_TYPE, HEIGHT, WIDTH, NPC1> imgInput(height, width);
    xf::cv::Mat<OUT_TYPE, HEIGHT, WIDTH, NPC1> imgOutput(height, width);

// clang-format off

// clang-format on

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on

    // Copy threshold to local memory:
    unsigned char local_low_thresh[XF_CHANNELS(IN_TYPE, NPC1)];
    unsigned char local_high_thresh[XF_CHANNELS(IN_TYPE, NPC1)];

    for (unsigned int i = 0; i < XF_CHANNELS(IN_TYPE, NPC1); ++i) {
        local_low_thresh[i] = lower_thresh;
        local_high_thresh[i] = upper_thresh;
    }

    // Retrieve xf::cv::Mat objects from img_in data:
    xf::cv::Array2xfMat<INPUT_PTR_WIDTH, IN_TYPE, HEIGHT, WIDTH, NPC1>(img_in, imgInput);

    // Run xfOpenCV kernel:
    xf::cv::inRange<IN_TYPE, OUT_TYPE, HEIGHT, WIDTH, NPC1>(imgInput, local_low_thresh, local_high_thresh, imgOutput);

    // Convert imgOutput xf::cv::Mat object to output array:
    xf::cv::xfMat2Array<OUTPUT_PTR_WIDTH, OUT_TYPE, HEIGHT, WIDTH, NPC1>(imgOutput, img_out);

    return;
} // End of kernel
