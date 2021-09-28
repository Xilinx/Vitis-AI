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

extern "C" {

void inrange_accel(ap_uint<PTR_IN_WIDTH>* img_in,
                   unsigned char lower_thresh,
                   unsigned char upper_thresh,
                   ap_uint<PTR_OUT_WIDTH>* img_out,
                   int height,
                   int width) {
// clang-format off
    #pragma HLS INTERFACE m_axi      port=img_in        offset=slave  bundle=gmem0
    #pragma HLS INTERFACE m_axi      port=img_out       offset=slave  bundle=gmem1
    #pragma HLS INTERFACE s_axilite  port=lower_thresh 		      
    #pragma HLS INTERFACE s_axilite  port=upper_thresh 		      
    #pragma HLS INTERFACE s_axilite  port=height 		      
    #pragma HLS INTERFACE s_axilite  port=width 		      
    #pragma HLS INTERFACE s_axilite  port=return
    // clang-format on

    xf::cv::Mat<IN_TYPE, HEIGHT, WIDTH, NPC1> imgInput(height, width);
    xf::cv::Mat<OUT_TYPE, HEIGHT, WIDTH, NPC1> imgOutput(height, width);

// clang-format off
    #pragma HLS STREAM variable=imgInput.data depth=2
    #pragma HLS STREAM variable=imgOutput.data depth=2
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
    xf::cv::Array2xfMat<PTR_IN_WIDTH, IN_TYPE, HEIGHT, WIDTH, NPC1>(img_in, imgInput);

    // Run xfOpenCV kernel:
    xf::cv::inRange<IN_TYPE, OUT_TYPE, HEIGHT, WIDTH, NPC1>(imgInput, local_low_thresh, local_high_thresh, imgOutput);

    // Convert imgOutput xf::cv::Mat object to output array:
    xf::cv::xfMat2Array<PTR_OUT_WIDTH, OUT_TYPE, HEIGHT, WIDTH, NPC1>(imgOutput, img_out);

    return;
} // End of kernel

} // End of extern C
