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

#include "xf_erosion_config.h"

extern "C" {

void erosion(
    ap_uint<PTR_WIDTH>* img_in, unsigned char* process_shape, ap_uint<PTR_WIDTH>* img_out, int height, int width) {
// clang-format off
    #pragma HLS INTERFACE m_axi      port=img_in        offset=slave  bundle=gmem0
    #pragma HLS INTERFACE m_axi      port=process_shape offset=slave  bundle=gmem1
   #pragma HLS INTERFACE m_axi      port=img_out       offset=slave  bundle=gmem2
    #pragma HLS INTERFACE s_axilite  port=height 			          
    #pragma HLS INTERFACE s_axilite  port=width 			          

    #pragma HLS INTERFACE s_axilite  port=return
    // clang-format on

    xf::cv::Mat<TYPE, HEIGHT, WIDTH, NPC1> imgInput(height, width);
    xf::cv::Mat<TYPE, HEIGHT, WIDTH, NPC1> imgOutput(height, width);
#pragma HLS STREAM variable = imgInput.data depth = 2
#pragma HLS STREAM variable = imgOutput.data depth = 2

    // Copy the shape data:
    unsigned char _kernel[FILTER_SIZE * FILTER_SIZE];
    for (unsigned int i = 0; i < FILTER_SIZE * FILTER_SIZE; ++i) {
// clang-format off
        #pragma HLS PIPELINE
        // clang-format on
        _kernel[i] = process_shape[i];
    }

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on
    // Retrieve xf::cv::Mat objects from img_in data:
    xf::cv::Array2xfMat<PTR_WIDTH, TYPE, HEIGHT, WIDTH, NPC1>(img_in, imgInput);

    // Run xfOpenCV kernel:
    xf::cv::erode<XF_BORDER_CONSTANT, TYPE, HEIGHT, WIDTH, KERNEL_SHAPE, FILTER_SIZE, FILTER_SIZE, ITERATIONS, NPC1>(
        imgInput, imgOutput, _kernel);

    // Convert imgOutput xf::cv::Mat object to output array:
    xf::cv::xfMat2Array<PTR_WIDTH, TYPE, HEIGHT, WIDTH, NPC1>(imgOutput, img_out);

    return;
} // End of kernel

} // End of extern C
