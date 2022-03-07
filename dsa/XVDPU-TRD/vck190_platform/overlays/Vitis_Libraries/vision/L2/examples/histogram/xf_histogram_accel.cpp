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

#include "xf_histogram_config.h"

extern "C" {

void histogram_accel(ap_uint<PTR_WIDTH>* img_in, unsigned int* histogram, int rows, int cols) {
// clang-format off
    #pragma HLS INTERFACE m_axi      port=img_in       offset=slave     bundle=gmem0
    
    #pragma HLS INTERFACE m_axi      port=histogram    offset=slave     bundle=gmem1
	#pragma HLS INTERFACE s_axilite  port=rows 			          
	#pragma HLS INTERFACE s_axilite  port=cols 			          
    #pragma HLS INTERFACE s_axilite  port=return
    // clang-format on

    xf::cv::Mat<TYPE, HEIGHT, WIDTH, NPC1> imgInput(rows, cols);

// clang-format off
    #pragma HLS STREAM variable=imgInput.data depth=2
// clang-format on

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on

    // Retrieve xf::cv::Mat objects from img_in data:
    xf::cv::Array2xfMat<PTR_WIDTH, TYPE, HEIGHT, WIDTH, NPC1>(img_in, imgInput);

    // Run xfOpenCV kernel:
    xf::cv::calcHist<TYPE, HEIGHT, WIDTH, NPC1>(imgInput, histogram);

    return;
} // End of kernel

} // End of extern C