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

#include "xf_laplacian_config.h"

extern "C" {

void laplacian(ap_uint<PTR_IN_WIDTH>* img_in,
               short int* filter,
               unsigned char shift,
               ap_uint<PTR_OUT_WIDTH>* img_out,
               int rows,
               int cols) {
// clang-format off
    #pragma HLS INTERFACE m_axi      port=img_in        offset=slave  bundle=gmem0
   
    #pragma HLS INTERFACE m_axi      port=filter        offset=slave  bundle=gmem1
    #pragma HLS INTERFACE m_axi      port=img_out       offset=slave  bundle=gmem2
   
    #pragma HLS INTERFACE s_axilite  port=shift 			          
	#pragma HLS INTERFACE s_axilite  port=rows 			          
	#pragma HLS INTERFACE s_axilite  port=cols 			          
    #pragma HLS INTERFACE s_axilite  port=return
    // clang-format on

    xf::cv::Mat<INTYPE, HEIGHT, WIDTH, NPC1> imgInput(rows, cols);
    xf::cv::Mat<OUTTYPE, HEIGHT, WIDTH, NPC1> imgOutput(rows, cols);

// clang-format off
 #pragma HLS STREAM variable=imgInput.data depth=2
 #pragma HLS STREAM variable=imgOutput.data depth=2
// clang-format on

#pragma HLS DATAFLOW

    // Retrieve xf::cv::Mat objects from img_in data:
    xf::cv::Array2xfMat<PTR_IN_WIDTH, INTYPE, HEIGHT, WIDTH, NPC1>(img_in, imgInput);

    // Run xfOpenCV kernel:
    xf::cv::filter2D<XF_BORDER_CONSTANT, FILTER_WIDTH, FILTER_HEIGHT, INTYPE, OUTTYPE, HEIGHT, WIDTH, NPC1>(
        imgInput, imgOutput, filter, shift);

    // Convert _dst xf::cv::Mat object to output array:
    xf::cv::xfMat2Array<PTR_OUT_WIDTH, OUTTYPE, HEIGHT, WIDTH, NPC1>(imgOutput, img_out);

    return;
} // End of kernel

} // End of extern C
