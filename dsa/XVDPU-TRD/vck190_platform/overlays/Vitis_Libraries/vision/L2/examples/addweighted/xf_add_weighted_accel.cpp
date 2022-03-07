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

#include "xf_add_weighted_config.h"

extern "C" {

void addweighted(ap_uint<PTR_IN_WIDTH>* img_in1,
                 ap_uint<PTR_IN_WIDTH>* img_in2,
                 float alpha,
                 float beta,
                 float gamma,
                 ap_uint<PTR_OUT_WIDTH>* img_out,
                 int height,
                 int width) {
// clang-format off
    #pragma HLS INTERFACE m_axi      port=img_in1       offset=slave  bundle=gmem0

    #pragma HLS INTERFACE m_axi      port=img_in2       offset=slave  bundle=gmem1

    #pragma HLS INTERFACE m_axi      port=img_out       offset=slave  bundle=gmem2
    #pragma HLS INTERFACE s_axilite  port=alpha 			          
    #pragma HLS INTERFACE s_axilite  port=beta 			                  
    #pragma HLS INTERFACE s_axilite  port=gamma 			          
    #pragma HLS INTERFACE s_axilite  port=height 			          
    #pragma HLS INTERFACE s_axilite  port=width 			          
    #pragma HLS INTERFACE s_axilite  port=return
    // clang-format on

    xf::cv::Mat<IN_TYPE, HEIGHT, WIDTH, NPC1> imgInput1(height, width);
    xf::cv::Mat<IN_TYPE, HEIGHT, WIDTH, NPC1> imgInput2(height, width);
    xf::cv::Mat<OUT_TYPE, HEIGHT, WIDTH, NPC1> imgOutput(height, width);

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on

    // Retrieve xf::cv::Mat objects from img_in data:
    xf::cv::Array2xfMat<PTR_IN_WIDTH, IN_TYPE, HEIGHT, WIDTH, NPC1>(img_in1, imgInput1);
    xf::cv::Array2xfMat<PTR_IN_WIDTH, IN_TYPE, HEIGHT, WIDTH, NPC1>(img_in2, imgInput2);

    // Run xfOpenCV kernel:
    xf::cv::addWeighted<IN_TYPE, OUT_TYPE, HEIGHT, WIDTH, NPC1>(imgInput1, alpha, imgInput2, beta, gamma, imgOutput);

    // Convert _dst xf::cv::Mat object to output array:
    xf::cv::xfMat2Array<PTR_OUT_WIDTH, OUT_TYPE, HEIGHT, WIDTH, NPC1>(imgOutput, img_out);

    return;
} // End of kernel

} // End of extern C
