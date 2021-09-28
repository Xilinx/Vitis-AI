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

#include "xf_channel_combine_config.h"

extern "C" {

void channel_combine(ap_uint<PTR_IN_WIDTH>* img_in1,
                     ap_uint<PTR_IN_WIDTH>* img_in2,
#if !TWO_INPUT
                     ap_uint<PTR_IN_WIDTH>* img_in3,
#endif
#if FOUR_INPUT
                     ap_uint<PTR_IN_WIDTH>* img_in4,
#endif
                     ap_uint<PTR_OUT_WIDTH>* img_out,
                     int height,
                     int width) {
// clang-format off
    #pragma HLS INTERFACE m_axi      port=img_in1       offset=slave  bundle=gmem0
    #pragma HLS INTERFACE m_axi      port=img_in2       offset=slave  bundle=gmem1
#if !TWO_INPUT
    #pragma HLS INTERFACE m_axi      port=img_in3       offset=slave  bundle=gmem2
#endif
#if FOUR_INPUT
    #pragma HLS INTERFACE m_axi      port=img_in4       offset=slave  bundle=gmem3
#endif
    #pragma HLS INTERFACE m_axi      port=img_out       offset=slave  bundle=gmem4
    #pragma HLS interface s_axilite  port=height	              
    #pragma HLS interface s_axilite  port=width 	              
    #pragma HLS interface s_axilite  port=return
    // clang-format on

    xf::cv::Mat<IN_TYPE, HEIGHT, WIDTH, NPC1> imgInput1(height, width);
    xf::cv::Mat<IN_TYPE, HEIGHT, WIDTH, NPC1> imgInput2(height, width);
#if !TWO_INPUT
    xf::cv::Mat<IN_TYPE, HEIGHT, WIDTH, NPC1> imgInput3(height, width);
#endif
#if FOUR_INPUT
    xf::cv::Mat<IN_TYPE, HEIGHT, WIDTH, NPC1> imgInput4(height, width);
#endif
    xf::cv::Mat<OUT_TYPE, HEIGHT, WIDTH, NPC1> imgOutput(height, width);

    const int sdepth = 2;

// clang-format off
    #pragma HLS STREAM variable=imgInput1.data depth=sdepth
    #pragma HLS STREAM variable=imgInput2.data depth=sdepth
#if !TWO_INPUT
    #pragma HLS STREAM variable=imgInput3.data depth=sdepth
#endif
#if FOUR_INPUT
    #pragma HLS STREAM variable=imgInput4.data depth=sdepth
#endif
    #pragma HLS STREAM variable=imgOutput.data depth=sdepth
// clang-format on

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on

    // Retrieve xf::cv::Mat objects from img_in data:
    xf::cv::Array2xfMat<PTR_IN_WIDTH, IN_TYPE, HEIGHT, WIDTH, NPC1>(img_in1, imgInput1);
    xf::cv::Array2xfMat<PTR_IN_WIDTH, IN_TYPE, HEIGHT, WIDTH, NPC1>(img_in2, imgInput2);
#if !TWO_INPUT
    xf::cv::Array2xfMat<PTR_IN_WIDTH, IN_TYPE, HEIGHT, WIDTH, NPC1>(img_in3, imgInput3);
#endif
#if FOUR_INPUT
    xf::cv::Array2xfMat<PTR_IN_WIDTH, IN_TYPE, HEIGHT, WIDTH, NPC1>(img_in4, imgInput4);
#endif
    // Run xfOpenCV kernel:
    xf::cv::merge<IN_TYPE, OUT_TYPE, HEIGHT, WIDTH, NPC1>(imgInput1, imgInput2,
#if !TWO_INPUT
                                                          imgInput3,
#endif
#if FOUR_INPUT
                                                          imgInput4,
#endif
                                                          imgOutput);

    // Convert imgOutput xf::cv::Mat object to output array:
    xf::cv::xfMat2Array<PTR_OUT_WIDTH, OUT_TYPE, HEIGHT, WIDTH, NPC1>(imgOutput, img_out);

    return;

} // End of kernel

} // End of extern C
