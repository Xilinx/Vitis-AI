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

static constexpr int __XF_DEPTH = (HEIGHT * WIDTH * (XF_PIXELWIDTH(IN_TYPE, NPC1)) / 8) / (INPUT_PTR_WIDTH / 8);
static constexpr int __XF_DEPTH_OUT = (HEIGHT * WIDTH * (XF_PIXELWIDTH(OUT_TYPE, NPC1)) / 8) / (OUTPUT_PTR_WIDTH / 8);

#if FOUR_INPUT
void channel_combine_accel(ap_uint<INPUT_PTR_WIDTH>* img_in1,
                           ap_uint<INPUT_PTR_WIDTH>* img_in2,
                           ap_uint<INPUT_PTR_WIDTH>* img_in3,
                           ap_uint<INPUT_PTR_WIDTH>* img_in4,
                           ap_uint<OUTPUT_PTR_WIDTH>* img_out,
                           int height,
                           int width) {
// clang-format off
    #pragma HLS INTERFACE m_axi      port=img_in1       offset=slave  bundle=gmem0 depth=__XF_DEPTH
    #pragma HLS INTERFACE m_axi      port=img_in2       offset=slave  bundle=gmem1 depth=__XF_DEPTH
    #pragma HLS INTERFACE m_axi      port=img_in3       offset=slave  bundle=gmem2 depth=__XF_DEPTH
    #pragma HLS INTERFACE m_axi      port=img_in4       offset=slave  bundle=gmem3 depth=__XF_DEPTH
    #pragma HLS INTERFACE m_axi      port=img_out       offset=slave  bundle=gmem4 depth=__XF_DEPTH_OUT
    #pragma HLS interface s_axilite  port=height	              
    #pragma HLS interface s_axilite  port=width 	              
    #pragma HLS interface s_axilite  port=return
    // clang-format on

    xf::cv::Mat<IN_TYPE, HEIGHT, WIDTH, NPC1> imgInput1(height, width);
    xf::cv::Mat<IN_TYPE, HEIGHT, WIDTH, NPC1> imgInput2(height, width);
    xf::cv::Mat<IN_TYPE, HEIGHT, WIDTH, NPC1> imgInput3(height, width);
    xf::cv::Mat<IN_TYPE, HEIGHT, WIDTH, NPC1> imgInput4(height, width);
    xf::cv::Mat<OUT_TYPE, HEIGHT, WIDTH, NPC1> imgOutput(height, width);

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on

    // Retrieve xf::cv::Mat objects from img_in data:
    xf::cv::Array2xfMat<INPUT_PTR_WIDTH, IN_TYPE, HEIGHT, WIDTH, NPC1>(img_in1, imgInput1);
    xf::cv::Array2xfMat<INPUT_PTR_WIDTH, IN_TYPE, HEIGHT, WIDTH, NPC1>(img_in2, imgInput2);
    xf::cv::Array2xfMat<INPUT_PTR_WIDTH, IN_TYPE, HEIGHT, WIDTH, NPC1>(img_in3, imgInput3);
    xf::cv::Array2xfMat<INPUT_PTR_WIDTH, IN_TYPE, HEIGHT, WIDTH, NPC1>(img_in4, imgInput4);

    // Run xfOpenCV kernel:
    xf::cv::merge<IN_TYPE, OUT_TYPE, HEIGHT, WIDTH, NPC1>(imgInput1, imgInput2, imgInput3, imgInput4, imgOutput);

    // Convert imgOutput xf::cv::Mat object to output array:
    xf::cv::xfMat2Array<OUTPUT_PTR_WIDTH, OUT_TYPE, HEIGHT, WIDTH, NPC1>(imgOutput, img_out);

    return;

} // End of kernel

#endif

#if THREE_INPUT
void channel_combine_accel(ap_uint<INPUT_PTR_WIDTH>* img_in1,
                           ap_uint<INPUT_PTR_WIDTH>* img_in2,
                           ap_uint<INPUT_PTR_WIDTH>* img_in3,
                           ap_uint<OUTPUT_PTR_WIDTH>* img_out,
                           int height,
                           int width) {
// clang-format off
    #pragma HLS INTERFACE m_axi      port=img_in1       offset=slave  bundle=gmem0 depth=__XF_DEPTH
    #pragma HLS INTERFACE m_axi      port=img_in2       offset=slave  bundle=gmem1 depth=__XF_DEPTH
    #pragma HLS INTERFACE m_axi      port=img_in3       offset=slave  bundle=gmem2 depth=__XF_DEPTH
    #pragma HLS INTERFACE m_axi      port=img_out       offset=slave  bundle=gmem4 depth=__XF_DEPTH_OUT
    #pragma HLS interface s_axilite  port=height	             
    #pragma HLS interface s_axilite  port=width 	             
    #pragma HLS interface s_axilite  port=return
    // clang-format on

    xf::cv::Mat<IN_TYPE, HEIGHT, WIDTH, NPC1> imgInput1(height, width);
    xf::cv::Mat<IN_TYPE, HEIGHT, WIDTH, NPC1> imgInput2(height, width);
    xf::cv::Mat<IN_TYPE, HEIGHT, WIDTH, NPC1> imgInput3(height, width);
    xf::cv::Mat<OUT_TYPE, HEIGHT, WIDTH, NPC1> imgOutput(height, width);

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on

    // Retrieve xf::cv::Mat objects from img_in data:
    xf::cv::Array2xfMat<INPUT_PTR_WIDTH, IN_TYPE, HEIGHT, WIDTH, NPC1>(img_in1, imgInput1);
    xf::cv::Array2xfMat<INPUT_PTR_WIDTH, IN_TYPE, HEIGHT, WIDTH, NPC1>(img_in2, imgInput2);
    xf::cv::Array2xfMat<INPUT_PTR_WIDTH, IN_TYPE, HEIGHT, WIDTH, NPC1>(img_in3, imgInput3);

    // Run xfOpenCV kernel:
    xf::cv::merge<IN_TYPE, OUT_TYPE, HEIGHT, WIDTH, NPC1>(imgInput1, imgInput2, imgInput3, imgOutput);

    // Convert imgOutput xf::cv::Mat object to output array:
    xf::cv::xfMat2Array<OUTPUT_PTR_WIDTH, OUT_TYPE, HEIGHT, WIDTH, NPC1>(imgOutput, img_out);

    return;
}
#endif

#if TWO_INPUT
void channel_combine_accel(ap_uint<INPUT_PTR_WIDTH>* img_in1,
                           ap_uint<INPUT_PTR_WIDTH>* img_in2,
                           ap_uint<OUTPUT_PTR_WIDTH>* img_out,
                           int height,
                           int width) {
// clang-format off
    #pragma HLS INTERFACE m_axi      port=img_in1       offset=slave  bundle=gmem0 depth=__XF_DEPTH
    #pragma HLS INTERFACE m_axi      port=img_in2       offset=slave  bundle=gmem1  depth=__XF_DEPTH
    #pragma HLS INTERFACE m_axi      port=img_out       offset=slave  bundle=gmem4 depth=__XF_DEPTH_OUT
    #pragma HLS interface s_axilite  port=height	              
    #pragma HLS interface s_axilite  port=width 	              
    #pragma HLS interface s_axilite  port=return
    // clang-format on

    xf::cv::Mat<IN_TYPE, HEIGHT, WIDTH, NPC1> imgInput1(height, width);
    xf::cv::Mat<IN_TYPE, HEIGHT, WIDTH, NPC1> imgInput2(height, width);
    xf::cv::Mat<OUT_TYPE, HEIGHT, WIDTH, NPC1> imgOutput(height, width);

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on

    // Retrieve xf::cv::Mat objects from img_in data:
    xf::cv::Array2xfMat<INPUT_PTR_WIDTH, IN_TYPE, HEIGHT, WIDTH, NPC1>(img_in1, imgInput1);
    xf::cv::Array2xfMat<INPUT_PTR_WIDTH, IN_TYPE, HEIGHT, WIDTH, NPC1>(img_in2, imgInput2);

    // Run xfOpenCV kernel:
    xf::cv::merge<IN_TYPE, OUT_TYPE, HEIGHT, WIDTH, NPC1>(imgInput1, imgInput2, imgOutput);

    // Convert imgOutput xf::cv::Mat object to output array:
    xf::cv::xfMat2Array<OUTPUT_PTR_WIDTH, OUT_TYPE, HEIGHT, WIDTH, NPC1>(imgOutput, img_out);

    return;
}
#endif
