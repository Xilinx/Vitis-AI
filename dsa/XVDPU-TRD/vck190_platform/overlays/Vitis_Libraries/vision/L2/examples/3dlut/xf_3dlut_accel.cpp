/*
 * Copyright 2021 Xilinx, Inc.
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

#include "xf_3dlut_config.h"

extern "C" {
void lut3d_accel(ap_uint<INPUT_PTR_WIDTH>* img_in,
                 ap_uint<OUTPUT_PTR_WIDTH>* img_out,
                 ap_uint<INPUT_PTR_WIDTH>* lut,
                 int height,
                 int width,
                 int lutDim) {
// clang-format off
    #pragma HLS INTERFACE m_axi      port=img_in        offset=slave  bundle=gmem0
	#pragma HLS INTERFACE m_axi      port=lut        offset=slave  bundle=gmem1
    #pragma HLS INTERFACE m_axi      port=img_out       offset=slave  bundle=gmem2
    #pragma HLS INTERFACE s_axilite  port=height
    #pragma HLS INTERFACE s_axilite  port=width
	#pragma HLS INTERFACE s_axilite  port=lutDim
    #pragma HLS INTERFACE s_axilite  port=return
    // clang-format on

    xf::cv::Mat<IN_TYPE, HEIGHT, WIDTH, NPC1> imgInput(height, width);
    xf::cv::Mat<XF_32FC3, SQ_LUTDIM, LUT_DIM, NPC1> lutMat(lutDim * lutDim, lutDim);
    xf::cv::Mat<OUT_TYPE, HEIGHT, WIDTH, NPC1> imgOutput(height, width);

#pragma HLS DATAFLOW

    // Retrieve xf::cv::Mat objects from img_in data:
    xf::cv::Array2xfMat<INPUT_PTR_WIDTH, IN_TYPE, HEIGHT, WIDTH, NPC1>(img_in, imgInput);
    xf::cv::Array2xfMat<INPUT_PTR_WIDTH, XF_32FC3, SQ_LUTDIM, LUT_DIM, NPC1>(lut, lutMat);

    // Run xfOpenCV kernel:
    xf::cv::lut3d<LUT_DIM, SQ_LUTDIM, IN_TYPE, OUT_TYPE, HEIGHT, WIDTH, NPC1, XF_USE_URAM>(imgInput, lutMat, imgOutput,
                                                                                           lutDim);

    // Convert _dst xf::cv::Mat object to output array:
    xf::cv::xfMat2Array<OUTPUT_PTR_WIDTH, OUT_TYPE, HEIGHT, WIDTH, NPC1>(imgOutput, img_out);

    return;
} // End of kernel
}