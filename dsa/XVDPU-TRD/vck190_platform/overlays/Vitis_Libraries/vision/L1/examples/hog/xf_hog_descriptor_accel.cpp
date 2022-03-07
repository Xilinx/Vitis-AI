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

#include "xf_hog_descriptor_config.h"

static constexpr int __XF_DEPTH = (XF_HEIGHT * XF_WIDTH * (XF_PIXELWIDTH(IN_TYPE, NPC)) / 8) / (INPUT_PTR_WIDTH / 8);
static constexpr int __XF_DEPTH_OUT = (1 * XF_DESC_SIZE * (XF_PIXELWIDTH(OUT_TYPE, NPC)) / 8) / (OUTPUT_PTR_WIDTH / 8);

void hog_descriptor_accel(
    ap_uint<INPUT_PTR_WIDTH>* img_in, ap_uint<OUTPUT_PTR_WIDTH>* desc_out, int rows, int cols, int _desc_size) {
// depth computed for the build config and corresponding test img 128x128
// clang-format off
    #pragma HLS INTERFACE m_axi      port=img_in        offset=slave  bundle=gmem0 depth=__XF_DEPTH
    #pragma HLS INTERFACE m_axi      port=desc_out       offset=slave  bundle=gmem1 depth=__XF_DEPTH_OUT
    #pragma HLS INTERFACE s_axilite  port=rows        	bundle=control
    #pragma HLS INTERFACE s_axilite  port=cols	      	bundle=control
    #pragma HLS INTERFACE s_axilite  port=_desc_size	bundle=control
    #pragma HLS INTERFACE s_axilite  port=return        bundle=control
// clang-format on

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on

    xf::cv::Mat<IN_TYPE, XF_HEIGHT, XF_WIDTH, NPC> imgInput(rows, cols);
    xf::cv::Mat<OUT_TYPE, 1, XF_DESC_SIZE, NPC> descOutput(1, _desc_size);

    // Retrieve xf::cv::Mat objects from img_in data:
    xf::cv::Array2xfMat<INPUT_PTR_WIDTH, IN_TYPE, XF_HEIGHT, XF_WIDTH, NPC>(img_in, imgInput);

    // Run xfOpenCV kernel:
    xf::cv::HOGDescriptor<XF_WIN_HEIGHT, XF_WIN_WIDTH, XF_WIN_STRIDE, XF_BLOCK_HEIGHT, XF_BLOCK_WIDTH, XF_CELL_HEIGHT,
                          XF_CELL_WIDTH, XF_NO_OF_BINS, XF_DESC_SIZE, XF_INPUT_COLOR, XF_OUTPUT_MODE, IN_TYPE, OUT_TYPE,
                          XF_HEIGHT, XF_WIDTH, NPC, XF_USE_URAM>(imgInput, descOutput);

    // Convert _dst xf::cv::Mat object to output array:
    xf::cv::xfMat2Array<OUTPUT_PTR_WIDTH, OUT_TYPE, 1, XF_DESC_SIZE, NPC>(descOutput, desc_out);

    return;
} // End of kernel
