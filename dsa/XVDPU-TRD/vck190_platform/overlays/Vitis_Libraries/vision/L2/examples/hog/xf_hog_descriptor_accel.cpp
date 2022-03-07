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

extern "C" {

void hog_descriptor_accel(
    ap_uint<PTR_IN_WIDTH>* img_in, ap_uint<PTR_OUT_WIDTH>* desc_out, int rows, int cols, int _desc_size) {
// clang-format off
    #pragma HLS INTERFACE m_axi      port=img_in        offset=slave  bundle=gmem0
    #pragma HLS INTERFACE m_axi      port=desc_out       offset=slave  bundle=gmem1
    #pragma HLS INTERFACE s_axilite  port=rows        	
    #pragma HLS INTERFACE s_axilite  port=cols	      	
    #pragma HLS INTERFACE s_axilite  port=_desc_size	
    #pragma HLS INTERFACE s_axilite  port=return
// clang-format on

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on

    xf::cv::Mat<IN_TYPE, XF_HEIGHT, XF_WIDTH, NPC> imgInput(rows, cols);
    xf::cv::Mat<OUT_TYPE, 1, XF_DESC_SIZE, NPC> descOutput(1, _desc_size);
// clang-format off
    #pragma HLS STREAM variable=imgInput.data depth=2
    #pragma HLS STREAM variable=descOutput.data depth=2
    // clang-format on

    // Retrieve xf::cv::Mat objects from img_in data:
    xf::cv::Array2xfMat<PTR_IN_WIDTH, IN_TYPE, XF_HEIGHT, XF_WIDTH, NPC>(img_in, imgInput);

    // Run xfOpenCV kernel:
    xf::cv::HOGDescriptor<XF_WIN_HEIGHT, XF_WIN_WIDTH, XF_WIN_STRIDE, XF_BLOCK_HEIGHT, XF_BLOCK_WIDTH, XF_CELL_HEIGHT,
                          XF_CELL_WIDTH, XF_NO_OF_BINS, XF_DESC_SIZE, XF_INPUT_COLOR, XF_OUTPUT_MODE, IN_TYPE, OUT_TYPE,
                          XF_HEIGHT, XF_WIDTH, NPC, XF_USE_URAM>(imgInput, descOutput);

    // Convert _dst xf::cv::Mat object to output array:
    xf::cv::xfMat2Array<PTR_OUT_WIDTH, OUT_TYPE, 1, XF_DESC_SIZE, NPC>(descOutput, desc_out);

    return;
} // End of kernel

} // End of extern C
