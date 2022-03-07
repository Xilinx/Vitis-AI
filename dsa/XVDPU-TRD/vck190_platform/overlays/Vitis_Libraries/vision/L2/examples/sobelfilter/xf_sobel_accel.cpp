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

#include "xf_sobel_config.h"

extern "C" {
void sobel_accel(ap_uint<INPUT_PTR_WIDTH>* img_inp,
                 ap_uint<OUTPUT_PTR_WIDTH>* img_out1,
                 ap_uint<OUTPUT_PTR_WIDTH>* img_out2,
                 int rows,
                 int cols) {
// clang-format off
    #pragma HLS INTERFACE m_axi     port=img_inp  offset=slave bundle=gmem1
    #pragma HLS INTERFACE m_axi     port=img_out1  offset=slave bundle=gmem2
    #pragma HLS INTERFACE m_axi     port=img_out2  offset=slave bundle=gmem3
// clang-format on

// clang-format off
  
    #pragma HLS INTERFACE s_axilite port=rows     
    #pragma HLS INTERFACE s_axilite port=cols     
    #pragma HLS INTERFACE s_axilite port=return
    // clang-format on

    xf::cv::Mat<IN_TYPE, HEIGHT, WIDTH, NPC1> in_mat(rows, cols);
// clang-format off
    #pragma HLS stream variable=in_mat.data depth=2
    // clang-format on

    xf::cv::Mat<OUT_TYPE, HEIGHT, WIDTH, NPC1> _dstgx(rows, cols);
// clang-format off
    #pragma HLS stream variable=_dstgx.data depth=2
    // clang-format on

    xf::cv::Mat<OUT_TYPE, HEIGHT, WIDTH, NPC1> _dstgy(rows, cols);
// clang-format off
    #pragma HLS stream variable=_dstgy.data depth=2
// clang-format on

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on

    printf("Array2xfMat .... !!!\n");
    xf::cv::Array2xfMat<INPUT_PTR_WIDTH, IN_TYPE, HEIGHT, WIDTH, NPC1>(img_inp, in_mat);
    printf("Sobel .... !!!\n");
    xf::cv::Sobel<XF_BORDER_CONSTANT, FILTER_WIDTH, IN_TYPE, OUT_TYPE, HEIGHT, WIDTH, NPC1, XF_USE_URAM>(in_mat, _dstgx,
                                                                                                         _dstgy);
    printf("xfMat2Array .... !!!\n");
    xf::cv::xfMat2Array<OUTPUT_PTR_WIDTH, OUT_TYPE, HEIGHT, WIDTH, NPC1>(_dstgx, img_out1);
    xf::cv::xfMat2Array<OUTPUT_PTR_WIDTH, OUT_TYPE, HEIGHT, WIDTH, NPC1>(_dstgy, img_out2);
}
}