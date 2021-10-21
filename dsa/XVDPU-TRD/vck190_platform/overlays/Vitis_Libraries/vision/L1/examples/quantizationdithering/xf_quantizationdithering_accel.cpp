/*
 * Copyright 2020 Xilinx, Inc.
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

#include "xf_quantizationdithering_config.h"

static constexpr int __XF_DEPTH_IN = (HEIGHT * WIDTH * (XF_PIXELWIDTH(TYPEIN, NPC_T)) / 8) / (INPUT_PTR_WIDTH / 8);
static constexpr int __XF_DEPTH_OUT = (HEIGHT * WIDTH * (XF_PIXELWIDTH(TYPEOUT, NPC_T)) / 8) / (OUTPUT_PTR_WIDTH / 8);

void quantizationdithering_accel(ap_uint<INPUT_PTR_WIDTH>* img_inp,
                                 ap_uint<OUTPUT_PTR_WIDTH>* img_out,
                                 int rows_in,
                                 int cols_in) {
// clang-format off
    #pragma HLS INTERFACE m_axi     port=img_inp  offset=slave bundle=gmem1 depth=__XF_DEPTH_IN
    #pragma HLS INTERFACE m_axi     port=img_out  offset=slave bundle=gmem2 depth=__XF_DEPTH_OUT
    #pragma HLS INTERFACE s_axilite port=rows_in              
    #pragma HLS INTERFACE s_axilite port=cols_in              
    #pragma HLS INTERFACE s_axilite port=return
    // clang-format on

    xf::cv::Mat<TYPEIN, HEIGHT, WIDTH, NPC_T> in_mat(rows_in, cols_in);
    xf::cv::Mat<TYPEOUT, HEIGHT, WIDTH, NPC_T> out_mat(rows_in, cols_in);

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on

    xf::cv::Array2xfMat<INPUT_PTR_WIDTH, TYPEIN, HEIGHT, WIDTH, NPC_T>(img_inp, in_mat);
    xf::cv::xf_QuatizationDithering<TYPEIN, TYPEOUT, HEIGHT, WIDTH, SCALEFACTOR, MAXREPRESENTEDVALUE, NPC_T>(in_mat,
                                                                                                             out_mat);
    xf::cv::xfMat2Array<OUTPUT_PTR_WIDTH, TYPEOUT, HEIGHT, WIDTH, NPC_T>(out_mat, img_out);
}
