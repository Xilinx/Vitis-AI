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

#include "xf_pyr_down_config.h"

static constexpr int __XF_DEPTH = (HEIGHT * WIDTH * (XF_PIXELWIDTH(TYPE, NPC_T)) / 8) / (INPUT_PTR_WIDTH / 8);
static constexpr int __XF_DEPTH_OUT =
    (OUT_HEIGHT * OUT_WIDTH * (XF_PIXELWIDTH(TYPE, NPC_T)) / 8) / (OUTPUT_PTR_WIDTH / 8);

void pyr_down_accel(ap_uint<INPUT_PTR_WIDTH>* img_inp,
                    ap_uint<OUTPUT_PTR_WIDTH>* img_out,
                    int in_rows,
                    int in_cols,
                    int out_rows,
                    int out_cols) {
// clang-format off
    #pragma HLS INTERFACE m_axi     port=img_inp  offset=slave bundle=gmem1 depth=__XF_DEPTH
    #pragma HLS INTERFACE m_axi     port=img_out  offset=slave bundle=gmem2 depth=__XF_DEPTH_OUT
    #pragma HLS INTERFACE s_axilite port=in_rows  
    #pragma HLS INTERFACE s_axilite port=in_cols  
    #pragma HLS INTERFACE s_axilite port=out_rows 
    #pragma HLS INTERFACE s_axilite port=out_cols 
    #pragma HLS INTERFACE s_axilite port=return
    // clang-format on

    xf::cv::Mat<TYPE, HEIGHT, WIDTH, NPC_T> in_mat(in_rows, in_cols);
    xf::cv::Mat<TYPE, HEIGHT, WIDTH, NPC_T> out_mat(out_rows, out_cols);

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on

    xf::cv::Array2xfMat<INPUT_PTR_WIDTH, TYPE, HEIGHT, WIDTH, NPC_T>(img_inp, in_mat);
    xf::cv::pyrDown<TYPE, HEIGHT, WIDTH, NPC_T, XF_USE_URAM>(in_mat, out_mat);
    xf::cv::xfMat2Array<OUTPUT_PTR_WIDTH, TYPE, HEIGHT, WIDTH, NPC_T>(out_mat, img_out);
}
