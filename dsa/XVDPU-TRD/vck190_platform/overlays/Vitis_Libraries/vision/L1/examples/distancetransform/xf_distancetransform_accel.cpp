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

#include "xf_distancetransform_config.h"

constexpr int __XF_DEPTH = 128 * 128; // modify the depth based on the image dimension for co-sim tests

void distancetransform_accel(
    ap_uint<INPUT_PTR_WIDTH>* img_inp, float* img_out, ap_uint<FWPASS_PTR_WIDTH>* fw_pass_data, int rows, int cols) {
// clang-format off
    #pragma HLS INTERFACE m_axi     port=img_inp  offset=slave bundle=gmem1 depth=__XF_DEPTH
    #pragma HLS INTERFACE m_axi     port=img_out  offset=slave bundle=gmem2 depth=__XF_DEPTH
    #pragma HLS INTERFACE m_axi     port=fw_pass_data  offset=slave bundle=gmem3 depth=__XF_DEPTH
    #pragma HLS INTERFACE s_axilite port=rows
    #pragma HLS INTERFACE s_axilite port=cols
    #pragma HLS INTERFACE s_axilite port=return
    // clang-format on

    xf::cv::distanceTransform<INPUT_PTR_WIDTH, FWPASS_PTR_WIDTH, HEIGHT, WIDTH, false>(img_inp, img_out, fw_pass_data,
                                                                                       rows, cols);
}
