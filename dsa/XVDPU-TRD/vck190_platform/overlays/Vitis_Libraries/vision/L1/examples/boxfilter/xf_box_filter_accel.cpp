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

#include "xf_box_filter_config.h"

static constexpr int __XF_DEPTH = (HEIGHT * WIDTH * (XF_PIXELWIDTH(IN_T, NPIX)) / 8) / (INPUT_PTR_WIDTH / 8);

void boxfilter_accel(ap_uint<INPUT_PTR_WIDTH>* img_inp, ap_uint<OUTPUT_PTR_WIDTH>* img_out, int rows, int cols) {
// clang-format off
    #pragma HLS INTERFACE m_axi     port=img_inp  offset=slave bundle=gmem1 depth=__XF_DEPTH
    #pragma HLS INTERFACE m_axi     port=img_out  offset=slave bundle=gmem2 depth=__XF_DEPTH
    
    #pragma HLS INTERFACE s_axilite port=rows     bundle=control
    #pragma HLS INTERFACE s_axilite port=cols     bundle=control
    #pragma HLS INTERFACE s_axilite port=return   bundle=control
    // clang-format on

    xf::cv::Mat<IN_T, HEIGHT, WIDTH, NPIX> in_mat(rows, cols);
    // clang-format off
    // clang-format on

    xf::cv::Mat<IN_T, HEIGHT, WIDTH, NPIX> _dst(rows, cols);
// clang-format off
// clang-format on

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on

    xf::cv::Array2xfMat<INPUT_PTR_WIDTH, IN_T, HEIGHT, WIDTH, NPIX>(img_inp, in_mat);

    xf::cv::boxFilter<XF_BORDER_CONSTANT, FILTER_WIDTH, IN_T, HEIGHT, WIDTH, NPIX, XF_USE_URAM>(in_mat, _dst);

    xf::cv::xfMat2Array<OUTPUT_PTR_WIDTH, IN_T, HEIGHT, WIDTH, NPIX>(_dst, img_out);
}
