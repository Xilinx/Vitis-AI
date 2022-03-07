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

#include "xf_magnitude_config.h"

static constexpr int __XF_DEPTH = (HEIGHT * WIDTH * (XF_PIXELWIDTH(XF_16SC1, NPC1)) / 8) / (INPUT_PTR_WIDTH / 8);

void magnitude_accel(ap_uint<INPUT_PTR_WIDTH>* img_inp1,
                     ap_uint<INPUT_PTR_WIDTH>* img_inp2,
                     ap_uint<OUTPUT_PTR_WIDTH>* img_out,
                     int rows,
                     int cols) {
// clang-format off
    #pragma HLS INTERFACE m_axi     port=img_inp1  offset=slave bundle=gmem1 depth=__XF_DEPTH
    #pragma HLS INTERFACE m_axi     port=img_inp2  offset=slave bundle=gmem2 depth=__XF_DEPTH
    #pragma HLS INTERFACE m_axi     port=img_out  offset=slave bundle=gmem3 depth=__XF_DEPTH
// clang-format on

// clang-format off

    //#pragma HLS INTERFACE s_axilite port=rows     bundle=control
    //#pragma HLS INTERFACE s_axilite port=cols     bundle=control
    #pragma HLS INTERFACE s_axilite port=return   bundle=control
    // clang-format on

    xf::cv::Mat<XF_16SC1, HEIGHT, WIDTH, NPC1> _src1(rows, cols);
    // clang-format off
    // clang-format on

    xf::cv::Mat<XF_16SC1, HEIGHT, WIDTH, NPC1> _src2(rows, cols);
    // clang-format off
    // clang-format on

    xf::cv::Mat<XF_16SC1, HEIGHT, WIDTH, NPC1> _dst(rows, cols);
// clang-format off
// clang-format on

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on

    xf::cv::Array2xfMat<INPUT_PTR_WIDTH, XF_16SC1, HEIGHT, WIDTH, NPC1>(img_inp1, _src1);
    xf::cv::Array2xfMat<INPUT_PTR_WIDTH, XF_16SC1, HEIGHT, WIDTH, NPC1>(img_inp2, _src2);

    xf::cv::magnitude<NORM_TYPE, XF_16SC1, XF_16SC1, HEIGHT, WIDTH, NPC1>(_src1, _src2, _dst);

    xf::cv::xfMat2Array<OUTPUT_PTR_WIDTH, XF_16SC1, HEIGHT, WIDTH, NPC1>(_dst, img_out);
}
