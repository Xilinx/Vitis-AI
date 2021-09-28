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

#include "xf_lensshading_config.h"

extern "C" {
void lensshading_accel(ap_uint<INPUT_PTR_WIDTH>* src, ap_uint<OUTPUT_PTR_WIDTH>* dst, int rows, int cols) {
// clang-format off
#pragma HLS INTERFACE m_axi      port=src        offset=slave  bundle=gmem0 
#pragma HLS INTERFACE s_axilite  port=rows
#pragma HLS INTERFACE s_axilite  port=cols
#pragma HLS INTERFACE m_axi      port=dst       offset=slave  bundle=gmem4 
#pragma HLS INTERFACE s_axilite  port=return
    // clang-format on

    xf::cv::Mat<IN_TYPE, HEIGHT, WIDTH, NPIX> imgInput(rows, cols);
    xf::cv::Mat<OUT_TYPE, HEIGHT, WIDTH, NPIX> imgOutput(rows, cols);

// clang-format off
#pragma HLS DATAFLOW
    // clang-format on

    xf::cv::Array2xfMat<INPUT_PTR_WIDTH, IN_TYPE, HEIGHT, WIDTH, NPIX>(src, imgInput);

    xf::cv::Lscdistancebased<IN_TYPE, OUT_TYPE, HEIGHT, WIDTH, NPIX>(imgInput, imgOutput);

    xf::cv::xfMat2Array<OUTPUT_PTR_WIDTH, OUT_TYPE, HEIGHT, WIDTH, NPIX>(imgOutput, dst);
}
}
