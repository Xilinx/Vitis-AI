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

#include "xf_sgbm_config.h"

static constexpr int __XF_DEPTH = (HEIGHT * WIDTH * (XF_PIXELWIDTH(IN_TYPE, NPC1)) / 8) / (INPUT_PTR_WIDTH / 8);

void semiglobalbm_accel(ap_uint<INPUT_PTR_WIDTH>* img_in_l,
                        ap_uint<INPUT_PTR_WIDTH>* img_in_r,
                        unsigned char penalty_small,
                        unsigned char penalty_large,
                        ap_uint<OUTPUT_PTR_WIDTH>* img_out,
                        int rows,
                        int cols) {
// clang-format off
    #pragma HLS INTERFACE m_axi      port=img_in_l      offset=slave  bundle=gmem0 depth=230400
    #pragma HLS INTERFACE m_axi      port=img_in_r      offset=slave  bundle=gmem1 depth=230400
    #pragma HLS INTERFACE m_axi      port=img_out       offset=slave  bundle=gmem2 depth=230400
    #pragma HLS INTERFACE s_axilite  port=penalty_small  	          bundle=control
    #pragma HLS INTERFACE s_axilite  port=penalty_large  	          bundle=control
    #pragma HLS INTERFACE s_axilite  port=rows  	          		  bundle=control
    #pragma HLS INTERFACE s_axilite  port=cols  	          		  bundle=control
    #pragma HLS INTERFACE s_axilite  port=return 			          bundle=control
    // clang-format on

    xf::cv::Mat<IN_TYPE, HEIGHT, WIDTH, NPC1> imgInputL(rows, cols);
    xf::cv::Mat<IN_TYPE, HEIGHT, WIDTH, NPC1> imgInputR(rows, cols);
    xf::cv::Mat<OUT_TYPE, HEIGHT, WIDTH, NPC1> imgOutput(rows, cols);

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on

    // Retrieve xf::cv::Mat objects from img_in data:
    xf::cv::Array2xfMat<INPUT_PTR_WIDTH, IN_TYPE, HEIGHT, WIDTH, NPC1>(img_in_l, imgInputL);
    xf::cv::Array2xfMat<INPUT_PTR_WIDTH, IN_TYPE, HEIGHT, WIDTH, NPC1>(img_in_r, imgInputR);

    // Run xfOpenCV kernel:
    xf::cv::SemiGlobalBM<XF_BORDER_CONSTANT, WINDOW_SIZE, TOTAL_DISPARITY, PARALLEL_UNITS, NUM_DIR, IN_TYPE, OUT_TYPE,
                         HEIGHT, WIDTH, NPC1>(imgInputL, imgInputR, imgOutput, penalty_small, penalty_large);

    // Convert _dst xf::cv::Mat object to output array:
    xf::cv::xfMat2Array<OUTPUT_PTR_WIDTH, OUT_TYPE, HEIGHT, WIDTH, NPC1>(imgOutput, img_out);

    return;
} // End of kernel
