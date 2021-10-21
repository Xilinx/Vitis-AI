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

#include "xf_stereolbm_config.h"

static constexpr int __XF_DEPTH_IN = (HEIGHT * WIDTH * (XF_PIXELWIDTH(IN_TYPE, NPC)) / 8) / (INPUT_PTR_WIDTH / 8);
static constexpr int __XF_DEPTH_OUT = (HEIGHT * WIDTH * (XF_PIXELWIDTH(OUT_TYPE, NPC)) / 8) / (OUTPUT_PTR_WIDTH / 8);

void stereolbm_accel(ap_uint<INPUT_PTR_WIDTH>* img_in_l,
                     ap_uint<INPUT_PTR_WIDTH>* img_in_r,
                     unsigned char* bm_state_in,
                     ap_uint<OUTPUT_PTR_WIDTH>* img_out,
                     int rows,
                     int cols) {
// clang-format off
	#pragma HLS INTERFACE m_axi      port=img_in_l      offset=slave  bundle=gmem0 depth=__XF_DEPTH_IN
	#pragma HLS INTERFACE m_axi      port=img_in_r      offset=slave  bundle=gmem1 depth=__XF_DEPTH_IN
	#pragma HLS INTERFACE m_axi      port=bm_state_in   offset=slave  bundle=gmem2 depth=4
	#pragma HLS INTERFACE m_axi      port=img_out       offset=slave  bundle=gmem3 depth=__XF_DEPTH_OUT
	#pragma HLS INTERFACE s_axilite  port=rows		bundle=control
	#pragma HLS INTERFACE s_axilite  port=cols		bundle=control
	#pragma HLS INTERFACE s_axilite  port=return		bundle=control
    // clang-format on

    xf::cv::Mat<IN_TYPE, HEIGHT, WIDTH, NPC> imgInputL(rows, cols);
    xf::cv::Mat<IN_TYPE, HEIGHT, WIDTH, NPC> imgInputR(rows, cols);
    xf::cv::Mat<OUT_TYPE, HEIGHT, WIDTH, NPC> imgOutput(rows, cols);

    xf::cv::xFSBMState<SAD_WINDOW_SIZE, NO_OF_DISPARITIES, PARALLEL_UNITS> bmState;

    // Initialize SBM State:
    bmState.preFilterCap = bm_state_in[0];
    bmState.uniquenessRatio = bm_state_in[1];
    bmState.textureThreshold = bm_state_in[2];
    bmState.minDisparity = bm_state_in[3];

// clang-format off
	#pragma HLS DATAFLOW
    // clang-format on

    // Retrieve xf::Mat objects from img_in data:
    xf::cv::Array2xfMat<INPUT_PTR_WIDTH, IN_TYPE, HEIGHT, WIDTH, NPC>(img_in_l, imgInputL);
    xf::cv::Array2xfMat<INPUT_PTR_WIDTH, IN_TYPE, HEIGHT, WIDTH, NPC>(img_in_r, imgInputR);

    // Run xfOpenCV kernel:
    xf::cv::StereoBM<SAD_WINDOW_SIZE, NO_OF_DISPARITIES, PARALLEL_UNITS, IN_TYPE, OUT_TYPE, HEIGHT, WIDTH, NPC,
                     XF_USE_URAM>(imgInputL, imgInputR, imgOutput, bmState);

    // Convert _dst xf::Mat object to output array:
    xf::cv::xfMat2Array<OUTPUT_PTR_WIDTH, OUT_TYPE, HEIGHT, WIDTH, NPC>(imgOutput, img_out);

    return;
} // End of kernel
