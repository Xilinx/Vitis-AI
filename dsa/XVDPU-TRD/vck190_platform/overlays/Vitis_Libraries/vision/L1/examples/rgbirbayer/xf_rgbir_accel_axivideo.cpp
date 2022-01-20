/*
 * Copyright 2021 Xilinx, Inc.
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

#include "xf_rgbir_config_axivideo.h"

void rgbir_accel(InStream& img_in,
                 OutStream& rggb_out,
                 OutStream& ir_out,
                 char R_IR_C1_wgts[25],
                 char R_IR_C2_wgts[25],
                 char B_at_R_wgts[25],
                 char IR_at_R_wgts[9],
                 char IR_at_B_wgts[9],
                 char sub_wgts[4],
                 int height,
                 int width) {
// clang-format off
	#pragma HLS INTERFACE axis port=img_in register
	#pragma HLS INTERFACE axis port=rggb_out register
	#pragma HLS INTERFACE axis port=ir_out register

    #pragma HLS INTERFACE s_axilite port=R_IR_C1_wgts
    #pragma HLS INTERFACE s_axilite port=R_IR_C2_wgts
    #pragma HLS INTERFACE s_axilite port=B_at_R_wgts
    #pragma HLS INTERFACE s_axilite port=IR_at_R_wgts
    #pragma HLS INTERFACE s_axilite port=IR_at_B_wgts
	#pragma HLS INTERFACE s_axilite port=height
    #pragma HLS INTERFACE s_axilite port=width
    #pragma HLS INTERFACE s_axilite port=return
    // clang-format on

    xf::cv::Mat<IN_TYPE, HEIGHT, WIDTH, NPC> imgInput(height, width);
    xf::cv::Mat<IN_TYPE, HEIGHT, WIDTH, NPC> rggbOutput(height, width);
    xf::cv::Mat<IN_TYPE, HEIGHT, WIDTH, NPC> fullIrOutput(height, width);

#pragma HLS DATAFLOW

    // Retrieve xf::cv::Mat objects from img_in data:
    xf::cv::AXIvideo2xfMat(img_in, imgInput);

    static constexpr int n = XF_DTPIXELDEPTH(IN_TYPE, NPC);

    // Run xfOpenCV kernels:

    xf::cv::rgbir2bayer<FILTERSIZE1, FILTERSIZE2, BPATTERN, IN_TYPE, HEIGHT, WIDTH, NPC, 3 * WIDTH, XF_BORDER_CONSTANT,
                        XF_USE_URAM>(imgInput, R_IR_C1_wgts, R_IR_C2_wgts, B_at_R_wgts, IR_at_R_wgts, IR_at_B_wgts,
                                     sub_wgts, rggbOutput, fullIrOutput);

    xf::cv::xfMat2AXIvideo(rggbOutput, rggb_out);
    xf::cv::xfMat2AXIvideo(fullIrOutput, ir_out);

    return;
} // End of kernel
