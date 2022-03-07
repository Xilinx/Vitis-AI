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

#include "xf_remap_config.h"

static constexpr int __XF_DEPTH = (HEIGHT * WIDTH * (XF_PIXELWIDTH(TYPE, NPC)) / 8) / (PTR_IMG_WIDTH / 8);
static constexpr int __XF_DEPTH_MAP = (HEIGHT * WIDTH * (XF_PIXELWIDTH(TYPE_XY, NPC)) / 8) / (4);

void remap_accel(
    ap_uint<PTR_IMG_WIDTH>* img_in, float* map_x, float* map_y, ap_uint<PTR_IMG_WIDTH>* img_out, int rows, int cols) {
// clang-format off
    #pragma HLS INTERFACE m_axi      port=img_in        offset=slave  bundle=gmem0 depth=__XF_DEPTH
    #pragma HLS INTERFACE m_axi      port=map_x         offset=slave  bundle=gmem1 depth=__XF_DEPTH_MAP
    #pragma HLS INTERFACE m_axi      port=map_y         offset=slave  bundle=gmem2 depth=__XF_DEPTH_MAP
    #pragma HLS INTERFACE m_axi      port=img_out       offset=slave  bundle=gmem3 depth=__XF_DEPTH
    #pragma HLS INTERFACE s_axilite  port=rows 	        
    #pragma HLS INTERFACE s_axilite  port=cols 	        
    #pragma HLS INTERFACE s_axilite  port=return
    // clang-format on

    xf::cv::Mat<TYPE, HEIGHT, WIDTH, NPC> imgInput(rows, cols);
    xf::cv::Mat<TYPE_XY, HEIGHT, WIDTH, NPC> mapX(rows, cols);
    xf::cv::Mat<TYPE_XY, HEIGHT, WIDTH, NPC> mapY(rows, cols);
    xf::cv::Mat<TYPE, HEIGHT, WIDTH, NPC> imgOutput(rows, cols);

    const int HEIGHT_WIDTH_LOOPCOUNT = HEIGHT * WIDTH / XF_NPIXPERCYCLE(NPC);
    for (unsigned int i = 0; i < rows * cols; ++i) {
// clang-format off
	#pragma HLS LOOP_TRIPCOUNT min=1 max=HEIGHT_WIDTH_LOOPCOUNT
        #pragma HLS PIPELINE II=1
        // clang-format on
        float map_x_val = map_x[i];
        float map_y_val = map_y[i];
        mapX.write_float(i, map_x_val);
        mapY.write_float(i, map_y_val);
    }

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on

    // Retrieve xf::cv::Mat objects from img_in data:
    xf::cv::Array2xfMat<PTR_IMG_WIDTH, TYPE, HEIGHT, WIDTH, NPC>(img_in, imgInput);

    // Run xfOpenCV kernel:
    xf::cv::remap<XF_WIN_ROWS, XF_INTERPOLATION_TYPE, TYPE, TYPE_XY, TYPE, HEIGHT, WIDTH, NPC, XF_USE_URAM>(
        imgInput, imgOutput, mapX, mapY);

    // Convert _dst xf::cv::Mat object to output array:
    xf::cv::xfMat2Array<PTR_IMG_WIDTH, TYPE, HEIGHT, WIDTH, NPC>(imgOutput, img_out);

    return;
} // End of kernel
