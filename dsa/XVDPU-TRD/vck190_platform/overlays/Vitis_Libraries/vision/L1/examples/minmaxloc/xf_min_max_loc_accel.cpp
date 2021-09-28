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

#include "xf_min_max_loc_config.h"

static constexpr int __XF_DEPTH = (HEIGHT * WIDTH * (XF_PIXELWIDTH(TYPE, NPC1)) / 8) / (PTR_WIDTH / 8);

void min_max_loc_accel(ap_uint<PTR_WIDTH>* img_in,
                       int32_t& min_value,
                       int32_t& max_value,
                       uint16_t& min_loc_x,
                       uint16_t& min_loc_y,
                       uint16_t& max_loc_x,
                       uint16_t& max_loc_y,
                       int height,
                       int width) {
// clang-format off
    #pragma HLS INTERFACE m_axi      port=img_in          offset=slave  bundle=gmem0 depth=__XF_DEPTH
    #pragma HLS INTERFACE m_axi      port=min_value       offset=slave  bundle=gmem1
    #pragma HLS INTERFACE m_axi      port=max_value       offset=slave  bundle=gmem1
    #pragma HLS INTERFACE m_axi      port=min_loc_x  offset=slave  bundle=gmem2
    #pragma HLS INTERFACE m_axi      port=min_loc_y  offset=slave  bundle=gmem2
    #pragma HLS INTERFACE m_axi      port=max_loc_x  offset=slave  bundle=gmem2
    #pragma HLS INTERFACE m_axi      port=max_loc_y  offset=slave  bundle=gmem2
    #pragma HLS INTERFACE s_axilite  port=height 			
    #pragma HLS INTERFACE s_axilite  port=width 			
    #pragma HLS INTERFACE s_axilite  port=return
    // clang-format on

    // Local objects:
    xf::cv::Mat<TYPE, HEIGHT, WIDTH, NPC1> imgInput(height, width);

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on

    // Retrieve xf::cv::Mat objects from img_in data:
    xf::cv::Array2xfMat<PTR_WIDTH, TYPE, HEIGHT, WIDTH, NPC1>(img_in, imgInput);

    // Run xfOpenCV kernel:
    xf::cv::minMaxLoc<TYPE, HEIGHT, WIDTH, NPC1>(imgInput, &min_value, &max_value, &min_loc_x, &min_loc_y, &max_loc_x,
                                                 &max_loc_y);

    return;
} // End of kernel
