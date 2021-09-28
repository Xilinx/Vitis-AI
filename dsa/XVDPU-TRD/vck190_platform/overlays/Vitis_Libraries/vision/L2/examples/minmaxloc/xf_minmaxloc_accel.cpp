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

extern "C" {

void minmaxloc_accel(
    ap_uint<PTR_WIDTH>* img_in, int32_t* min_max_value, uint16_t* min_max_loc_xy, int height, int width) {
// clang-format off
    #pragma HLS INTERFACE m_axi      port=img_in          offset=slave  bundle=gmem0
    #pragma HLS INTERFACE m_axi      port=min_max_value   offset=slave  bundle=gmem1
    #pragma HLS INTERFACE m_axi      port=min_max_loc_xy  offset=slave  bundle=gmem2
    #pragma HLS INTERFACE s_axilite  port=height 			
    #pragma HLS INTERFACE s_axilite  port=width 			
    #pragma HLS INTERFACE s_axilite  port=return
    // clang-format on

    // Local objects:
    xf::cv::Mat<TYPE, HEIGHT, WIDTH, NPC1> imgInput(height, width);
    int32_t min_value, max_value;
    uint16_t _min_locx, _min_locy, _max_locx, _max_locy;

// clang-format off
    #pragma HLS STREAM variable=imgInput.data depth=2
// clang-format on

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on

    // Retrieve xf::cv::Mat objects from img_in data:
    xf::cv::Array2xfMat<PTR_WIDTH, TYPE, HEIGHT, WIDTH, NPC1>(img_in, imgInput);

    // Run xfOpenCV kernel:
    xf::cv::minMaxLoc<TYPE, HEIGHT, WIDTH, NPC1>(imgInput, &min_value, &max_value, &_min_locx, &_min_locy, &_max_locx,
                                                 &_max_locy);

    // Copy local outputs to global pointer:
    min_max_value[0] = min_value;
    min_max_value[1] = max_value;
    min_max_loc_xy[0] = _min_locx;
    min_max_loc_xy[1] = _min_locy;
    min_max_loc_xy[2] = _max_locx;
    min_max_loc_xy[3] = _max_locy;

    return;
} // End of kernel

} // End of extern C
