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

#include "xf_mean_shift_config.h"

extern "C" {
void mean_shift_accel(ap_uint<INPUT_PTR_WIDTH>* img_inp,
                      uint16_t* tlx,
                      uint16_t* tly,
                      uint16_t* obj_height,
                      uint16_t* obj_width,
                      uint16_t* dx,
                      uint16_t* dy,
                      uint16_t* track,
                      uint8_t frame_status,
                      uint8_t no_objects,
                      uint8_t no_of_iterations,
                      int rows,
                      int cols) {
// clang-format off
    #pragma HLS INTERFACE m_axi     port=img_inp  depth=2073600 offset=slave bundle=gmem1
    #pragma HLS INTERFACE m_axi     port=tlx  offset=slave bundle=gmem2
    #pragma HLS INTERFACE m_axi     port=tly  offset=slave bundle=gmem2
    #pragma HLS INTERFACE m_axi     port=obj_height  offset=slave bundle=gmem4
    #pragma HLS INTERFACE m_axi     port=obj_width  offset=slave bundle=gmem4
    #pragma HLS INTERFACE m_axi     port=dx  offset=slave bundle=gmem3
    #pragma HLS INTERFACE m_axi     port=dy  offset=slave bundle=gmem3
    #pragma HLS INTERFACE m_axi     port=track  offset=slave bundle=gmem5
    #pragma HLS INTERFACE s_axilite port=frame_status     
    #pragma HLS INTERFACE s_axilite port=no_objects     
    #pragma HLS INTERFACE s_axilite port=no_of_iterations     
    #pragma HLS INTERFACE s_axilite port=rows     
    #pragma HLS INTERFACE s_axilite port=cols     
    #pragma HLS INTERFACE s_axilite port=return
    // clang-format on

    xf::cv::Mat<XF_8UC4, XF_HEIGHT, XF_WIDTH, XF_NPPC1> inMat(rows, cols, img_inp);

    xf::cv::MeanShift<XF_MAX_OBJECTS, XF_MAX_ITERS, XF_MAX_OBJ_HEIGHT, XF_MAX_OBJ_WIDTH, XF_8UC4, XF_HEIGHT, XF_WIDTH,
                      XF_NPPC1>(inMat, tlx, tly, obj_height, obj_width, dx, dy, track, frame_status, no_objects,
                                no_of_iterations);
}
}
