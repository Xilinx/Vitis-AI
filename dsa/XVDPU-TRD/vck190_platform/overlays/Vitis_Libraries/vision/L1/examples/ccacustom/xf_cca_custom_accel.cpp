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

#include "xf_cca_custom_config.h"

constexpr int __XF_DEPTH = 128 * 128; // modify the depth based on the image dimension for co-sim tests

void cca_custom_accel(uint8_t* in_ptr1,
                      uint8_t* in_ptr2,
                      uint8_t* tmp_out_ptr1,
                      uint8_t* tmp_out_ptr2,
                      uint8_t* out_ptr,
                      int* obj_pix,
                      int* def_pix,
                      int height,
                      int width) {
// clang-format off
    #pragma HLS INTERFACE m_axi     port=in_ptr1  offset=slave bundle=gmem1 depth=__XF_DEPTH
    #pragma HLS INTERFACE m_axi     port=in_ptr2  offset=slave bundle=gmem2 depth=__XF_DEPTH
    #pragma HLS INTERFACE m_axi     port=tmp_out_ptr1  offset=slave bundle=gmem3 depth=__XF_DEPTH
    #pragma HLS INTERFACE m_axi     port=tmp_out_ptr2  offset=slave bundle=gmem4 depth=__XF_DEPTH
    #pragma HLS INTERFACE m_axi     port=out_ptr  offset=slave bundle=gmem5 depth=__XF_DEPTH
    #pragma HLS INTERFACE m_axi port=obj_pix offset=slave bundle=gmem6 depth=1
    #pragma HLS INTERFACE m_axi port=def_pix offset=slave bundle=gmem6 depth=1
    #pragma HLS INTERFACE s_axilite port=height
    #pragma HLS INTERFACE s_axilite port=width
    #pragma HLS INTERFACE s_axilite port=return
    // clang-format on

    int tmp_obj, tmp_def;
    xf::cv::ccaCustom<HEIGHT, WIDTH>(in_ptr1, in_ptr2, tmp_out_ptr1, tmp_out_ptr2, out_ptr, tmp_obj, tmp_def, height,
                                     width);
    *obj_pix = tmp_obj;
    *def_pix = tmp_def;
}
