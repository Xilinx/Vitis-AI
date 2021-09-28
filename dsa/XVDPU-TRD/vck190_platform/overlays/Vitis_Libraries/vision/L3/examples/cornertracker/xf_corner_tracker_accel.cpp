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

#include "xf_corner_tracker_config.h"
#include "HarrisImg.hpp"

extern "C" {
void cornerTracker(
    ap_uint<INPUT_PTR_WIDTH>* inHarris, unsigned int* list, unsigned int* params, int harris_rows, int harris_cols)

{
// clang-format off
    #pragma HLS INTERFACE m_axi     port=inHarris  offset=slave bundle=gmem1
    #pragma HLS INTERFACE m_axi     port=list  offset=slave bundle=gmem2
    #pragma HLS INTERFACE m_axi     port=params  offset=slave bundle=gmem3
// clang-format on

// clang-format off
    #pragma HLS INTERFACE s_axilite port=harris_rows     
    #pragma HLS INTERFACE s_axilite port=harris_cols     
    #pragma HLS INTERFACE s_axilite port=return
    // clang-format on

    uint16_t Thresh = params[1];
    unsigned int num_corners = params[0];
    bool harris_flag = (bool)params[2];
    float K_f = 0.04;
    uint16_t k = K_f * (1 << 16);
    uint32_t nCorners = 0;

    HarrisImg(inHarris, list, params, harris_rows, harris_cols, Thresh, k, &nCorners, harris_flag);

    if (harris_flag == true) {
        num_corners = nCorners;
    }

    params[0] = num_corners;

    return;
}
}