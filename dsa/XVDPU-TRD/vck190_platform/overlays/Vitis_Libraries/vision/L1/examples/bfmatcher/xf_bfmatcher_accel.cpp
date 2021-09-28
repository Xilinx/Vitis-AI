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

// =========================================================================
// Required files
// =========================================================================

// Configuration file
#include "xf_bfmatcher_config.h"

// =========================================================================
// Some internal macros for ease of use and readability
// =========================================================================
#define __BF_KERNEL__ xf::cv::bfMatcher<PARALLEL_COMPUTEUNIT, MAX_KEYPOINTS>

static constexpr int __XF_DEPTH = 236;

// -----------------------------------------------------------------------
// BF matcher implementation: Brute force method of desc matching
// -----------------------------------------------------------------------
void bfmatcher_accel(
    // ORB descriptor pointers
    ap_uint<INPUT_PTR_WIDTH>* desc_list1,
    ap_uint<INPUT_PTR_WIDTH>* desc_list2,

    // Matching descriptor
    ap_int<OUTPUT_PTR_WIDTH>* desc_match_idx,

    // number of valid keypoints in the corresponding desc lists
    uint32_t num_keypoints1,
    uint32_t num_keypoints2,

    // ratio threshold for lowe's ratio test
    float ratio_thresh) {
// clang-format off
    #pragma HLS INTERFACE m_axi      port=desc_list1       offset=slave  bundle=gmem0 depth=__XF_DEPTH
    #pragma HLS INTERFACE m_axi      port=desc_list2       offset=slave  bundle=gmem1 depth=__XF_DEPTH
    #pragma HLS INTERFACE m_axi      port=desc_match_idx       offset=slave  bundle=gmem2 depth=__XF_DEPTH
    #pragma HLS INTERFACE s_axilite  port=num_keypoints1 			          
    #pragma HLS INTERFACE s_axilite  port=num_keypoints2 			          
    #pragma HLS INTERFACE s_axilite  port=ratio_thresh
    #pragma HLS INTERFACE s_axilite  port=return
    // clang-format on

    // ................................
    // Run the BF kernel
    // ................................
    __BF_KERNEL__(desc_list1, desc_list2, desc_match_idx, num_keypoints1, num_keypoints2, ratio_thresh);

    return;
} // End of kernel
