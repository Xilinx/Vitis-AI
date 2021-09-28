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

extern "C" {
void cornerupdate_accel(unsigned long* list_fix,
                        unsigned int* list,
                        uint32_t nCorners,
                        unsigned int* flow_vectors,
                        bool harris_flag,
                        int flow_rows,
                        int flow_cols) {
// clang-format off
    #pragma HLS INTERFACE m_axi     port=list_fix  offset=slave bundle=gmem7
    #pragma HLS INTERFACE m_axi     port=list  offset=slave bundle=gmem8
    #pragma HLS INTERFACE m_axi     port=flow_vectors  offset=slave bundle=gmem9
// clang-format on

// clang-format off
    #pragma HLS INTERFACE s_axilite port=nCorners     
    #pragma HLS INTERFACE s_axilite port=harris_flag     
    #pragma HLS INTERFACE s_axilite port=flow_rows     
    #pragma HLS INTERFACE s_axilite port=flow_cols     
    #pragma HLS INTERFACE s_axilite port=return
    // clang-format on

    xf::cv::Mat<XF_32UC1, HEIGHT, WIDTH, XF_NPPC1> flow_mat(flow_rows, flow_cols, flow_vectors);

    xf::cv::cornerUpdate<MAXCORNERS, XF_32UC1, HEIGHT, WIDTH, XF_NPPC1>(list_fix, list, nCorners, flow_mat,
                                                                        (ap_uint<1>)(harris_flag));
}
}
