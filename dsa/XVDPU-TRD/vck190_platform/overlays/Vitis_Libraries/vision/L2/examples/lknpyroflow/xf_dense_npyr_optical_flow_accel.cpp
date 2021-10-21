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

#include "xf_dense_npyr_optical_flow_config.h"

extern "C" {
void dense_non_pyr_of_accel(ap_uint<INPUT_PTR_WIDTH>* img_curr,
                            ap_uint<INPUT_PTR_WIDTH>* img_prev,
                            ap_uint<OUTPUT_PTR_WIDTH>* img_outx,
                            ap_uint<OUTPUT_PTR_WIDTH>* img_outy,
                            int rows,
                            int cols) {
// clang-format off
    #pragma HLS INTERFACE m_axi     port=img_curr  offset=slave bundle=gmem1
    #pragma HLS INTERFACE m_axi     port=img_prev  offset=slave bundle=gmem2
    #pragma HLS INTERFACE m_axi     port=img_outx  offset=slave bundle=gmem3
    #pragma HLS INTERFACE m_axi     port=img_outy  offset=slave bundle=gmem4
    #pragma HLS INTERFACE s_axilite port=cols  
    #pragma HLS INTERFACE s_axilite port=rows  
    #pragma HLS INTERFACE s_axilite port=return
    // clang-format on

    xf::cv::Mat<XF_8UC1, MAX_HEIGHT, MAX_WIDTH, NPPC> in_curr_mat(rows, cols);

    xf::cv::Mat<XF_8UC1, MAX_HEIGHT, MAX_WIDTH, NPPC> in_prev_mat(rows, cols);

    xf::cv::Mat<XF_32FC1, MAX_HEIGHT, MAX_WIDTH, NPPC> outx_mat(rows, cols);

    xf::cv::Mat<XF_32FC1, MAX_HEIGHT, MAX_WIDTH, NPPC> outy_mat(rows, cols);

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on

    xf::cv::Array2xfMat<INPUT_PTR_WIDTH, XF_8UC1, MAX_HEIGHT, MAX_WIDTH, NPPC>(img_curr, in_curr_mat);
    xf::cv::Array2xfMat<INPUT_PTR_WIDTH, XF_8UC1, MAX_HEIGHT, MAX_WIDTH, NPPC>(img_prev, in_prev_mat);

    xf::cv::DenseNonPyrLKOpticalFlow<KMED, XF_8UC1, MAX_HEIGHT, MAX_WIDTH, NPPC>(in_curr_mat, in_prev_mat, outx_mat,
                                                                                 outy_mat);

    xf::cv::xfMat2Array<OUTPUT_PTR_WIDTH, XF_32FC1, MAX_HEIGHT, MAX_WIDTH, NPPC>(outx_mat, img_outx);
    xf::cv::xfMat2Array<OUTPUT_PTR_WIDTH, XF_32FC1, MAX_HEIGHT, MAX_WIDTH, NPPC>(outy_mat, img_outy);
}
}
