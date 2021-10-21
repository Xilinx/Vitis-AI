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
void pyr_down_accel(ap_uint<INPUT_PTR_WIDTH>* inImgPyr1,
                    ap_uint<OUTPUT_PTR_WIDTH>* outImgPyr1,
                    ap_uint<INPUT_PTR_WIDTH>* inImgPyr2,
                    ap_uint<OUTPUT_PTR_WIDTH>* outImgPyr2,
                    int pyr_h,
                    int pyr_w,
                    int pyr_out_h,
                    int pyr_out_w) {
// clang-format off
    #pragma HLS INTERFACE m_axi     port=inImgPyr1  offset=slave bundle=gmem1
    #pragma HLS INTERFACE m_axi     port=outImgPyr1  offset=slave bundle=gmem2
    #pragma HLS INTERFACE m_axi     port=inImgPyr2  offset=slave bundle=gmem3
    #pragma HLS INTERFACE m_axi     port=outImgPyr2  offset=slave bundle=gmem4
// clang-format on

// clang-format off
    #pragma HLS INTERFACE s_axilite port=inImgPyr1  
    #pragma HLS INTERFACE s_axilite port=outImgPyr1  
    #pragma HLS INTERFACE s_axilite port=inImgPyr2     
    #pragma HLS INTERFACE s_axilite port=outImgPyr2
// clang-format on

// clang-format off
    #pragma HLS INTERFACE s_axilite port=pyr_h     
    #pragma HLS INTERFACE s_axilite port=pyr_w     
    #pragma HLS INTERFACE s_axilite port=pyr_out_h     
    #pragma HLS INTERFACE s_axilite port=pyr_out_w
// clang-format on

// clang-format off
    #pragma HLS INTERFACE s_axilite port=return
    // clang-format on

    const int pROWS = HEIGHT;
    const int pCOLS = WIDTH;

    xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, XF_NPPC1> pyr1_in_mat;

    pyr1_in_mat.rows = pyr_h;
    pyr1_in_mat.cols = pyr_w;

    xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, XF_NPPC1> pyr1_out_mat;

    pyr1_out_mat.rows = pyr_out_h;
    pyr1_out_mat.cols = pyr_out_w;

    xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, XF_NPPC1> pyr2_in_mat;

    pyr2_in_mat.rows = pyr_h;
    pyr2_in_mat.cols = pyr_w;
    xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, XF_NPPC1> pyr2_out_mat;

    pyr2_out_mat.rows = pyr_out_h;
    pyr2_out_mat.cols = pyr_out_w;

// creating image pyramid
// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on
    xf::cv::Array2xfMat<INPUT_PTR_WIDTH, XF_8UC1, HEIGHT, WIDTH, XF_NPPC1>(inImgPyr1, pyr1_in_mat);
    xf::cv::pyrDown<XF_8UC1, HEIGHT, WIDTH, XF_NPPC1>(pyr1_in_mat, pyr1_out_mat);
    xf::cv::xfMat2Array<INPUT_PTR_WIDTH, XF_8UC1, HEIGHT, WIDTH, XF_NPPC1>(pyr1_out_mat, outImgPyr1);

    xf::cv::Array2xfMat<INPUT_PTR_WIDTH, XF_8UC1, HEIGHT, WIDTH, XF_NPPC1>(inImgPyr2, pyr2_in_mat);
    xf::cv::pyrDown<XF_8UC1, HEIGHT, WIDTH, XF_NPPC1>(pyr2_in_mat, pyr2_out_mat);
    xf::cv::xfMat2Array<INPUT_PTR_WIDTH, XF_8UC1, HEIGHT, WIDTH, XF_NPPC1>(pyr2_out_mat, outImgPyr2);

    return;
}
}
