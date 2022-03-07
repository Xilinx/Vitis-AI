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

#include "xf_pyr_dense_optical_flow_config.h"

extern "C" {
void pyr_dense_optical_flow_accel(ap_uint<INPUT_PTR_WIDTH>* _current_img,
                                  ap_uint<INPUT_PTR_WIDTH>* _next_image,
                                  ap_uint<OUTPUT_PTR_WIDTH>* _streamFlowin,
                                  ap_uint<OUTPUT_PTR_WIDTH>* _streamFlowout,
                                  int level,
                                  int scale_up_flag,
                                  float scale_in,
                                  int init_flag,
                                  int cur_img_rows,
                                  int cur_img_cols,
                                  int next_img_rows,
                                  int next_img_cols,
                                  int flow_rows,
                                  int flow_cols,
                                  int flow_iter_rows,
                                  int flow_iter_cols) {
// clang-format off
    #pragma HLS INTERFACE m_axi     port=_current_img  offset=slave bundle=gmem1
    #pragma HLS INTERFACE m_axi     port=_next_image  offset=slave bundle=gmem2
    #pragma HLS INTERFACE m_axi     port=_streamFlowin  offset=slave bundle=gmem3
    #pragma HLS INTERFACE m_axi     port=_streamFlowout  offset=slave bundle=gmem4
// clang-format on

// clang-format off
    #pragma HLS INTERFACE s_axilite port=level   
    #pragma HLS INTERFACE s_axilite port=scale_up_flag   
    #pragma HLS INTERFACE s_axilite port=scale_in   
    #pragma HLS INTERFACE s_axilite port=init_flag   
    #pragma HLS INTERFACE s_axilite port=cur_img_rows   
    #pragma HLS INTERFACE s_axilite port=cur_img_cols   
    #pragma HLS INTERFACE s_axilite port=next_img_rows   
    #pragma HLS INTERFACE s_axilite port=next_img_cols   
    #pragma HLS INTERFACE s_axilite port=flow_rows   
    #pragma HLS INTERFACE s_axilite port=flow_cols   
    #pragma HLS INTERFACE s_axilite port=flow_iter_rows   
    #pragma HLS INTERFACE s_axilite port=flow_iter_cols   
    #pragma HLS INTERFACE s_axilite port=return
    // clang-format on

    xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, XF_NPPC1> current_img_mat(cur_img_rows, cur_img_cols);

    xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, XF_NPPC1> next_img_mat(next_img_rows, next_img_cols);

    xf::cv::Mat<XF_32UC1, HEIGHT, WIDTH, XF_NPPC1> streamFlowin_mat(flow_rows, flow_cols);

    xf::cv::Mat<XF_32UC1, HEIGHT, WIDTH, XF_NPPC1> streamFlowout_mat(flow_iter_rows, flow_iter_cols);

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on
    xf::cv::Array2xfMat<INPUT_PTR_WIDTH, XF_8UC1, HEIGHT, WIDTH, XF_NPPC1>(_current_img, current_img_mat);
    xf::cv::Array2xfMat<INPUT_PTR_WIDTH, XF_8UC1, HEIGHT, WIDTH, XF_NPPC1>(_next_image, next_img_mat);
    xf::cv::Array2xfMat<INPUT_PTR_WIDTH, XF_32UC1, HEIGHT, WIDTH, XF_NPPC1>(_streamFlowin, streamFlowin_mat);

    xf::cv::densePyrOpticalFlow<NUM_LEVELS, NUM_LINES_FINDIT, WINSIZE_OFLOW, TYPE_FLOW_WIDTH, TYPE_FLOW_INT, XF_8UC1,
                                HEIGHT, WIDTH, XF_NPPC1, XF_USE_URAM>(
        current_img_mat, next_img_mat, streamFlowin_mat, streamFlowout_mat, level, scale_up_flag, scale_in, init_flag);

    xf::cv::xfMat2Array<OUTPUT_PTR_WIDTH, XF_32UC1, HEIGHT, WIDTH, XF_NPPC1>(streamFlowout_mat, _streamFlowout);
}
}
