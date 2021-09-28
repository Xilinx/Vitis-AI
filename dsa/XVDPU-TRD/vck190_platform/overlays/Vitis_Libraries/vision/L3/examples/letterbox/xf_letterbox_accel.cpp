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

#include "xf_letterbox_config.h"

extern "C" {
void letterbox_accel(ap_uint<INPUT_PTR_WIDTH>* img_inp,
                     ap_uint<OUTPUT_PTR_WIDTH>* img_out,
                     int rows_in,
                     int cols_in,
                     int rows_out,
                     int cols_out,
                     int insert_pad_value) {
// clang-format off
#pragma HLS INTERFACE m_axi     port=img_inp  offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi     port=img_out  offset=slave bundle=gmem2
#pragma HLS INTERFACE s_axilite port=rows_in               
#pragma HLS INTERFACE s_axilite port=cols_in               
#pragma HLS INTERFACE s_axilite port=rows_out
#pragma HLS INTERFACE s_axilite port=cols_out
#pragma HLS INTERFACE s_axilite port=insert_pad_value
#pragma HLS INTERFACE s_axilite port=return
    // clang-format on

    // Compute Resize output image size for Letterbox
    float scale_height = (float)rows_out / (float)rows_in;
    float scale_width = (float)cols_out / (float)cols_in;
    int rows_out_resize, cols_out_resize;
    if (scale_width < scale_height) {
        cols_out_resize = cols_out;
        rows_out_resize = (int)((float)(rows_in * cols_out) / (float)cols_in);
    } else {
        cols_out_resize = (int)((float)(cols_in * rows_out) / (float)rows_in);
        rows_out_resize = rows_out;
    }

    xf::cv::Mat<TYPE, HEIGHT, WIDTH, NPC_T> imgInput0(rows_in, cols_in);
    xf::cv::Mat<TYPE, NEWHEIGHT, NEWWIDTH, NPC_T> out_mat_resize(rows_out_resize, cols_out_resize);
    xf::cv::Mat<TYPE, NEWHEIGHT, NEWWIDTH, NPC_T> out_mat(rows_out, cols_out);
// clang-format off
    #pragma HLS DATAFLOW
// clang-format on	
    xf::cv::Array2xfMat<INPUT_PTR_WIDTH,XF_8UC3,HEIGHT, WIDTH, NPC_T>  (img_inp, imgInput0);
    xf::cv::resize<INTERPOLATION,TYPE,HEIGHT,WIDTH,NEWHEIGHT,NEWWIDTH,NPC_T,MAXDOWNSCALE> (imgInput0, out_mat_resize);
    xf::cv::insertBorder<TYPE, NEWHEIGHT, NEWWIDTH, NEWHEIGHT, NEWWIDTH, NPC_T>(out_mat_resize, out_mat, insert_pad_value);
    xf::cv::xfMat2Array<OUTPUT_PTR_WIDTH, TYPE, NEWHEIGHT, NEWWIDTH, NPC_T>(out_mat, img_out);
    return;
}// end kernel
}// end extern C
