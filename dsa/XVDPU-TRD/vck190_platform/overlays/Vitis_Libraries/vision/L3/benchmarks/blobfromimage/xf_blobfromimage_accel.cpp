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

#include "xf_blobfromimage_config.h"

extern "C" {
void blobfromimage_accel(ap_uint<INPUT_PTR_WIDTH>* img_inp,  // Input image pointer
                         ap_uint<OUTPUT_PTR_WIDTH>* img_out, // output image pointer
                         float params[2 * XF_CHANNELS(IN_TYPE, NPC)],
                         int in_img_width,
                         int in_img_height,
                         int in_img_linestride,
                         int resize_width,
                         int resize_height,
                         int out_img_width,      // Final Output image width
                         int out_img_height,     // Final Output image height
                         int out_img_linestride, // Final Output image line stride
                         int roi_posx,
                         int roi_posy) {
// clang-format off
#pragma HLS INTERFACE m_axi     port=img_inp  offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi     port=img_out  offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi     port=params  offset=slave bundle=gmem3

#pragma HLS INTERFACE s_axilite port=in_img_width     
#pragma HLS INTERFACE s_axilite port=in_img_height     
#pragma HLS INTERFACE s_axilite port=in_img_linestride     
#pragma HLS INTERFACE s_axilite port=out_img_width     
#pragma HLS INTERFACE s_axilite port=out_img_height     
#pragma HLS INTERFACE s_axilite port=out_img_linestride     
#pragma HLS INTERFACE s_axilite port=roi_posx     
#pragma HLS INTERFACE s_axilite port=roi_posy     

#pragma HLS INTERFACE s_axilite port=return
    // clang-format on
    xf::cv::Mat<IN_TYPE, HEIGHT, WIDTH, NPC> imgInput(in_img_height, in_img_width);

#pragma HLS stream variable = imgInput.data depth = 2

#if BGR2RGB
    xf::cv::Mat<OUT_TYPE, HEIGHT, WIDTH, NPC> ch_swap_mat(in_img_height, in_img_width);
#endif
    xf::cv::Mat<OUT_TYPE, NEWHEIGHT, NEWWIDTH, NPC> resize_out_mat(resize_height, resize_width);

#if CROP
    xf::cv::Rect_<unsigned int> roi;
    roi.x = roi_posx;
    roi.y = roi_posy;
    roi.height = out_img_height;
    roi.width = out_img_width;

    xf::cv::Mat<OUT_TYPE, NEWHEIGHT, NEWWIDTH, NPC> crop_mat(out_img_height, out_img_width);
#endif
    xf::cv::Mat<OUT_TYPE, NEWHEIGHT, NEWWIDTH, NPC> out_mat(out_img_height, out_img_width);

// clang-format off
#pragma HLS stream variable = resize_out_mat.data depth = 2
#pragma HLS DATAFLOW
    // clang-format on
    xf::cv::Array2xfMat<INPUT_PTR_WIDTH, IN_TYPE, HEIGHT, WIDTH, NPC>(img_inp, imgInput, in_img_linestride);
#if BGR2RGB
    xf::cv::bgr2rgb<IN_TYPE, OUT_TYPE, HEIGHT, WIDTH, NPC>(imgInput, ch_swap_mat);
    xf::cv::resize<INTERPOLATION, IN_TYPE, HEIGHT, WIDTH, NEWHEIGHT, NEWWIDTH, NPC, MAXDOWNSCALE>(ch_swap_mat,
                                                                                                  resize_out_mat);
#else

    xf::cv::resize<INTERPOLATION, IN_TYPE, HEIGHT, WIDTH, NEWHEIGHT, NEWWIDTH, NPC, MAXDOWNSCALE>(imgInput,
                                                                                                  resize_out_mat);
#endif

#if CROP
    xf::cv::crop<OUT_TYPE, NEWHEIGHT, NEWWIDTH, 0, NPC>(resize_out_mat, crop_mat, roi);
    xf::cv::preProcess<IN_TYPE, OUT_TYPE, NEWHEIGHT, NEWWIDTH, NPC, WIDTH_A, IBITS_A, WIDTH_B, IBITS_B, WIDTH_OUT,
                       IBITS_OUT>(crop_mat, out_mat, params);
#else

    xf::cv::preProcess<IN_TYPE, OUT_TYPE, NEWHEIGHT, NEWWIDTH, NPC, WIDTH_A, IBITS_A, WIDTH_B, IBITS_B, WIDTH_OUT,
                       IBITS_OUT>(resize_out_mat, out_mat, params);

#endif
    xf::cv::xfMat2Array<OUTPUT_PTR_WIDTH, OUT_TYPE, NEWHEIGHT, NEWWIDTH, NPC>(out_mat, img_out, out_img_linestride);
}
}
