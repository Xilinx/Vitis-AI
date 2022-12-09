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
void blobfromimage_accel(ap_uint<INPUT_PTR_WIDTH> *Y_OR_RGB_img_inp,
			 ap_uint<INPUT_PTR_WIDTH> *U_img_inp, 	
			 ap_uint<INPUT_PTR_WIDTH> *V_img_inp,  
                         ap_uint<OUTPUT_PTR_WIDTH>* img_out, 
                         float params[2 * XF_CHANNELS(IN_TYPE, NPC)],
                         int in_img_width,
                         int in_img_height,
                         int in_img_linestride,
                         int resize_width,
                         int resize_height,
                         int out_img_width,      
                         int out_img_height,     
                         int out_img_linestride, 
                         int roi_posx,
                         int roi_posy) {
// clang-format off
#pragma HLS INTERFACE m_axi     port=Y_OR_RGB_img_inp  	offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi     port=U_img_inp  	offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi     port=V_img_inp  	offset=slave bundle=gmem3
#pragma HLS INTERFACE m_axi     port=img_out  		offset=slave bundle=gmem4
#pragma HLS INTERFACE m_axi     port=params  		offset=slave bundle=gmem5

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

    int t_out_img_linestride;

    xf::cv::Mat<XF_8UC3, HEIGHT, WIDTH, NPC> rgb_imgInput(in_img_height, in_img_width);
// clang-format off
#pragma HLS stream variable = rgb_imgInput.data depth = 2
    // clang-format on

#if EN_COLOR_CONV

    xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, NPC> y_img_inmat(in_img_height, in_img_width);
// clang-format off
#pragma HLS stream variable=y_img_inmat.data depth=2
    // clang-format on

    xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, NPC> u_img_inmat(in_img_height, in_img_width);
// clang-format off
#pragma HLS stream variable=u_img_inmat.data depth=2
    // clang-format on
	
    xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, NPC> v_img_inmat(in_img_height, in_img_width);
// clang-format off
#pragma HLS stream variable=v_img_inmat.data depth=2
    // clang-format on

#endif //EN_COLOR_CONV

#if BGR2RGB
    xf::cv::Mat<OUT_TYPE, HEIGHT, WIDTH, NPC> ch_swap_mat(in_img_height, in_img_width);
#endif //BGR2RGB
 
    xf::cv::Mat<OUT_TYPE, NEWHEIGHT, NEWWIDTH, NPC> resize_out_mat(out_img_height, out_img_width);

#if CROP
    xf::cv::Rect_<unsigned int> roi;
    roi.x = roi_posx;
    roi.y = roi_posy;
    roi.height = out_img_height;
    roi.width = out_img_width;

    xf::cv::Mat<OUT_TYPE, NEWHEIGHT, NEWWIDTH, NPC> crop_mat(out_img_height, out_img_width);
#endif //CROP

    xf::cv::Mat<OUT_TYPE, NEWHEIGHT, NEWWIDTH, NPC> out_mat(out_img_height, out_img_width);

// clang-format off
#pragma HLS stream variable = resize_out_mat.data depth = 2
#pragma HLS DATAFLOW
    // clang-format on

    xf::cv::accel_utils obj;

#if EN_COLOR_CONV
    obj.Array2xfMat<INPUT_PTR_WIDTH,XF_8UC1,HEIGHT, WIDTH, NPC>  (Y_OR_RGB_img_inp, y_img_inmat,in_img_linestride);
    obj.Array2xfMat<INPUT_PTR_WIDTH,XF_8UC1,HEIGHT, WIDTH, NPC>  (U_img_inp, u_img_inmat,in_img_linestride);
    obj.Array2xfMat<INPUT_PTR_WIDTH,XF_8UC1,HEIGHT, WIDTH, NPC>  (V_img_inp, v_img_inmat,in_img_linestride);
    t_out_img_linestride = out_img_linestride;
    xf::cv::yuv42rgb<XF_8UC1, XF_8UC3, HEIGHT, WIDTH, NPC>(y_img_inmat, u_img_inmat, v_img_inmat, rgb_imgInput);
#else
    obj.Array2xfMat<INPUT_PTR_WIDTH,XF_8UC3,HEIGHT, WIDTH, NPC>  (Y_OR_RGB_img_inp, rgb_imgInput, in_img_linestride);
    t_out_img_linestride = out_img_linestride + U_img_inp[0] - V_img_inp[0]; //adding this so that the unused interfaces won't be optimized away
#endif //En_COLOR_CONV
	

#if BGR2RGB
    xf::cv::bgr2rgb<IN_TYPE, OUT_TYPE, HEIGHT, WIDTH, NPC>(rgb_imgInput, ch_swap_mat);

#if USE_LETTERBOX
    xf::cv::letterbox<INTERPOLATION,IN_TYPE,HEIGHT,WIDTH,NEWHEIGHT,NEWWIDTH,NPC,MAXDOWNSCALE,128> (ch_swap_mat, resize_out_mat, resize_height, resize_width);
#else
    xf::cv::resize<INTERPOLATION, IN_TYPE, HEIGHT, WIDTH, NEWHEIGHT, NEWWIDTH, NPC, MAXDOWNSCALE>(ch_swap_mat, resize_out_mat);
#endif // USE_LETTERBOX

#else

#if USE_LETTERBOX
    xf::cv::letterbox<INTERPOLATION,IN_TYPE,HEIGHT,WIDTH,NEWHEIGHT,NEWWIDTH,NPC,MAXDOWNSCALE,128> (rgb_imgInput, resize_out_mat, resize_height, resize_width);
#else
    xf::cv::resize<INTERPOLATION, IN_TYPE, HEIGHT, WIDTH, NEWHEIGHT, NEWWIDTH, NPC, MAXDOWNSCALE>(rgb_imgInput, resize_out_mat);
#endif // USE_LETTERBOX

#endif //BGR2RGB

#if CROP
    xf::cv::crop<OUT_TYPE, NEWHEIGHT, NEWWIDTH, 0, NPC>(resize_out_mat, crop_mat, roi);
    xf::cv::preProcess<IN_TYPE, OUT_TYPE, NEWHEIGHT, NEWWIDTH, NPC, WIDTH_A, IBITS_A, WIDTH_B, IBITS_B, WIDTH_OUT,
                       IBITS_OUT>(crop_mat, out_mat, params);
#else

    xf::cv::preProcess<IN_TYPE, OUT_TYPE, NEWHEIGHT, NEWWIDTH, NPC, WIDTH_A, IBITS_A, WIDTH_B, IBITS_B, WIDTH_OUT,
                       IBITS_OUT>(resize_out_mat, out_mat, params);

#endif //CROP
    xf::cv::xfMat2Array<OUTPUT_PTR_WIDTH, OUT_TYPE, NEWHEIGHT, NEWWIDTH, NPC>(out_mat, img_out, out_img_linestride);
}
}
