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

#include "xf_pp_pipeline_config.h"

extern "C" {
void pp_pipeline_accel(ap_uint<INPUT_PTR_WIDTH> *Y_img_inp, 
			ap_uint<INPUT_PTR_WIDTH> *U_img_inp, 
			ap_uint<INPUT_PTR_WIDTH> *V_img_inp, 
			ap_uint<OUTPUT_PTR_WIDTH> *img_out, 
			int rows_in, 
			int cols_in, 
			int rows_out, 
			int cols_out, 
			int rows_out_resize,
			int cols_out_resize,
			float params[3*T_CHANNELS], 
			int th1, 
			int th2)
{
// clang-format off
#pragma HLS INTERFACE m_axi     port=Y_img_inp  offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi     port=U_img_inp  offset=slave bundle=gmem4
#pragma HLS INTERFACE m_axi     port=V_img_inp  offset=slave bundle=gmem5
#pragma HLS INTERFACE m_axi     port=img_out  offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi     port=params  offset=slave bundle=gmem3


#pragma HLS INTERFACE s_axilite port=Y_img_inp     bundle=control
#pragma HLS INTERFACE s_axilite port=U_img_inp     bundle=control
#pragma HLS INTERFACE s_axilite port=V_img_inp     bundle=control
#pragma HLS INTERFACE s_axilite port=img_out     bundle=control
#pragma HLS INTERFACE s_axilite port=params     bundle=control

#pragma HLS INTERFACE s_axilite port=rows_in     bundle=control
#pragma HLS INTERFACE s_axilite port=cols_in     bundle=control
#pragma HLS INTERFACE s_axilite port=rows_out     bundle=control
#pragma HLS INTERFACE s_axilite port=cols_out     bundle=control
#pragma HLS INTERFACE s_axilite port=rows_out_resize     bundle=control
#pragma HLS INTERFACE s_axilite port=cols_out_resize     bundle=control

#pragma HLS INTERFACE s_axilite port=th1     bundle=control
#pragma HLS INTERFACE s_axilite port=th2     bundle=control

#pragma HLS INTERFACE s_axilite port=return   bundle=control
// clang-format on

xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, NPC> y_img_inmat(rows_in, cols_in);
// clang-format off
	#pragma HLS stream variable=y_img_inmat.data depth=2
// clang-format on
	
xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, NPC> u_img_inmat(rows_in, cols_in);
// clang-format off
	#pragma HLS stream variable=u_img_inmat.data depth=2
// clang-format on
	
xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, NPC> v_img_inmat(rows_in, cols_in);
	#pragma HLS stream variable=v_img_inmat.data depth=2
// clang-format on
 
xf::cv::Mat<XF_8UC3, HEIGHT, WIDTH, NPC>imgInput0(rows_in, cols_in);   
// clang-format off
	#pragma HLS stream variable=imgInput0.data depth=2
// clang-format on
		
xf::cv::Mat<TYPE, NEWHEIGHT, NEWWIDTH, NPC> out_mat(rows_out, cols_out);
// clang-format off
#pragma HLS stream variable=out_mat.data depth=2
// clang-format on
	
	hls::stream<ap_uint<INPUT_PTR_WIDTH> > resizeStrmout;
	int srcMat_cols_align_npc = ((out_mat.cols + (NPC - 1)) >> XF_BITSHIFT(NPC)) << XF_BITSHIFT(NPC);
	
	
// clang-format off
	#pragma HLS DATAFLOW
// clang-format on
	xf::cv::accel_utils obj;
	obj.Array2xfMat<INPUT_PTR_WIDTH,XF_8UC1,HEIGHT, WIDTH, NPC>  (Y_img_inp, y_img_inmat);
	obj.Array2xfMat<INPUT_PTR_WIDTH,XF_8UC1,HEIGHT, WIDTH, NPC>  (U_img_inp, u_img_inmat);
	obj.Array2xfMat<INPUT_PTR_WIDTH,XF_8UC1,HEIGHT, WIDTH, NPC>  (V_img_inp, v_img_inmat);
	
	//xf::cv::yuv42rgb - YUV to RGB conversion
	xf::cv::yuv42rgb<XF_8UC1, XF_8UC3, HEIGHT, WIDTH, NPC>(y_img_inmat, u_img_inmat, v_img_inmat, imgInput0);
		
	//xf::cv::resize - Resize 8bit BGR image
	xf::cv::resize<INTERPOLATION,TYPE,HEIGHT,WIDTH,NEWHEIGHT,NEWWIDTH,NPC,MAXDOWNSCALE> (imgInput0, out_mat);
	
	//conversion of xf::Mat to stream
	obj.xfMat2hlsStrm<INPUT_PTR_WIDTH, TYPE, NEWHEIGHT, NEWWIDTH, NPC, (NEWWIDTH*NEWHEIGHT/8)>(out_mat, resizeStrmout, srcMat_cols_align_npc);

	//xf::cv::preProcess - Mean-Sub and scaling 
	xf::cv::preProcess <INPUT_PTR_WIDTH, OUTPUT_PTR_WIDTH, T_CHANNELS, CPW, HEIGHT, WIDTH, NPC, PACK_MODE, X_WIDTH, ALPHA_WIDTH, BETA_WIDTH, GAMMA_WIDTH, OUT_WIDTH, X_IBITS, ALPHA_IBITS, BETA_IBITS, GAMMA_IBITS, OUT_IBITS, SIGNED_IN, OPMODE> (resizeStrmout, img_out, params, rows_out, cols_out, th1, th2);

	
}
}
