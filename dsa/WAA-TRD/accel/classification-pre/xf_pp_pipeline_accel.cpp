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
void pp_pipeline_accel(ap_uint<INPUT_PTR_WIDTH>* img_inp,
                       ap_uint<OUTPUT_PTR_WIDTH>* img_out,
                       int rows_in,
                       int cols_in,
                       int rows_out_resize,
                       int cols_out_resize,
                       int rows_out,
                       int cols_out,
                       float params[3 * T_CHANNELS],
                       int th1,
                       int th2) {
// clang-format off
#pragma HLS INTERFACE m_axi     port=img_inp  offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi     port=img_out  offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi     port=params  offset=slave bundle=gmem3

#pragma HLS INTERFACE s_axilite port=rows_in     
#pragma HLS INTERFACE s_axilite port=cols_in     
#pragma HLS INTERFACE s_axilite port=rows_out_resize 
#pragma HLS INTERFACE s_axilite port=cols_out_resize
#pragma HLS INTERFACE s_axilite port=rows_out     
#pragma HLS INTERFACE s_axilite port=cols_out     
#pragma HLS INTERFACE s_axilite port=th1     
#pragma HLS INTERFACE s_axilite port=th2     

#pragma HLS INTERFACE s_axilite port=return
    // clang-format on
    xf::cv::Mat<XF_8UC3, HEIGHT, WIDTH, NPC1> imgInput0(rows_in, cols_in);
// clang-format off
	#pragma HLS stream variable=imgInput0.data depth=2

    xf::cv::Mat<TYPE, NEWHEIGHT, NEWWIDTH, NPC_T> out_mat(rows_out, cols_out);
#pragma HLS stream variable=out_mat.data depth=2
// clang-format on	
	hls::stream<ap_uint<INPUT_PTR_WIDTH> > resizeStrmout;
	int srcMat_cols_align_npc = ((out_mat.cols + (NPC_T - 1)) >> XF_BITSHIFT(NPC_T)) << XF_BITSHIFT(NPC_T);
// clang-format off	
	#pragma HLS DATAFLOW
// clang-format on

	//conversion of pointer to xf::Mat
	xf::cv::Array2xfMat<INPUT_PTR_WIDTH,XF_8UC3,HEIGHT, WIDTH, NPC1>  (img_inp, imgInput0);

	//xf::cv::resize - Resize 8bit BGR image
	xf::cv::resize<INTERPOLATION,TYPE,HEIGHT,WIDTH,NEWHEIGHT,NEWWIDTH,NPC_T,MAXDOWNSCALE> (imgInput0, out_mat);

	xf::cv::accel_utils obj;
	//conversion of xf::Mat to stream
	obj.xfMat2hlsStrm<INPUT_PTR_WIDTH, TYPE, NEWHEIGHT, NEWWIDTH, NPC_T, (NEWWIDTH*NEWHEIGHT/8)>(out_mat, resizeStrmout, srcMat_cols_align_npc);

	//xf::cv::preProcess - Mean-Sub, scaling and int8 to float conversion
	xf::cv::preProcess <INPUT_PTR_WIDTH, OUTPUT_PTR_WIDTH, T_CHANNELS, CPW, HEIGHT, WIDTH, NPC_TEST, PACK_MODE, X_WIDTH, ALPHA_WIDTH, BETA_WIDTH, GAMMA_WIDTH, OUT_WIDTH, X_IBITS, ALPHA_IBITS, BETA_IBITS, GAMMA_IBITS, OUT_IBITS, SIGNED_IN, OPMODE> (resizeStrmout, img_out, params, rows_out, cols_out, th1, th2);

}

}
