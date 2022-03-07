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

#include "common/xf_common.hpp"
#include "common/xf_utility.hpp"
#include "common/xf_extra_utility.hpp"
#include "imgproc/xf_resize.hpp"
#include "xf_img_pyramid_params.h"

namespace xf {
namespace cv {
void ResizeStageN (
		// Resize image pointers
		ap_uint<IMAGE_PTR_WIDTH_P> *resize_img_i,
		ap_uint<IMAGE_PTR_WIDTH_P> *resize_img_o,
		// Parameters
		int     rows_i,
		int     cols_i,
		int     rows_o,
		int     cols_o) 
{
	// clang-format off
#pragma HLS INLINE OFF
	// clang-format on

	xf::cv::Mat<IMG_TYPE_P, HEIGHT, WIDTH, NPC_PYRAMID> in_mat(rows_i,cols_i);
	xf::cv::Mat<IMG_TYPE_P, NEWHEIGHT, NEWWIDTH, NPC_PYRAMID> out_mat(rows_o,cols_o);
	// clang-format off
#pragma HLS DATAFLOW
	// clang-format on
	xf::cv::Array2xfMat<IMAGE_PTR_WIDTH_P, IMG_TYPE_P, HEIGHT, WIDTH, NPC_PYRAMID>(resize_img_i, in_mat);
	xf::cv::resize<INTERPOLATION_P, IMG_TYPE_P, HEIGHT, WIDTH, NEWHEIGHT, NEWWIDTH, NPC_PYRAMID, MAXDOWNSCALE_P>(in_mat, out_mat);
	xf::cv::xfMat2Array<IMAGE_PTR_WIDTH_P, IMG_TYPE_P, NEWHEIGHT, NEWWIDTH, NPC_PYRAMID>(out_mat, resize_img_o);
} // ResizeStageN
} // namespace cv
} // namespace xf

extern "C" {
void img_pyramid_accel(ap_uint<IMAGE_PTR_WIDTH_P>* img_inp,
		ap_uint<IMAGE_PTR_WIDTH_P>* img_out,
		uint32_t *params) {
	// clang-format off
#pragma HLS INTERFACE m_axi     port=img_inp  offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi     port=img_out  offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi     port=params   offset=slave bundle=gmem1
#pragma HLS INTERFACE s_axilite port=return 
	// clang-format on

	uint32_t pyr_levels;
	const int TC_MAX_NUM_LEVELS    = 5; // maximum supported levels
	const int TC_MAX_NUM_LEVELS_P_1 = 5+1;

	pyr_levels = params[0];
	uint32_t img_rows[MAX_NUM_LEVELS_P+1], img_cols[MAX_NUM_LEVELS_P+1];
	uint32_t offsets[MAX_NUM_LEVELS_P+1];

	PARAMS_READ_LOOP : for (uint16_t i=0; i<=pyr_levels; i++) {
	// clang-format off
#pragma HLS LOOP_TRIPCOUNT min=TC_MAX_NUM_LEVELS_P_1 max=TC_MAX_NUM_LEVELS_P_1
	// clang-format on
		img_rows[i] = params[(4*i)+1];
		img_cols[i] = params[(4*i)+2];
		offsets[i] = params[(4*i)+3];
		int tvl_threshold = params[(4*i)+4];

	} // PARAMS_READ_LOOP : for...

	uint8_t  *u_tmp_img_inp;
	uint8_t  *u_tmp_img_out;
	ap_uint<IMAGE_PTR_WIDTH_P>* ap_tmp_img_inp;
	ap_uint<IMAGE_PTR_WIDTH_P>* ap_tmp_img_out;
	u_tmp_img_inp = reinterpret_cast <uint8_t *>(img_inp);
	u_tmp_img_out = reinterpret_cast <uint8_t *>(img_out);
	KERNEL_CALL_LOOP : for (uint8_t i=0; i<pyr_levels; i++) {
	// clang-format off
#pragma HLS LOOP_TRIPCOUNT min=TC_MAX_NUM_LEVELS max=TC_MAX_NUM_LEVELS
	// clang-format on
		xf::cv::ResizeStageN(img_inp+offsets[i],img_out+offsets[i+1],img_rows[i], img_cols[i], img_rows[i+1], img_cols[i+1]);

	} // KERNEL_CALL_LOOP : for...

} // createimgpyramid_accel
} // extern "C"
