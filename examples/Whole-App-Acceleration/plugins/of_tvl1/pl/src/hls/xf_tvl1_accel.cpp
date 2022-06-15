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
#include "xf_tvlonescale.hpp"
#include "xf_tvl1_params.h"

extern "C" {

void tvl1_accel(
		ap_uint<IMAGE_PTR_WIDTH>* _I0,
		ap_uint<IMAGE_PTR_WIDTH>* _I1,
		ap_uint<OUT_PTR_WIDTH>* _flowout,
		ap_uint<INT_PTR_WIDTH>* _U1,
		ap_uint<INT_PTR_WIDTH>* _U2,
		ap_uint<INT_PTR_WIDTH>* _I1wx,
		ap_uint<INT_PTR_WIDTH>* _I1wy,
		ap_uint<INT_PTR_WIDTH>* _grad,
		ap_uint<INT_PTR_WIDTH>* _rhoc,
		ap_uint<INT_PTR_WIDTH>* _p11,
		ap_uint<INT_PTR_WIDTH>* _p12,
		ap_uint<INT_PTR_WIDTH>* _p21,
		ap_uint<INT_PTR_WIDTH>* _p22,
		int *_algo_param,
		int *_design_param,
		int *_debug_error
)
{

	// clang-format off
#pragma HLS INTERFACE m_axi      port=_I0       	offset=slave  bundle=gmem0
#pragma HLS INTERFACE m_axi      port=_I1       	offset=slave  bundle=gmem1
#pragma HLS INTERFACE m_axi      port=_U1       	offset=slave  bundle=gmem2 
#pragma HLS INTERFACE m_axi      port=_U2       	offset=slave  bundle=gmem3 
#pragma HLS INTERFACE m_axi      port=_I1wx     	offset=slave  bundle=gmem4 
#pragma HLS INTERFACE m_axi      port=_I1wy     	offset=slave  bundle=gmem5 
#pragma HLS INTERFACE m_axi      port=_grad     	offset=slave  bundle=gmem6 
#pragma HLS INTERFACE m_axi      port=_rhoc     	offset=slave  bundle=gmem7 
#pragma HLS INTERFACE m_axi      port=_p11     		offset=slave  bundle=gmem9 
#pragma HLS INTERFACE m_axi      port=_p12     		offset=slave  bundle=gmem10 
#pragma HLS INTERFACE m_axi      port=_p21     		offset=slave  bundle=gmem11 
#pragma HLS INTERFACE m_axi      port=_p22     		offset=slave  bundle=gmem12 
#pragma HLS INTERFACE m_axi      port=_flowout     	offset=slave  bundle=gmem13
#pragma HLS INTERFACE m_axi      port=_algo_param     	offset=slave  bundle=gmem14
#pragma HLS INTERFACE m_axi      port=_design_param     offset=slave  bundle=gmem15
#pragma HLS INTERFACE m_axi      port=_debug_error     	offset=slave  bundle=gmem15
#pragma HLS INTERFACE s_axilite  port=return 			          
	// clang-format on

	const int FLOW_BW = XF_PIXELWIDTH(FLOW_TYPE, NPC_TVL_OTHER);

	// Load algorithm param & design paramteter
	int param_reg[16];
	for(int i = 0; i < 16; i++)
	{
	// clang-format off
#pragma HLS pipeline II=1
	// clang-format on
		param_reg[i] = _algo_param[i];
	}
	int design_param_reg[4*MAX_NUM_LEVELS + 1];
	for(int i = 0; i < (4*MAX_NUM_LEVELS + 1); i++)
	{
	// clang-format off
#pragma HLS pipeline II=1
	// clang-format on
		design_param_reg[i] = _design_param[i];
	}

	//////design parameter
	ap_int<FLOW_BW> _lt = param_reg[0]; 
	ap_int<FLOW_BW> _theta = (ap_int<FLOW_BW>)param_reg[1]; 
	ap_int<FLOW_BW> _gamma = (ap_int<FLOW_BW>)param_reg[2];
	ap_int<FLOW_BW> _epsilon = (ap_int<FLOW_BW>)param_reg[3];
	ap_int<FLOW_BW> _taut = (ap_int<FLOW_BW>)param_reg[4];
	ap_int<INVSCALE_BITWIDTH> invscale = (ap_int<INVSCALE_BITWIDTH>)param_reg[5];
	unsigned short _innerIterations = (unsigned short)param_reg[6];
	unsigned short _outerIterations = (unsigned short)param_reg[7];
	unsigned short _warps = (unsigned short)param_reg[8];
	int _U_offset = param_reg[9];
	bool _Error_debug_enable = param_reg[10];
	int nscales = design_param_reg[0];

	//////parameter buffer
	int pyramid_height_reg[MAX_NUM_LEVELS];
	int pyramid_width_reg[MAX_NUM_LEVELS];
	int offset_reg[MAX_NUM_LEVELS];
	int threshold_reg[MAX_NUM_LEVELS];
	int org_height = design_param_reg[1];
	int org_width  = design_param_reg[2];
	for(int i = 0, i3 = 4*(nscales+1); i < nscales; i++, i3-=4)
	{
	// clang-format off
#pragma HLS pipeline II=1
	// clang-format on
		pyramid_height_reg[i] 	= design_param_reg[i3-3];
		pyramid_width_reg[i] 	= design_param_reg[i3-2];
		offset_reg[i] 		= design_param_reg[i3-1];
		threshold_reg[i] 	= design_param_reg[i3];
	}

	//local variable
	int U_height = pyramid_height_reg[0];
	int U_width = pyramid_width_reg[0];
	int height,width,width_ncpr,height_upscale,width_upscale,width_upscale_ncpr,offset_img ;
	int error_count = 0;
	bool switch_U1U2_offset = 0;
	int ping_U_offset = 0;
	int pong_U_offset = _U_offset;

	//intialize U1 & U2
	initU<INT_PTR_WIDTH, FLOW_TYPE, NPC_TVL_OTHER>(_U1,_U2 , U_height, U_width);

	for (int s = nscales - 1, n = 0; s >= 0; --s, ++n)
	{
	// clang-format off
#pragma HLS LOOP_TRIPCOUNT min=5 max=5
	// clang-format on

		height = pyramid_height_reg[n];
		width = pyramid_width_reg[n];
		width_ncpr = (width + (NPC_TVL_OTHER - 1)) >> XF_BITSHIFT(NPC_TVL_OTHER);
		if(s!=0)
		{
		   height_upscale = pyramid_height_reg[n+1];
		   width_upscale  = pyramid_width_reg[n+1];
		}
		else
		{
		   height_upscale = org_height;
		   width_upscale  = org_width;
		}
		offset_img = offset_reg[n];
		ap_int<ERROR_BW> tvl_threshold = (ap_int<ERROR_BW>)threshold_reg[n];

		if(switch_U1U2_offset==0)
		{
			TVLonescale_n_resizeScaleU< IMAGE_PTR_WIDTH, INT_PTR_WIDTH, IMG_TYPE, FLOW_TYPE, IMG_F_BITS, FLOW_F_BITS, HEIGHT, WIDTH, NPC_TVL_OTHER, NPC_TVL_INNERCORE, ERROR_BW, MEDIANBLURFILTERING, INVSCALE_BITWIDTH, MAX_FLOW_VALUE, FLOW_BW, MB_INST>
			(_I0+offset_img, _I1+offset_img, _U1+ping_U_offset, _U2+ping_U_offset,
				_I1wx,_I1wy,_grad,_rhoc,
				_p11,_p12,_p21,_p22,
				_U1+pong_U_offset, _U2+pong_U_offset,
				_lt,_theta,_gamma, _taut,
				_innerIterations,_outerIterations,_warps,height,
				width, height_upscale,
				width_upscale, invscale, s, tvl_threshold, _debug_error, &error_count,_Error_debug_enable);
		}
		else
		{	
			TVLonescale_n_resizeScaleU< IMAGE_PTR_WIDTH, INT_PTR_WIDTH, IMG_TYPE, FLOW_TYPE, IMG_F_BITS, FLOW_F_BITS, HEIGHT, WIDTH, NPC_TVL_OTHER, NPC_TVL_INNERCORE, ERROR_BW, MEDIANBLURFILTERING, INVSCALE_BITWIDTH, MAX_FLOW_VALUE, FLOW_BW, MB_INST>
			(_I0+offset_img, _I1+offset_img, _U1+pong_U_offset, _U2+pong_U_offset,
				_I1wx,_I1wy,_grad,_rhoc,
				_p11,_p12,_p21,_p22,
				_U1+ping_U_offset, _U2+ping_U_offset,
				_lt,_theta,_gamma, _taut,
				_innerIterations,_outerIterations,_warps,height,
				width, height_upscale,
				width_upscale, invscale, s, tvl_threshold, _debug_error, &error_count, _Error_debug_enable);


		}

		switch_U1U2_offset = !switch_U1U2_offset;
	}// End s for loop

	if(switch_U1U2_offset==1)
	{
	    merge_out<INT_PTR_WIDTH, OUT_PTR_WIDTH, FLOW_TYPE,  HEIGHT, WIDTH, NPC_TVL_OTHER, FLOW_F_BITS>(_U1+pong_U_offset, _U2+pong_U_offset, _flowout, org_height, org_width);
	}
	else
	{
	    merge_out<INT_PTR_WIDTH, OUT_PTR_WIDTH, FLOW_TYPE,  HEIGHT, WIDTH, NPC_TVL_OTHER, FLOW_F_BITS>(_U1+ping_U_offset, _U2+ping_U_offset, _flowout, org_height, org_width);
	}


	return;
} // End of kernel
}//extern
