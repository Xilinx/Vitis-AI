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
#include "tvl1_subfunct.hpp"
#include "imgproc/xf_median_blur.hpp"
#include "imgproc/xf_resize.hpp"

#define _MBFUNCT_TPLT_DEC template<int PTR_WIDTH = 16, int FLOW_T = XF_16SC1, int ROWS = 126, int COLS = 224, int NPC = 1, int MEDIANFILTERING = 5, int MBINST = 2>
#define _PPFUNCT_TPLT_DEC template<int PTR_WIDTH = 16, int IMG_FBITS = 0, int FLOW_FBITS = 8, int IMG_T = XF_8UC1,int FLOW_T = XF_16SC1, int ROWS = 126, int COLS = 224, int NPC = 1, int ERR_BW = 32, int FLOW_BW>
#define _WGFUNCT_TPLT_DEC template<int IMG_PTRWIDTH = 8, int FLOW_PTRWIDTH = 16, int IMG_FBITS = 0, int FLOW_FBITS = 8, int IMG_T = XF_8UC1,int FLOW_T = XF_16SC1, int ROWS = 126, int COLS = 224, int NPC = 1, int ERR_BW = 32, int MFV = 2>
#define _TVL1FUNCT_TPLT_DEC template<int IMG_PTRWIDTH = 8, int FLOW_PTRWIDTH = 16, int IMG_T, int FLOW_T, int IMG_FBITS, int FLOW_FBITS, int ROWS, int COLS, int NPC = 1, int NPCTVLINNERCORE = 1, int ERR_BW = 32,int MEDIANFILTERSIZE=5, int INVSCALE_BW, int MFV = 2, int FLOW_BW, int MBINST>
#define _MRGFUNCT_TPLT_DEC template <int FLOW_PTRWIDTH = 32, int OUT_PTRWIDTH = 64, int FLOW_T, int ROWS, int COLS, int NPC, int FLOW_FBITS>
#define _RSFUNCT_TPLT_DEC template<int FLOW_PTRWIDTH, int FLOW_T, int FLOW_FB, int ROWS, int COLS, int NPC, int SCALE_BW>

_MBFUNCT_TPLT_DEC void medianBlurFunct_1inst(
		ap_uint<PTR_WIDTH>* _U1,
		unsigned short height,
		unsigned short width)
{
	// clang-format off
#pragma HLS inline off
	// clang-format on

	xf::cv::Mat<FLOW_T, ROWS, COLS, NPC> U1_in(height, width);
	xf::cv::Mat<FLOW_T, ROWS, COLS, NPC> U1_out(height, width);

	// clang-format off
#pragma HLS dataflow
	// clang-format on

	xf::cv::Array2xfMat<PTR_WIDTH, FLOW_T, ROWS, COLS, NPC>(_U1, U1_in);

	xf::cv::medianBlur<MEDIANFILTERING, XF_BORDER_REPLICATE, FLOW_T, ROWS, COLS, NPC>(U1_in, U1_out);

	xf::cv::xfMat2Array<PTR_WIDTH, FLOW_T, ROWS, COLS, NPC>(U1_out,  _U1);

	return ;
}

_MBFUNCT_TPLT_DEC void medianBlurFunct(
		ap_uint<PTR_WIDTH>* _U1,
		ap_uint<PTR_WIDTH>* _U2,
		unsigned short height,
		unsigned short width)
{
	// clang-format off
#pragma HLS inline off
	// clang-format on

     	if(MBINST == 1)
     	{
	    for(int inst = 0;inst<2;inst++)
	    {
	        if(inst==0)
		    medianBlurFunct_1inst<PTR_WIDTH, FLOW_T, ROWS, COLS, NPC,  MEDIANFILTERING>(_U1, height, width);
	        else	
		    medianBlurFunct_1inst<PTR_WIDTH, FLOW_T, ROWS, COLS, NPC,  MEDIANFILTERING>(_U2, height, width);
	    }
     	}
     	else
     	{
	    xf::cv::Mat<FLOW_T, ROWS, COLS, NPC> U1_in(height, width);
	    xf::cv::Mat<FLOW_T, ROWS, COLS, NPC> U2_in(height, width);
	    xf::cv::Mat<FLOW_T, ROWS, COLS, NPC> U1_out(height, width);
	    xf::cv::Mat<FLOW_T, ROWS, COLS, NPC> U2_out(height, width);

	// clang-format off
#pragma HLS dataflow
	// clang-format on

	    xf::cv::Array2xfMat<PTR_WIDTH, FLOW_T, ROWS, COLS, NPC>(_U1, U1_in);
	    xf::cv::Array2xfMat<PTR_WIDTH, FLOW_T, ROWS, COLS, NPC>(_U2, U2_in);

	    xf::cv::medianBlur<MEDIANFILTERING, XF_BORDER_REPLICATE, FLOW_T, ROWS, COLS, NPC>(U1_in, U1_out);
	    xf::cv::medianBlur<MEDIANFILTERING, XF_BORDER_REPLICATE, FLOW_T, ROWS, COLS, NPC>(U2_in, U2_out);

	    xf::cv::xfMat2Array<PTR_WIDTH, FLOW_T, ROWS, COLS, NPC>(U1_out,  _U1);
	    xf::cv::xfMat2Array<PTR_WIDTH, FLOW_T, ROWS, COLS, NPC>(U2_out,  _U2);
      	}
	return ;
}

_PPFUNCT_TPLT_DEC void TVL_INNER_Core(
		ap_uint<PTR_WIDTH>* _I1wx,
		ap_uint<PTR_WIDTH>* _I1wy,
		ap_uint<PTR_WIDTH>* _grad,
		ap_uint<PTR_WIDTH>* _rhoc,
		ap_uint<PTR_WIDTH>* _U1,
		ap_uint<PTR_WIDTH>* _U2,
		ap_uint<PTR_WIDTH>* _p11,
		ap_uint<PTR_WIDTH>* _p12,
		ap_uint<PTR_WIDTH>* _p21,
		ap_uint<PTR_WIDTH>* _p22,
		ap_int<ERR_BW> *error,
		ap_int<FLOW_BW> theta,
		ap_int<FLOW_BW> gamma,
		ap_int<FLOW_BW> lt,
		ap_int<FLOW_BW> taut,
		int height,
		int width,
		bool intilize_P_with_zeros
)
{
	// clang-format off
#pragma HLS inline off
	// clang-format on

	int width_ncpr = (width + (NPC - 1)) >> XF_BITSHIFT(NPC);
	//input Mat obj
	xf::cv::Mat<FLOW_T, ROWS, COLS, NPC> I1wx(height, width);
	xf::cv::Mat<FLOW_T, ROWS, COLS, NPC> I1wy(height, width);
	xf::cv::Mat<FLOW_T, ROWS, COLS, NPC> U1_in(height, width);
	xf::cv::Mat<FLOW_T, ROWS, COLS, NPC> U2_in(height, width);
	xf::cv::Mat<FLOW_T, ROWS, COLS, NPC> grad(height, width);
	xf::cv::Mat<FLOW_T, ROWS, COLS, NPC> rhoc(height, width);
	xf::cv::Mat<FLOW_T, ROWS, COLS, NPC> p11_in(height, width);
	xf::cv::Mat<FLOW_T, ROWS, COLS, NPC> p12_in(height, width);
	xf::cv::Mat<FLOW_T, ROWS, COLS, NPC> p21_in(height, width);
	xf::cv::Mat<FLOW_T, ROWS, COLS, NPC> p22_in(height, width);
	//output Mat obj
	xf::cv::Mat<FLOW_T, ROWS, COLS, NPC> p11_out(height, width);
	xf::cv::Mat<FLOW_T, ROWS, COLS, NPC> p12_out(height, width);
	xf::cv::Mat<FLOW_T, ROWS, COLS, NPC> p21_out(height, width);
	xf::cv::Mat<FLOW_T, ROWS, COLS, NPC> p22_out(height, width);
	//intermediate result Mat obj
	xf::cv::Mat<FLOW_T, ROWS, COLS, NPC> V1(height, width);
	xf::cv::Mat<FLOW_T, ROWS, COLS, NPC> V2(height, width);
	xf::cv::Mat<FLOW_T, ROWS, COLS, NPC> divP1(height, width);
	xf::cv::Mat<FLOW_T, ROWS, COLS, NPC> divP2(height, width);
	xf::cv::Mat<FLOW_T, ROWS, COLS, NPC> U1x(height, width);
	xf::cv::Mat<FLOW_T, ROWS, COLS, NPC> U1y(height, width);
	xf::cv::Mat<FLOW_T, ROWS, COLS, NPC> U2x(height, width);
	xf::cv::Mat<FLOW_T, ROWS, COLS, NPC> U2y(height, width);
	xf::cv::Mat<FLOW_T, ROWS, COLS, NPC> U1_passEV(height, width);
	xf::cv::Mat<FLOW_T, ROWS, COLS, NPC> U2_passEV(height, width);
	xf::cv::Mat<FLOW_T, ROWS, COLS, NPC> U1_out(height, width);
	xf::cv::Mat<FLOW_T, ROWS, COLS, NPC> U1_out_ddr(height, width);
	xf::cv::Mat<FLOW_T, ROWS, COLS, NPC> U2_out(height, width);
	xf::cv::Mat<FLOW_T, ROWS, COLS, NPC> U2_out_ddr(height, width);
	xf::cv::Mat<FLOW_T, ROWS, COLS, NPC> p11_passDIV(height, width);
	xf::cv::Mat<FLOW_T, ROWS, COLS, NPC> p12_passDIV(height, width);
	xf::cv::Mat<FLOW_T, ROWS, COLS, NPC> p21_passDIV(height, width);
	xf::cv::Mat<FLOW_T, ROWS, COLS, NPC> p22_passDIV(height, width);
	xf::cv::Mat<FLOW_T, ROWS, COLS, NPC> p11_passUU(height, width);
	xf::cv::Mat<FLOW_T, ROWS, COLS, NPC> p12_passUU(height, width);
	xf::cv::Mat<FLOW_T, ROWS, COLS, NPC> p21_passUU(height, width);
	xf::cv::Mat<FLOW_T, ROWS, COLS, NPC> p22_passUU(height, width);
	xf::cv::Mat<FLOW_T, ROWS, COLS, NPC> p11_passFG(height, width);
	xf::cv::Mat<FLOW_T, ROWS, COLS, NPC> p12_passFG(height, width);
	xf::cv::Mat<FLOW_T, ROWS, COLS, NPC> p21_passFG(height, width);
	xf::cv::Mat<FLOW_T, ROWS, COLS, NPC> p22_passFG(height, width);
	// clang-format off
#pragma HLS dataflow
	// clang-format on

	xf::cv::Ptr2xfMat<PTR_WIDTH, FLOW_T, ROWS, COLS, NPC>(_I1wx, I1wx);
	xf::cv::Ptr2xfMat<PTR_WIDTH, FLOW_T, ROWS, COLS, NPC>(_I1wy, I1wy);
	xf::cv::Ptr2xfMat<PTR_WIDTH, FLOW_T, ROWS, COLS, NPC>(_U1, U1_in);
	xf::cv::Ptr2xfMat<PTR_WIDTH, FLOW_T, ROWS, COLS, NPC>(_U2, U2_in);
	xf::cv::Ptr2xfMat<PTR_WIDTH, FLOW_T, ROWS, COLS, NPC>(_grad, grad);
	xf::cv::Ptr2xfMat<PTR_WIDTH, FLOW_T, ROWS, COLS, NPC>(_rhoc, rhoc);
	xf::cv::Ptr2xfMat<PTR_WIDTH, FLOW_T, ROWS, COLS, NPC>(_p11, p11_in);
	xf::cv::Ptr2xfMat<PTR_WIDTH, FLOW_T, ROWS, COLS, NPC>(_p12, p12_in);
	xf::cv::Ptr2xfMat<PTR_WIDTH, FLOW_T, ROWS, COLS, NPC>(_p21, p21_in);
	xf::cv::Ptr2xfMat<PTR_WIDTH, FLOW_T, ROWS, COLS, NPC>(_p22, p22_in);


	estimateV<IMG_FBITS, FLOW_FBITS, IMG_T, FLOW_T, ROWS, COLS, NPC, ERR_BW, FLOW_BW>
	(I1wx, I1wy, U1_in, U2_in, U1_passEV, U2_passEV, grad, rhoc, V1, V2, lt,  height, width_ncpr);

	divergence<IMG_FBITS, FLOW_FBITS, IMG_T, FLOW_T, ROWS, COLS, NPC, ERR_BW>
	(p11_in, p12_in, divP1, p11_passDIV, p12_passDIV, height, width_ncpr, intilize_P_with_zeros);

	divergence<IMG_FBITS, FLOW_FBITS, IMG_T, FLOW_T, ROWS, COLS, NPC, ERR_BW>
	(p21_in, p22_in, divP2, p21_passDIV, p22_passDIV, height, width_ncpr, intilize_P_with_zeros);

	updateU<IMG_FBITS, FLOW_FBITS, IMG_T, FLOW_T, ROWS, COLS, NPC, ERR_BW, FLOW_BW>
	(V1, V2 , divP1, divP2, U1_passEV, U2_passEV, U1_out, U2_out, U1_out_ddr, U2_out_ddr, p11_passDIV, p12_passDIV, p21_passDIV, p22_passDIV, p11_passUU, p12_passUU, p21_passUU, p22_passUU,  error, theta, height, width_ncpr);

	forwardgradient<IMG_FBITS, FLOW_FBITS, IMG_T, FLOW_T, ROWS, COLS, NPC, ERR_BW>
	(U1_out, U1x, U1y, p11_passUU, p12_passUU, p11_passFG, p12_passFG,  height, width_ncpr);

	forwardgradient<IMG_FBITS, FLOW_FBITS, IMG_T, FLOW_T, ROWS, COLS, NPC, ERR_BW>
	(U2_out, U2x, U2y, p21_passUU, p22_passUU, p21_passFG, p22_passFG, height, width_ncpr);

	estimatedualvariables<IMG_FBITS, FLOW_FBITS, IMG_T, FLOW_T, ROWS, COLS, NPC, ERR_BW>
	( U1x, U1y, U2x, U2y, p11_passFG, p12_passFG, p21_passFG, p22_passFG,
			p11_out, p12_out, p21_out, p22_out, taut, intilize_P_with_zeros, height, width_ncpr);

	xf::cv::xfMat2Ptr<PTR_WIDTH, FLOW_T, ROWS, COLS, NPC>(U1_out_ddr,  _U1);
	xf::cv::xfMat2Ptr<PTR_WIDTH, FLOW_T, ROWS, COLS, NPC>(U2_out_ddr,  _U2);
	xf::cv::xfMat2Ptr<PTR_WIDTH, FLOW_T, ROWS, COLS, NPC>(p11_out,  _p11);
	xf::cv::xfMat2Ptr<PTR_WIDTH, FLOW_T, ROWS, COLS, NPC>(p12_out,  _p12);
	xf::cv::xfMat2Ptr<PTR_WIDTH, FLOW_T, ROWS, COLS, NPC>(p21_out,  _p21);
	xf::cv::xfMat2Ptr<PTR_WIDTH, FLOW_T, ROWS, COLS, NPC>(p22_out,  _p22);

	return;
}

_WGFUNCT_TPLT_DEC void warpGradientFunct(
		ap_uint<FLOW_PTRWIDTH>* _U1,
		ap_uint<FLOW_PTRWIDTH>* _U2,
		ap_uint<IMG_PTRWIDTH>* _I0,
		ap_uint<IMG_PTRWIDTH>* _I1,
		ap_uint<FLOW_PTRWIDTH>* _I1wx,
		ap_uint<FLOW_PTRWIDTH>* _I1wy,
		ap_uint<FLOW_PTRWIDTH>* _grad,
		ap_uint<FLOW_PTRWIDTH>* _rhoc,
		unsigned short height,
		unsigned short width)
{
	// clang-format off
#pragma HLS inline off
	// clang-format on

	enum{
		REMAP_FILTER_WIDTH = 2*MFV + 4
	};

	const int CNG_DELAY = COLS + 100;
	const int REMAP_DELAY = REMAP_FILTER_WIDTH*COLS + (REMAP_FILTER_WIDTH/2) + 100;
	const int CNG_REMAP_DELAY = COLS + REMAP_FILTER_WIDTH*COLS + (REMAP_FILTER_WIDTH/2) + 200;
	const int TC_DELAY = 1 + 100;

	//input Mat obj
	xf::cv::Mat<FLOW_T, ROWS, COLS, NPC> U1(height, width);
	xf::cv::Mat<FLOW_T, ROWS, COLS, NPC> U2(height, width);
	xf::cv::Mat<IMG_T, ROWS, COLS, NPC> I0(height, width);
	xf::cv::Mat<IMG_T, ROWS, COLS, NPC> I1(height, width);

	//output Mat obj
	xf::cv::Mat<FLOW_T, ROWS, COLS, NPC> I1wx(height, width);
	xf::cv::Mat<FLOW_T, ROWS, COLS, NPC> I1wy(height, width);
	xf::cv::Mat<FLOW_T, ROWS, COLS, NPC> grad(height, width);
	xf::cv::Mat<FLOW_T, ROWS, COLS, NPC> rhoc(height, width);

	//intermediate result Mat obj
	xf::cv::Mat<FLOW_T, ROWS, COLS, NPC> I1x(height, width);
	xf::cv::Mat<FLOW_T, ROWS, COLS, NPC> I1y(height, width);
	xf::cv::Mat<FLOW_T, ROWS, COLS, NPC, CNG_DELAY> I1w(height, width);
	xf::cv::Mat<IMG_T, ROWS, COLS, NPC> I1_copy1(height, width);
	xf::cv::Mat<IMG_T, ROWS, COLS, NPC> I1_copy2(height, width);
	xf::cv::Mat<FLOW_T, ROWS, COLS, NPC> I1_flowtype(height, width);
	xf::cv::Mat<FLOW_T, ROWS, COLS, NPC, TC_DELAY> U1_copy1(height, width);
	xf::cv::Mat<FLOW_T, ROWS, COLS, NPC, CNG_DELAY> U1_copy2(height, width);
	xf::cv::Mat<FLOW_T, ROWS, COLS, NPC, CNG_DELAY> U1_copy3(height, width);
	xf::cv::Mat<FLOW_T, ROWS, COLS, NPC, CNG_REMAP_DELAY> U1_copy4(height, width);
	xf::cv::Mat<FLOW_T, ROWS, COLS, NPC, TC_DELAY> U2_copy1(height, width);
	xf::cv::Mat<FLOW_T, ROWS, COLS, NPC, CNG_DELAY> U2_copy2(height, width);
	xf::cv::Mat<FLOW_T, ROWS, COLS, NPC, CNG_DELAY> U2_copy3(height, width);
	xf::cv::Mat<FLOW_T, ROWS, COLS, NPC, CNG_REMAP_DELAY> U2_copy4(height, width);
	xf::cv::Mat<FLOW_T, ROWS, COLS, NPC> I1wx_copy1(height, width);
	xf::cv::Mat<FLOW_T, ROWS, COLS, NPC> I1wy_copy1(height, width);
	xf::cv::Mat<FLOW_T, ROWS, COLS, NPC> I1wx_copy2(height, width);
	xf::cv::Mat<FLOW_T, ROWS, COLS, NPC> I1wy_copy2(height, width);

	// clang-format off
#pragma HLS dataflow
	// clang-format on

	xf::cv::Array2xfMat<FLOW_PTRWIDTH, FLOW_T, ROWS, COLS, NPC>(_U1, U1);
	xf::cv::Array2xfMat<FLOW_PTRWIDTH, FLOW_T, ROWS, COLS, NPC>(_U2, U2);
	xf::cv::Array2xfMat<IMG_PTRWIDTH, IMG_T, ROWS, COLS, NPC>(_I0, I0);
	xf::cv::Array2xfMat<IMG_PTRWIDTH, IMG_T, ROWS, COLS, NPC>(_I1, I1);

	dupMat<FLOW_T, ROWS , COLS, NPC, TC_DELAY, CNG_DELAY, CNG_DELAY, CNG_REMAP_DELAY>
	(U1, U1_copy1, U1_copy2, U1_copy3, U1_copy4, height, width);

	dupMat<FLOW_T, ROWS , COLS, NPC, TC_DELAY, CNG_DELAY, CNG_DELAY, CNG_REMAP_DELAY>
	(U2, U2_copy1, U2_copy2, U2_copy3, U2_copy4, height, width);

	dupMat<IMG_T, ROWS , COLS, NPC>
	(I1, I1_copy1, I1_copy2, height, width);

	centeredgradient<IMG_FBITS , FLOW_FBITS, IMG_T , FLOW_T, ROWS , COLS, NPC, ERR_BW>
	(I1_copy1, I1x, I1y, height, width);

	ConvertType<IMG_T,FLOW_T,ROWS,COLS,NPC,IMG_FBITS,FLOW_FBITS>(I1_copy2, I1_flowtype, height, width);
	Remap_Bicubic<FLOW_T, FLOW_T, FLOW_T,  ROWS, COLS, REMAP_FILTER_WIDTH, NPC, XF_BORDER_REFLECT_101, 0, MFV, FLOW_FBITS, FLOW_FBITS, TC_DELAY, CNG_DELAY>
	(I1_flowtype, U1_copy1, U2_copy1, I1w);
	Remap_Bicubic<FLOW_T, FLOW_T, FLOW_T,  ROWS, COLS, REMAP_FILTER_WIDTH, NPC, XF_BORDER_REFLECT_101, 0, MFV, FLOW_FBITS, FLOW_FBITS, CNG_DELAY>
	(I1x, U1_copy2, U2_copy2, I1wx);
	Remap_Bicubic<FLOW_T, FLOW_T, FLOW_T,  ROWS, COLS, REMAP_FILTER_WIDTH, NPC, XF_BORDER_REFLECT_101, 0, MFV, FLOW_FBITS, FLOW_FBITS, CNG_DELAY>
	(I1y, U1_copy3, U2_copy3, I1wy);

	dupMat<FLOW_T, ROWS , COLS, NPC>
	(I1wx, I1wx_copy1, I1wx_copy2, height, width);

	dupMat<FLOW_T, ROWS , COLS, NPC>
	(I1wy, I1wy_copy1, I1wy_copy2, height, width);

	calcgrad<IMG_FBITS , FLOW_FBITS, IMG_T , FLOW_T, ROWS , COLS, NPC, ERR_BW, CNG_DELAY, CNG_REMAP_DELAY>
	(I0, I1w, I1wx_copy1, I1wy_copy1, U1_copy4, U2_copy4, grad, rhoc, height, width);

	xf::cv::xfMat2Array<FLOW_PTRWIDTH, FLOW_T, ROWS, COLS, NPC>(I1wx_copy2,  _I1wx);
	xf::cv::xfMat2Array<FLOW_PTRWIDTH, FLOW_T, ROWS, COLS, NPC>(I1wy_copy2,  _I1wy);
	xf::cv::xfMat2Array<FLOW_PTRWIDTH, FLOW_T, ROWS, COLS, NPC>(grad,  _grad);
	xf::cv::xfMat2Array<FLOW_PTRWIDTH, FLOW_T, ROWS, COLS, NPC>(rhoc,  _rhoc);

	return;
}

void Write_TvlError(int l1, int l2, int l3, int l4, int tvlerror, int * _debug_error, int * error_count)
{
	// clang-format off
#pragma HLS inline off
	// clang-format on

	int start_idx = *error_count;
	*error_count = start_idx+5;
	_debug_error[start_idx]   = l1;
	_debug_error[start_idx+1] = l2;
	_debug_error[start_idx+2] = l3;
	_debug_error[start_idx+3] = l4;
	_debug_error[start_idx+4] = tvlerror;
}

_TVL1FUNCT_TPLT_DEC void TVLonescale(
		ap_uint<IMG_PTRWIDTH>* _I0,
		ap_uint<IMG_PTRWIDTH>* _I1,
		ap_uint<FLOW_PTRWIDTH>* _U1,
		ap_uint<FLOW_PTRWIDTH>* _U2,
		ap_uint<FLOW_PTRWIDTH>* _I1wx,
		ap_uint<FLOW_PTRWIDTH>* _I1wy,
		ap_uint<FLOW_PTRWIDTH>* _grad,
		ap_uint<FLOW_PTRWIDTH>* _rhoc,
		ap_uint<FLOW_PTRWIDTH>* _p11,
		ap_uint<FLOW_PTRWIDTH>* _p12,
		ap_uint<FLOW_PTRWIDTH>* _p21,
		ap_uint<FLOW_PTRWIDTH>* _p22,
		ap_int<FLOW_BW> _lt,
		ap_int<FLOW_BW> _theta,
		ap_int<FLOW_BW> _gamma,
		ap_int<FLOW_BW> _taut,
		unsigned short _innerIterations,
		unsigned short _outerIterations,
		unsigned short _warps,
		int height,
		int width,
		int s_loopcnt,
		ap_int<ERR_BW> tvl_threshold,
		int * _debug_error, int * error_count, bool _Error_debug_enable )
{
	// clang-format off
#pragma HLS inline off
	// clang-format on

	bool intilize_P_with_zeros=1;

	for (int warpings = 0; warpings < _warps; ++warpings)
	{
	// clang-format off
#pragma HLS LOOP_TRIPCOUNT min=5 max=5
	// clang-format on

	    //input: _U1, _U2, _I0, _I1
	    //output:  _I1wx, _I1wy, _grad, _rhoc
	    warpGradientFunct<IMG_PTRWIDTH, FLOW_PTRWIDTH, IMG_FBITS, FLOW_FBITS, IMG_T,FLOW_T ,ROWS, COLS, NPC, ERR_BW, MFV>
	    (_U1, _U2, _I0, _I1, _I1wx, _I1wy, _grad, _rhoc, height, width);

	    //init error with max value.
	    ap_int<ERR_BW> error = (1<<(ERR_BW-1) - 1);
	    ap_int<ERR_BW> error_prev = (1<<(ERR_BW-1) - 1);


	    for (int n_outer = 0; error > tvl_threshold && n_outer < _outerIterations && error_prev >= error; ++n_outer)
	    {
	// clang-format off
#pragma HLS LOOP_TRIPCOUNT min=10 max=10
	// clang-format on
					
	        if (MEDIANFILTERSIZE > 1) {
		    //input: _U1, _U2
		    //output: _U1, _U2
		    medianBlurFunct<FLOW_PTRWIDTH, FLOW_T, ROWS, COLS, NPC, MEDIANFILTERSIZE, MBINST>
		    (_U1, _U2, height, width);
	        }

	        error_prev = (1<<(ERR_BW-1) - 1); 
	        for (int n_inner = 0; error > tvl_threshold && n_inner < _innerIterations && error_prev >= error; ++n_inner)
	        {
	// clang-format off
#pragma HLS LOOP_TRIPCOUNT min=30 max=30
	// clang-format on

		    error_prev = error;

		    //input: _I1wx, _I1wy, _grad, _rhoc, _U1, _U2, _p11, _p12, _p21, _p22, error
		    //output: _U1, _U2, _p11, _p12, _p21, _p22, error
		    TVL_INNER_Core<FLOW_PTRWIDTH, IMG_FBITS, FLOW_FBITS, IMG_T, FLOW_T, ROWS, COLS,NPCTVLINNERCORE, ERR_BW, FLOW_BW>
		    (_I1wx, _I1wy, _grad, _rhoc, _U1, _U2, _p11, _p12, _p21, _p22,
			&error, _theta, _gamma, _lt, _taut, height, width, intilize_P_with_zeros);
		    intilize_P_with_zeros = 0;
		
		    if(_Error_debug_enable)
		        Write_TvlError(s_loopcnt, warpings, n_outer, n_inner, (int)error, _debug_error, error_count);

		}//inner loop
	    }//outer loop
	}// warps loop

}

_RSFUNCT_TPLT_DEC void resizeScale_flow(
		ap_uint<FLOW_PTRWIDTH>* _U1_in,
		ap_uint<FLOW_PTRWIDTH>* _U2_in,
		ap_uint<FLOW_PTRWIDTH>* _U1_out,
		ap_uint<FLOW_PTRWIDTH>* _U2_out,
		ap_int<SCALE_BW> invscale,
		unsigned short height,
		unsigned short width,
		unsigned short newheight,
		unsigned short newwidth
)
{
	// clang-format off
#pragma HLS inline off
	// clang-format on

	xf::cv::Mat<FLOW_T, ROWS, COLS, NPC> U1_in(height, width);
	xf::cv::Mat<FLOW_T, ROWS, COLS, NPC> U2_in(height, width);
	xf::cv::Mat<FLOW_T, ROWS, COLS, NPC> U1_out(newheight, newwidth);
	xf::cv::Mat<FLOW_T, ROWS, COLS, NPC> U2_out(newheight, newwidth);

	xf::cv::Mat<FLOW_T, ROWS, COLS, NPC> U1_resize(newheight, newwidth);
	xf::cv::Mat<FLOW_T, ROWS, COLS, NPC> U2_resize(newheight, newwidth);

	// clang-format off
#pragma HLS dataflow
	// clang-format on

	xf::cv::Array2xfMat<FLOW_PTRWIDTH, FLOW_T, ROWS, COLS, NPC>(_U1_in, U1_in);
	xf::cv::Array2xfMat<FLOW_PTRWIDTH, FLOW_T, ROWS, COLS, NPC>(_U2_in, U2_in);

	//resize<XF_INTERPOLATION_BILINEAR, FLOW_T, ROWS, COLS, ROWS, COLS,NPC,  2>(U1_in, U1_resize);
	//resize<XF_INTERPOLATION_BILINEAR, FLOW_T, ROWS, COLS, ROWS, COLS,NPC,  2>(U2_in, U2_resize);
        resizeNNBilinear<FLOW_T, ROWS, COLS, NPC, ROWS, COLS, XF_INTERPOLATION_BILINEAR, 2>(U1_in, U1_resize);
        resizeNNBilinear<FLOW_T, ROWS, COLS, NPC, ROWS, COLS, XF_INTERPOLATION_BILINEAR, 2>(U2_in, U2_resize);

	multiplyU<FLOW_T, FLOW_FB, ROWS, COLS, NPC, SCALE_BW>(U1_resize, U2_resize, U1_out, U2_out, invscale, newheight, newwidth);

	xf::cv::xfMat2Array<FLOW_PTRWIDTH, FLOW_T, ROWS, COLS, NPC>(U1_out,  _U1_out);
	xf::cv::xfMat2Array<FLOW_PTRWIDTH, FLOW_T, ROWS, COLS, NPC>(U2_out,  _U2_out);

}

_MRGFUNCT_TPLT_DEC void merge_out(
		ap_uint<FLOW_PTRWIDTH>* _U1,
		ap_uint<FLOW_PTRWIDTH>* _U2,
		ap_uint<OUT_PTRWIDTH>* _flow,
		int height,
		int width)
{
	xf::cv::Mat<FLOW_T, ROWS, COLS, NPC> U1(height, width);
	xf::cv::Mat<FLOW_T, ROWS, COLS, NPC> U2(height, width);
	xf::cv::Mat<XF_64UC1, ROWS, COLS, NPC> U12(height, width);

	// clang-format off
#pragma HLS dataflow
	// clang-format on

	xf::cv::Array2xfMat<FLOW_PTRWIDTH, FLOW_T, ROWS, COLS, NPC>(_U1, U1);
	xf::cv::Array2xfMat<FLOW_PTRWIDTH, FLOW_T, ROWS, COLS, NPC>(_U2, U2);

	mergeU<FLOW_T, XF_64UC1, ROWS, COLS, NPC, FLOW_FBITS>(U1, U2, U12, height, width);

	xf::cv::xfMat2Array<OUT_PTRWIDTH, XF_64UC1, ROWS, COLS, NPC>(U12, _flow);
}

_TVL1FUNCT_TPLT_DEC void TVLonescale_n_resizeScaleU(
		ap_uint<IMG_PTRWIDTH> * _I0,
		ap_uint<IMG_PTRWIDTH> * _I1,
		ap_uint<FLOW_PTRWIDTH> * _U1,
		ap_uint<FLOW_PTRWIDTH> * _U2,
		ap_uint<FLOW_PTRWIDTH> * _I1wx,
		ap_uint<FLOW_PTRWIDTH> * _I1wy,
		ap_uint<FLOW_PTRWIDTH> * _grad,
		ap_uint<FLOW_PTRWIDTH> * _rhoc,
		ap_uint<FLOW_PTRWIDTH> * _p11,
		ap_uint<FLOW_PTRWIDTH> * _p12,
		ap_uint<FLOW_PTRWIDTH> * _p21,
		ap_uint<FLOW_PTRWIDTH> * _p22,
		ap_uint<FLOW_PTRWIDTH> * _U1_upscaled,
		ap_uint<FLOW_PTRWIDTH> * _U2_upscaled,
		ap_int<FLOW_BW> _lt,
		ap_int<FLOW_BW> _theta,
		ap_int<FLOW_BW> _gamma,
		ap_int<FLOW_BW> _taut,
		unsigned short _innerIterations,
		unsigned short _outerIterations ,
		unsigned short _warps,
		unsigned short height,
		unsigned short width ,
		unsigned short height_upscaled,
		unsigned short width_upscaled ,
		ap_int<INVSCALE_BW> invscale,
		int s_loopcnt, 
		ap_int<ERR_BW> tvl_threshold,
		int * _debug_error, int * error_count, bool _Error_debug_enable)
{
	// clang-format off
#pragma HLS inline off
	// clang-format on

	TVLonescale< IMG_PTRWIDTH, FLOW_PTRWIDTH, IMG_T, FLOW_T, IMG_FBITS, FLOW_FBITS, ROWS, COLS, NPC, NPCTVLINNERCORE, ERR_BW, MEDIANFILTERSIZE, INVSCALE_BW, MFV, FLOW_BW, MBINST>
	(_I0, _I1, _U1, _U2, _I1wx,_I1wy,_grad,_rhoc,_p11,_p12,_p21,_p22,_lt,_theta,_gamma,_taut,_innerIterations,_outerIterations,_warps,height,width, s_loopcnt, tvl_threshold, _debug_error, error_count,_Error_debug_enable );

	resizeScale_flow<FLOW_PTRWIDTH, FLOW_T, FLOW_FBITS, ROWS, COLS, NPC, INVSCALE_BW>
	(_U1, _U2, _U1_upscaled, _U2_upscaled, invscale, height, width, height_upscaled, width_upscaled);

}


