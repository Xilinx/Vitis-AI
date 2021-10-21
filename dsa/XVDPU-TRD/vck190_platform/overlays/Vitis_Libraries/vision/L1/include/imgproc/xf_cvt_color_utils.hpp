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

#ifndef _XF_CVT_COLOR_UTILS_HPP_
#define _XF_CVT_COLOR_UTILS_HPP_

#ifndef _XF_CVT_COLOR_HPP_
#error This file can not be included independently !
#endif

#include <string.h>
#include "ap_int.h"
#include "common/xf_types.hpp"
#include "common/xf_structs.hpp"
#include "common/xf_params.hpp"
#include "common/xf_utility.hpp"

/***************************************************************************
 * 	Parameters reqd. for RGB to YUV conversion
 *	   -- Q1.15 format
 *	   -- A2X  A's contribution in calculation of X
 **************************************************************************/
#define R2Y 8422  // 0.257
#define G2Y 16516 // 0.504
#define B2Y 3212  // 0.098
#define R2V 14382 // 0.4389
#define G2V 53477 //-0.368
#define B2V 63209 //-0.071
#define R2U 60686 //-0.148
#define G2U 56000 //-0.291
#define B2U 14386 // 0.439

/***************************************************************************
 * 	Parameters reqd. for YUV to RGB conversion
 *	   -- Q1.15 format
 *	   -- A2X  A's contribution in calculation of X
 *	   -- Only fractional part is taken for weigths greater
 *	      than 1 and interger part is added as offset
 **************************************************************************/
#define Y2R 5374  // 0.164
#define U2R 0     // 0
#define V2R 19530 // 0.596
#define Y2G 5374  // 0.164
#define U2G 52723 //-0.391
#define V2G 38895 //-0.813
#define Y2B 5374  // 0.164
#define U2B 590   // 0.018
#define V2B 0     // 0

#define F_05 16384 // 0.5 in Q1.15 format
/********************************************************************************
 * 	Parameters reqd. for RGB to GRAY conversion
 *	   -- Q1.15 format
 *	   -- A2X  A's contribution in calculation of X
 *	   -- Only fractional part is taken for weigths greater
 *	      than 1 and interger part is added as offset
 *
 *
 ********************************************************************************/
#define _CVT_WEIGHT1 9798  // 0.299
#define _CVT_WEIGHT2 19235 // 0.587
#define _CVT_WEIGHT3 3736  // 0.114
/********************************************************************************
 * 	Parameters reqd. for RGB to XYZ conversion
 *	   -- Q1.15 format
 *	   -- A2X  A's contribution in calculation of X
 *	   -- Only fractional part is taken for weigths greater
 *	      than 1 and interger part is added as offset
 *
 *
 ********************************************************************************/
#define _CVT_X1 13515 // 0.412453
#define _CVT_X2 11717 // 0.357580
#define _CVT_X3 5915  // 0.180523

#define _CVT_Y1 6969  // 0.212671
#define _CVT_Y2 23434 // 0.715160
#define _CVT_Y3 2364  // 0.072169

#define _CVT_Z1 636   // 0.019334
#define _CVT_Z2 3906  // 0.119193
#define _CVT_Z3 31137 // 0.950227
/********************************************************************************
 * 	Parameters reqd. for RGB to XYZ conversion
 *	   -- Q1.15 format
 *	   -- A2X  A's contribution in calculation of X
 *	   -- Only fractional part is taken for weigths greater
 *	      than 1 and interger part is added as offset
 *
 *
 ********************************************************************************/
#define _CVT_R1 26546 // 3.240479
#define _CVT_R2 52944 //-1.53715
#define _CVT_R3 61452 //-0.498535

#define _CVT_G1 57596 //-0.969256
#define _CVT_G2 15368 // 1.875991
#define _CVT_G3 340   // 0.041556

#define _CVT_B1 456   // 0.055648
#define _CVT_B2 63864 //-0.204043
#define _CVT_B3 8662  // 1.057311

#define CR_WEIGHT 23364 // 0.713
#define CB_WEIGHT 18481 // 0.564
/**************************************************************************
 * Pack Chroma values into a single variable
 *************************************************************************/
// PackPixels
template <int WORDWIDTH>
XF_SNAME(WORDWIDTH)
PackPixels(ap_uint8_t* buf) {
    XF_SNAME(WORDWIDTH) val;
    for (int k = 0, l = 0; k < XF_WORDDEPTH(WORDWIDTH); k += 8, l++) {
// clang-format off
        #pragma HLS unroll
        // clang-format on
        // Get bits from certain range of positions.
        val.range(k + 7, k) = buf[l];
    }
    return val;
}

/**************************************************************************
 * Extract UYVY Pixels from input stream
 *************************************************************************/
// ExtractUYVYPixels
template <int WORDWIDTH>
void ExtractUYVYPixels(XF_SNAME(WORDWIDTH) pix, ap_uint8_t* buf) {
    int k;
    XF_SNAME(WORDWIDTH) val;
    int pos = 0;
    val = (XF_SNAME(WORDWIDTH))pix;
    for (k = 0; k < (XF_WORDDEPTH(WORDWIDTH)); k += 8) {
// clang-format off
        #pragma HLS unroll
        // clang-format on
        uint8_t p;
        // Get bits from certain range of positions.
        p = val.range(k + 7, k);
        buf[pos++] = p;
    }
}

/****************************************************************************
 * Extract R, G, B, A values into a buffer
 ***************************************************************************/
// ExtractRGBAPixels
template <int WORDDEPTH>
void ExtractRGBAPixels(XF_SNAME(WORDDEPTH) pix, uint8_t* buf) {
    int k, pos = 0;
    uint8_t p;
    XF_SNAME(WORDDEPTH) val;
    val = (XF_SNAME(WORDDEPTH))pix;
    for (k = 0; k < XF_WORDDEPTH(WORDDEPTH); k += 8) {
// clang-format off
        #pragma HLS unroll
        // clang-format on
        // Get bits from certain range of positions.
        p = val.range(k + 7, k);
        buf[pos++] = p;
    }
}

/***********************************************************************
 * 	Pack R,G,B,A values into a single variable
 **********************************************************************/
// PackRGBAPixels
template <int WORDWIDTH>
XF_SNAME(WORDWIDTH)
PackRGBAPixels(ap_uint8_t* buf) {
    XF_SNAME(WORDWIDTH) val;
    for (int k = 0, l = 0; k < (XF_WORDDEPTH(WORDWIDTH)); k += 8, l++) {
// clang-format off
        #pragma HLS unroll
        // clang-format on
        // Get bits from certain range of positions.
        val.range(k + 7, k) = buf[l];
    }
    return val;
}

////auWriteChroma420
// template<int ROWS,int COLS,int NPC,int WORDWIDTH>
// void auWriteChroma420(auviz::Mat<ROWS, COLS, AU_8UP, NPC, WORDWIDTH>& plane,
//		XF_SNAME(WORDWIDTH) *dst, int off)
//{
//	bool flag = 0;
//	XF_SNAME(WORDWIDTH) ping[COLS>>NPC], pong[COLS>>NPC];
//	int nppc = AU_NPIXPERCYCLE(NPC);
//	int wordsize = plane.cols * nppc *(AU_PIXELDEPTH(AU_8UP)>>3);
//
//	int i, dst_off = off*(plane.cols);
//	auReadFromMat<ROWS,COLS,AU_8UP,NPC,WORDWIDTH,(COLS>>NPC)>(plane, ping);
//	WRUV420:
//	for( i = 0 ; i < (plane.rows-1); i++, dst_off += plane.cols)
//	{
//#pragma HLS LOOP_TRIPCOUNT min=ROWS max=ROWS
//		if(flag == 0)
//		{
//			auReadFromMat<ROWS,COLS,AU_8UP,NPC,WORDWIDTH,(COLS>>NPC)>(plane, pong);
//			auCopyMemoryOut<COLS, WORDWIDTH>(ping, dst, dst_off,  wordsize);
//			flag = 1;
//		}
//		else if(flag == 1)
//		{
//			auReadFromMat<ROWS,COLS,AU_8UP,NPC,WORDWIDTH,(COLS>>NPC)>(plane, ping);
//			auCopyMemoryOut<COLS, WORDWIDTH>(pong, dst, dst_off,  wordsize);
//			flag = 0;
//		}
//	}
//	if(flag == 1)
//		auCopyMemoryOut<COLS, WORDWIDTH>(pong, dst, dst_off, wordsize);
//	else
//		auCopyMemoryOut<COLS, WORDWIDTH>(ping, dst, dst_off, wordsize);
//}
//
// template <int WORDWIDTH>
// void auReadDummy(XF_SNAME(WORDWIDTH)* src)
//{
//	XF_SNAME(WORDWIDTH) dummy[1];
//	int pixelsize = AU_WORDDEPTH(WORDWIDTH)>>3;
//	memcpy((XF_SNAME(WORDWIDTH)*)dummy , (XF_SNAME(WORDWIDTH)*)src , pixelsize);
//}
//
// template <int WORDWIDTH>
// void auWriteDummy(XF_SNAME(WORDWIDTH)* ptr,XF_SNAME(WORDWIDTH) *dst)
//{
//	int pixelsize = AU_WORDDEPTH(WORDWIDTH)>>3;
//	memcpy((XF_SNAME(WORDWIDTH)*)dst , (XF_SNAME(WORDWIDTH)*)ptr , pixelsize);
//}
//
// template <int WORDWIDTH>
// void auWriteDummy1(XF_SNAME(WORDWIDTH)* ptr,XF_SNAME(WORDWIDTH) *dst)
//{
//	int pixelsize = AU_WORDDEPTH(WORDWIDTH)>>3;
//	memcpy((XF_SNAME(WORDWIDTH)*)dst , (XF_SNAME(WORDWIDTH)*)ptr , pixelsize);
//}
// auWriteUV420
// template<int WORDWIDTH_UV, int WORDWIDTH_DST, int NPC, int ROWS, int COLS>
// void auWriteUV420(auviz::Mat<ROWS, COLS, AU_8UP, NPC, WORDWIDTH_UV>& plane, XF_SNAME(WORDWIDTH_DST)* dst, int off)
//{
//
//	bool flag = 0;
//	XF_SNAME(WORDWIDTH_DST) ping[COLS>>NPC], pong[COLS>>NPC];
//	int nppc = AU_NPIXPERCYCLE(NPC);
//	int wordsize = plane.cols * nppc*(AU_PIXELDEPTH(AU_8UP)>>3);
//	int i;
//
//	int dst_off = off * plane.cols;
//
//	auPullFromMat<ROWS,COLS,AU_8UP,NPC,WORDWIDTH_UV,WORDWIDTH_DST,(COLS>>NPC)>(plane, ping);
//	WRUV420:
//	for( i = 0 ; i < (plane.rows-1); i++)
//	{
//#pragma HLS LOOP_TRIPCOUNT min=ROWS max=ROWS
//		if(flag == 0)
//		{
//			auPullFromMat<ROWS,COLS,AU_8UP,NPC,WORDWIDTH_UV,WORDWIDTH_DST,(COLS>>NPC)>(plane, pong);
//			auCopyMemoryOut<COLS, WORDWIDTH_DST>(ping, dst, dst_off,  wordsize);
//			flag = 1;
//		}
//		else if(flag == 1)
//		{
//			auPullFromMat<ROWS,COLS,AU_8UP,NPC,WORDWIDTH_UV,WORDWIDTH_DST,(COLS>>NPC)>(plane, ping);
//			auCopyMemoryOut<COLS, WORDWIDTH_DST>(pong, dst, dst_off,  wordsize);
//			flag = 0;
//		}
//		dst_off += plane.cols;
//	}
//	if(flag == 1)
//		auCopyMemoryOut<COLS, WORDWIDTH_DST>(pong, dst, dst_off,  wordsize);
//	else
//		auCopyMemoryOut<COLS, WORDWIDTH_DST>(ping, dst, dst_off,  wordsize);
//}
// auWriteYuv444
// template<int ROWS, int COLS, int NPC, int WORDWIDTH>
// void auWriteYuv444(
//		auviz::Mat<ROWS, COLS, AU_8UP, NPC, WORDWIDTH>& y_plane,
//		auviz::Mat<ROWS, COLS, AU_8UP, NPC, WORDWIDTH>& u_plane,
//		auviz::Mat<ROWS, COLS, AU_8UP, NPC, WORDWIDTH>& v_plane,
//		XF_SNAME(WORDWIDTH)* dst0,
//		XF_SNAME(WORDWIDTH)* dst1,
//		XF_SNAME(WORDWIDTH)* dst2)
//{
//	bool flag = 0;
//	XF_SNAME(WORDWIDTH) ping1[COLS>>NPC], pong1[COLS>>NPC];
//	XF_SNAME(WORDWIDTH) ping2[COLS>>NPC], pong2[COLS>>NPC];
//	XF_SNAME(WORDWIDTH) ping3[COLS>>NPC], pong3[COLS>>NPC];
//
//	int nppc = AU_NPIXPERCYCLE(NPC);
//	int wordsize = y_plane.cols * nppc *(AU_PIXELDEPTH(AU_8UP)>>3);
//	int i;
//
//	auReadFromMat<ROWS,COLS,AU_8UP,NPC,WORDWIDTH,(COLS>>NPC)>(y_plane, ping1);
//	auReadFromMat<ROWS,COLS,AU_8UP,NPC,WORDWIDTH,(COLS>>NPC)>(u_plane, ping2);
//	auReadFromMat<ROWS,COLS,AU_8UP,NPC,WORDWIDTH,(COLS>>NPC)>(v_plane, ping3);
//
//	WRUV420:
//	for( i = 0 ; i < (y_plane.rows-1); i++)
//	{
//#pragma HLS LOOP_TRIPCOUNT min=ROWS max=ROWS
//		if(flag == 0)
//		{
//			auReadFromMat<ROWS,COLS,AU_8UP,NPC,WORDWIDTH,(COLS>>NPC)>(y_plane, pong1);
//			auReadFromMat<ROWS,COLS,AU_8UP,NPC,WORDWIDTH,(COLS>>NPC)>(u_plane, pong2);
//			auReadFromMat<ROWS,COLS,AU_8UP,NPC,WORDWIDTH,(COLS>>NPC)>(v_plane, pong3);
//
//			auCopyMemoryOut<COLS, WORDWIDTH>(ping1, dst0, (i)*y_plane.cols,  wordsize);
//			auCopyMemoryOut<COLS, WORDWIDTH>(ping2, dst1, (i)*u_plane.cols,  wordsize);
//			auCopyMemoryOut<COLS, WORDWIDTH>(ping3, dst2, (i)*v_plane.cols,  wordsize);
//			//auCopyMemoryOut<COLS, WORDWIDTH>(ping2, dst, (i+y_plane.rows)*u_plane.cols,  wordsize);
//			//auCopyMemoryOut<COLS, WORDWIDTH>(ping3, dst, (i+(y_plane.rows<<1))*v_plane.cols,  wordsize);
//			flag = 1;
//		}
//		else if(flag == 1)
//		{
//			auReadFromMat<ROWS,COLS,AU_8UP,NPC,WORDWIDTH,(COLS>>NPC)>(y_plane, ping1);
//			auReadFromMat<ROWS,COLS,AU_8UP,NPC,WORDWIDTH,(COLS>>NPC)>(u_plane, ping2);
//			auReadFromMat<ROWS,COLS,AU_8UP,NPC,WORDWIDTH,(COLS>>NPC)>(v_plane, ping3);
//
//			auCopyMemoryOut<COLS, WORDWIDTH>(pong1, dst0, (i)*y_plane.cols,  wordsize);
//			auCopyMemoryOut<COLS, WORDWIDTH>(pong2, dst1, (i)*u_plane.cols,  wordsize);
//			auCopyMemoryOut<COLS, WORDWIDTH>(pong3, dst2, (i)*v_plane.cols,  wordsize);
//			//auCopyMemoryOut<COLS, WORDWIDTH>(pong2, dst, (i+y_plane.rows)*u_plane.cols,  wordsize);
//			//auCopyMemoryOut<COLS, WORDWIDTH>(pong3, dst, (i+(y_plane.rows<<1))*v_plane.cols,  wordsize);
//			flag = 0;
//		}
//	}
//	if(flag == 1)
//	{
//		auCopyMemoryOut<COLS, WORDWIDTH>(pong1, dst0, (i)*y_plane.cols, wordsize);
//		auCopyMemoryOut<COLS, WORDWIDTH>(pong2, dst1, (i)*u_plane.cols,  wordsize);
//		auCopyMemoryOut<COLS, WORDWIDTH>(pong3, dst2, (i)*v_plane.cols,  wordsize);
//		//auCopyMemoryOut<COLS, WORDWIDTH>(pong2, dst, (i+y_plane.rows)*u_plane.cols,  wordsize);
//		//auCopyMemoryOut<COLS, WORDWIDTH>(pong3, dst, (i+(y_plane.rows<<1))*v_plane.cols,  wordsize);
//	}
//	else
//	{
//		auCopyMemoryOut<COLS, WORDWIDTH>(ping1, dst0, (i)*y_plane.cols, wordsize);
//		auCopyMemoryOut<COLS, WORDWIDTH>(ping2, dst1, (i+y_plane.rows)*u_plane.cols, wordsize);
//		auCopyMemoryOut<COLS, WORDWIDTH>(ping3, dst2, (i+(y_plane.rows<<1))*v_plane.cols, wordsize);
//		//auCopyMemoryOut<COLS, WORDWIDTH>(ping2, dst, (i+y_plane.rows)*u_plane.cols, wordsize);
//		//auCopyMemoryOut<COLS, WORDWIDTH>(ping3, dst, (i+(y_plane.rows<<1))*v_plane.cols, wordsize);
//	}
//}
// auWriteRgba
// template<int ROWS, int COLS,int DEPTH,int NPC,int WORDWIDTH>
// void auWriteRgba(auviz::Mat<ROWS, COLS, AU_32UP, NPC, WORDWIDTH>& plane, XF_SNAME(WORDWIDTH)* dst0,
// XF_SNAME(WORDWIDTH)* dst1, XF_SNAME(WORDWIDTH)* dst2)
//{
//	XF_SNAME(WORDWIDTH) dummy0[1],dummy1[1];
//	dummy0[0] = 0;dummy1[0] = 0;
//
//	auWriteImage<ROWS, COLS, DEPTH, NPC, WORDWIDTH>(plane, dst0);
//	auWriteDummy<WORDWIDTH>(dummy0,dst1);
//	auWriteDummy1<WORDWIDTH>(dummy1,dst2);
//}
//
//
// template<int WORDWIDTH, int NPC, int ROWS, int COLS>
// void auWriteRgba_in(auviz::Mat<ROWS, COLS, AU_32UP, NPC, WORDWIDTH>& plane, XF_SNAME(WORDWIDTH)* dst0,
// XF_SNAME(WORDWIDTH)* dst1, XF_SNAME(WORDWIDTH)* dst2)
//{
//	bool flag = 0;
//	XF_SNAME(WORDWIDTH) ping[COLS>>NPC], pong[COLS>>NPC];
//	XF_SNAME(WORDWIDTH) dummy0[1],dummy1[1];
//	dummy0[0] = 0;dummy1[0] = 0;
//
//	int pixelsize = AU_WORDDEPTH(WORDWIDTH);
//	int nppc = AU_NPIXPERCYCLE(NPC);
//	int size = plane.cols * nppc *(AU_PIXELDEPTH(AU_32UP)>>3);
//	int i, offset = 0;
//
//	auReadFromMat<ROWS,COLS,AU_32UP,NPC,WORDWIDTH,(COLS>>NPC)>(plane, ping);
//	WR_Rgba:
//	for( i = 0 ; i < (plane.rows-1); i++)
//	{
//#pragma HLS LOOP_TRIPCOUNT min=ROWS max=ROWS
//		if(flag == 0)
//		{
//			auReadFromMat<ROWS,COLS,AU_32UP,NPC,WORDWIDTH,(COLS>>NPC)>(plane, pong);
//			auCopyMemoryOut<COLS*(AU_PIXELDEPTH(AU_32UP)>>3), WORDWIDTH>(ping, dst0, offset, size);
//			flag = 1;
//		}
//		else if(flag == 1)
//		{
//			auReadFromMat<ROWS,COLS,AU_32UP,NPC,WORDWIDTH,(COLS>>NPC)>(plane, ping);
//			auCopyMemoryOut<COLS*(AU_PIXELDEPTH(AU_32UP)>>3), WORDWIDTH>(pong, dst0, offset,  size);
//			flag = 0;
//		}
//		offset += plane.cols;
//	}
//	if(flag == 1)
//		auCopyMemoryOut<COLS*(AU_PIXELDEPTH(AU_32UP)>>3), WORDWIDTH>(pong, dst0, offset,  size);
//	else
//		auCopyMemoryOut<COLS*(AU_PIXELDEPTH(AU_32UP)>>3), WORDWIDTH>(ping, dst0, offset, size);
//
//	memcpy((XF_SNAME(WORDWIDTH)*)dst1, (XF_SNAME(WORDWIDTH)*)dummy0 , pixelsize);
//	memcpy((XF_SNAME(WORDWIDTH)*)dst2, (XF_SNAME(WORDWIDTH)*)dummy0 , pixelsize);
//}
//
//
////auWriteUV444
// template<int TC, int WORDWIDTH, int NPC, int ROWS, int COLS>
// void auWriteUV444(auviz::Mat<ROWS,COLS,AU_8UP,NPC,WORDWIDTH>& src, XF_SNAME(WORDWIDTH)* dst, int Offset)
//{
//	XF_SNAME(WORDWIDTH) ping[COLS>>NPC], pong[COLS>>NPC];
//	bool flag = 0;
//	int nppc = AU_NPIXPERCYCLE(NPC);
//	int wordsize = src.cols*nppc*(AU_PIXELDEPTH(AU_8UP)>>3);
//	int i;
//
//	auReadFromMat<ROWS,COLS,AU_8UP,NPC,WORDWIDTH,(COLS>>NPC)>(src, ping);
//	WR_UV444:
//	for(i = 0; i < ((src.rows+1)>>1)-1; i++)
//	{
//#pragma HLS LOOP_TRIPCOUNT min=TC max=TC
//		if(flag == 0)
//		{
//			auReadFromMat<ROWS,COLS,AU_8UP,NPC,WORDWIDTH,(COLS>>NPC)>(src, pong);
//			auCopyMemoryOut<COLS, WORDWIDTH>(ping, dst,  ((i<<1)+Offset)*src.cols,  wordsize);
//			auCopyMemoryOut<COLS, WORDWIDTH>(ping, dst,  (((i<<1)+1)+Offset)*src.cols,  wordsize);
//			flag = 1;
//		}
//		else if(flag == 1)
//		{
//			auReadFromMat<ROWS,COLS,AU_8UP,NPC,WORDWIDTH,(COLS>>NPC)>(src, ping);
//			auCopyMemoryOut<COLS, WORDWIDTH>(pong, dst,  ((i<<1)+Offset)*src.cols,  wordsize);
//			auCopyMemoryOut<COLS, WORDWIDTH>(pong, dst,  (((i<<1)+1)+Offset)*src.cols,  wordsize);
//			flag = 0;
//		}
//	}
//	if(flag == 1)
//	{
//		auCopyMemoryOut<COLS, WORDWIDTH>(pong, dst, ((i<<1)+Offset)*src.cols,  wordsize);
//		auCopyMemoryOut<COLS, WORDWIDTH>(pong, dst, (((i<<1)+1)+Offset)*src.cols,  wordsize);
//	}
//	else if(flag == 0)
//	{
//		auCopyMemoryOut<COLS, WORDWIDTH>(ping, dst, ((i<<1)+Offset)*src.cols,  wordsize);
//		auCopyMemoryOut<COLS, WORDWIDTH>(ping, dst, (((i<<1)+1)+Offset)*src.cols,  wordsize);
//	}
//}
//
//
////auReadUV420
// template<int WORDWIDTH_SRC, int WORDWIDTH_DST, int NPC, int ROWS, int COLS>
// void auReadUV420(XF_SNAME(WORDWIDTH_SRC)* src, auviz::Mat<ROWS, COLS, AU_8UP, NPC,WORDWIDTH_DST>& dst, int Offset)
//{
//	bool flag = 0;
//	XF_SNAME(WORDWIDTH_SRC) ping[COLS>>NPC], pong[COLS>>NPC];
//	int src_off = Offset*(dst.cols);
//	int nppc = AU_NPIXPERCYCLE(NPC);
//	int wordsize = dst.cols*nppc*(AU_PIXELDEPTH(AU_8UP)>>3);
//
//	auCopyMemoryIn<COLS, WORDWIDTH_SRC>(src, ping, src_off, wordsize);
//	RD_UV420:
//	for(int i = 1 ; i < (dst.rows); i++)
//	{
//#pragma HLS LOOP_TRIPCOUNT min=ROWS max=ROWS
//		src_off += dst.cols;
//		if(flag == 0)
//		{
//			auCopyMemoryIn<COLS, WORDWIDTH_SRC>(src, pong, src_off, wordsize);
//			auPushIntoMat<ROWS,COLS,AU_8UP,NPC,WORDWIDTH_SRC,WORDWIDTH_DST,(COLS>>NPC)>(ping,dst,0);
//			flag = 1;
//		}
//		else if(flag == 1)
//		{
//			auCopyMemoryIn<COLS, WORDWIDTH_SRC>(src, ping, src_off, wordsize);
//			auPushIntoMat<ROWS,COLS,AU_8UP,NPC,WORDWIDTH_SRC,WORDWIDTH_DST,(COLS>>NPC)>(pong,dst,0);
//			flag = 0;
//		}
//	}
//	if(flag == 1)
//		auPushIntoMat<ROWS,COLS,AU_8UP,NPC,WORDWIDTH_SRC,WORDWIDTH_DST,(COLS>>NPC)>(pong,dst,0);
//	else if(flag == 0)
//		auPushIntoMat<ROWS,COLS,AU_8UP,NPC,WORDWIDTH_SRC,WORDWIDTH_DST,(COLS>>NPC)>(ping,dst,0);
//}

// template<int WORDWIDTH, int NPC, int ROWS, int COLS>
// void auReadRgb_plane(XF_SNAME(WORDWIDTH)* src, auviz::Mat<ROWS, COLS, AU_32UP, NPC, WORDWIDTH>& dst)
//{
//	bool flag = 0;
//	XF_SNAME(WORDWIDTH) ping[COLS>>NPC], pong[COLS>>NPC];
//	int nppc = AU_NPIXPERCYCLE(NPC);
//	int size = dst.cols* nppc*(AU_PIXELDEPTH(AU_32UP)>>3);
//	int src_off = 0;
//
//	auCopyMemoryIn<COLS*(AU_PIXELDEPTH(AU_32UP)>>3), WORDWIDTH>(src, ping, 0, size);
//	for(int i = 1 ; i < (dst.rows); i++)
//	{
//#pragma HLS LOOP_TRIPCOUNT min=ROWS max=ROWS
//		src_off += dst.cols;
//		if(flag == 0)
//		{
//			auCopyMemoryIn<COLS*(AU_PIXELDEPTH(AU_32UP)>>3), WORDWIDTH>(src, pong, src_off, size);
//			auWriteIntoMat<ROWS,COLS,AU_32UP,NPC,WORDWIDTH,(COLS>>NPC)>(ping, dst);
//			flag = 1;
//		}
//		else if(flag == 1)
//		{
//			auCopyMemoryIn<COLS*(AU_PIXELDEPTH(AU_32UP)>>3), WORDWIDTH>(src, ping, src_off, size);
//			auWriteIntoMat<ROWS,COLS,AU_32UP,NPC,WORDWIDTH,(COLS>>NPC)>(pong, dst);
//			flag = 0;
//		}
//	}
//	if(flag == 1)
//		auWriteIntoMat<ROWS,COLS,AU_32UP,NPC,WORDWIDTH,(COLS>>NPC)>(pong, dst);
//	else if(flag == 0)
//		auWriteIntoMat<ROWS,COLS,AU_32UP,NPC,WORDWIDTH,(COLS>>NPC)>(ping, dst);
//}

// auReadRgb
// template<int WORDWIDTH, int NPC, int ROWS, int COLS>
// void auReadRgb(
//		XF_SNAME(WORDWIDTH)* src0,
//		XF_SNAME(WORDWIDTH)* src1,
//		XF_SNAME(WORDWIDTH)* src2,
//		auviz::Mat<ROWS, COLS, AU_32UP, NPC, WORDWIDTH>& rgba
//		)
//{
//	auReadImage<ROWS,COLS,AU_32UP,NPC,WORDWIDTH>(src0, rgba);
//	auReadDummy<WORDWIDTH>(src1);
//	auReadDummy<WORDWIDTH>(src2);
//}
//
//
// template<int WORDWIDTH_SRC, int WORDWIDTH_DST, int NPC, int ROWS, int COLS>
// void auReadUyvy_plane(XF_SNAME(WORDWIDTH_SRC)* src, auviz::Mat<ROWS, COLS, AU_8UP, NPC, WORDWIDTH_DST>& dst)
//{
//	bool flag = 0;
//	XF_SNAME(WORDWIDTH_SRC) ping[COLS>>(NPC+1)], pong[COLS>>(NPC+1)];
//	int nppc = AU_NPIXPERCYCLE(NPC);
//	int wordsize = (dst.cols)*nppc*(AU_PIXELDEPTH(AU_8UP)>>3);
//	int offset = (dst.cols>>1);
//
//	auCopyMemoryIn<COLS, WORDWIDTH_SRC>(src, ping, 0, wordsize);
//	for(int i = 1 ; i < (dst.rows); i++)
//	{
//#pragma HLS LOOP TRIPCOUNT min=ROWS max=ROWS
//
//		if(flag == 0)
//		{
//			auCopyMemoryIn<COLS, WORDWIDTH_SRC>(src, pong, offset, wordsize);
//			auPushIntoMat<ROWS,COLS,AU_8UP,NPC,WORDWIDTH_SRC,WORDWIDTH_DST,((COLS>>NPC)>>1)>(ping, dst, 1);
//			flag = 1;
//		}
//		else if(flag == 1)
//		{
//			auCopyMemoryIn<COLS, WORDWIDTH_SRC>(src, ping, offset, wordsize);
//			auPushIntoMat<ROWS,COLS,AU_8UP,NPC,WORDWIDTH_SRC,WORDWIDTH_DST,((COLS>>NPC)>>1)>(pong, dst, 1);
//			flag = 0;
//		}
//		offset += (dst.cols>>1);
//	}
//	if(flag == 1)
//		auPushIntoMat<ROWS,COLS,AU_8UP,NPC,WORDWIDTH_SRC,WORDWIDTH_DST,((COLS>>NPC)>>1)>(pong, dst, 1);
//	else if(flag == 0)
//		auPushIntoMat<ROWS,COLS,AU_8UP,NPC,WORDWIDTH_SRC,WORDWIDTH_DST,((COLS>>NPC)>>1)>(ping, dst, 1);
//}

// auReadUyvy
// template<int WORDWIDTH_SRC, int WORDWIDTH_DST, int NPC, int ROWS, int COLS>
// void auReadUyvy(
//		XF_SNAME(WORDWIDTH_SRC)* src0,XF_SNAME(WORDWIDTH_SRC)* src1,XF_SNAME(WORDWIDTH_SRC)* src2,
//		 auviz::Mat<ROWS, COLS, AU_8UP, NPC, WORDWIDTH_DST>& in_uyvy)
//{
//	auReadUyvy_plane<WORDWIDTH_SRC>(src0, in_uyvy);
//	auReadDummy<WORDWIDTH_SRC>(src1);
//	auReadDummy<WORDWIDTH_SRC>(src2);
//}
/*
 template<int WORDWIDTH_SRC, int WORDWIDTH_DST, int NPC, int ROWS, int COLS>
 void auReadUyvy(XF_SNAME(WORDWIDTH_SRC)* src, auviz::Mat<ROWS, COLS, AU_8UP, NPC, WORDWIDTH_DST>& dst)
 {




 bool flag = 0;
 XF_SNAME(WORDWIDTH_SRC) ping[COLS>>(NPC+1)], pong[COLS>>(NPC+1)];
 int nppc = AU_NPIXPERCYCLE(NPC);
 int wordsize = (dst.cols)*nppc*(AU_PIXELDEPTH(AU_8UP)>>3);
 int offset = (dst.cols>>1);

 auCopyMemoryIn<COLS, WORDWIDTH_SRC>(src, ping, 0, wordsize);
 for(int i = 1 ; i < (dst.rows); i++)
 {
// clang-format off
 #pragma HLS LOOP TRIPCOUNT min=ROWS max=ROWS
// clang-format on

 if(flag == 0)
 {
 auCopyMemoryIn<COLS, WORDWIDTH_SRC>(src, pong, offset, wordsize);
 auPushIntoMat<ROWS,COLS,AU_8UP,NPC,WORDWIDTH_SRC,WORDWIDTH_DST,((COLS>>NPC)>>1)>(ping, dst, 1);
 flag = 1;
 }
 else if(flag == 1)
 {
 auCopyMemoryIn<COLS, WORDWIDTH_SRC>(src, ping, offset, wordsize);
 auPushIntoMat<ROWS,COLS,AU_8UP,NPC,WORDWIDTH_SRC,WORDWIDTH_DST,((COLS>>NPC)>>1)>(pong, dst, 1);
 flag = 0;
 }
 offset += (dst.cols>>1);
 }
 if(flag == 1)
 auPushIntoMat<ROWS,COLS,AU_8UP,NPC,WORDWIDTH_SRC,WORDWIDTH_DST,((COLS>>NPC)>>1)>(pong, dst, 1);
 else if(flag == 0)
 auPushIntoMat<ROWS,COLS,AU_8UP,NPC,WORDWIDTH_SRC,WORDWIDTH_DST,((COLS>>NPC)>>1)>(ping, dst, 1);
 }
 */
// auReadChroma420
// template<int WORDWIDTH, int NPC, int ROWS, int COLS>
// void auReadChroma420(
//		XF_SNAME(WORDWIDTH)* src,
//		auviz::Mat<ROWS, COLS, AU_8UP, NPC, WORDWIDTH>& dst,
//		int Offset)
//{
//	bool flag = 0;
//	XF_SNAME(WORDWIDTH) ping[COLS>>NPC], pong[COLS>>NPC];
//	int nppc = AU_NPIXPERCYCLE(NPC);
//	int wordsize = dst.cols*nppc*(AU_PIXELDEPTH(AU_8UP)>>3);
//	int src_off = Offset*(dst.cols);
//
//	auCopyMemoryIn<COLS, WORDWIDTH>(src, ping, src_off, wordsize);
//	for(int i = 1 ; i < (dst.rows); i++)
//	{
//#pragma HLS LOOP_TRIPCOUNT min=ROWS max=ROWS
//		src_off += dst.cols;
//		if(flag == 0)
//		{
//			auCopyMemoryIn<COLS, WORDWIDTH>(src, pong, src_off, wordsize);
//			auWriteIntoMat<ROWS,COLS,AU_8UP,NPC,WORDWIDTH,(COLS>>NPC)>(ping, dst);
//			flag = 1;
//		}
//		else if(flag == 1)
//		{
//			auCopyMemoryIn<COLS, WORDWIDTH>(src, ping, src_off, wordsize);
//			auWriteIntoMat<ROWS,COLS,AU_8UP,NPC,WORDWIDTH,(COLS>>NPC)>(pong, dst);
//			flag = 0;
//		}
//
//	}
//	if(flag == 1)
//		auWriteIntoMat<ROWS,COLS,AU_8UP,NPC,WORDWIDTH,(COLS>>NPC)>(pong, dst);
//	else if(flag == 0)
//		auWriteIntoMat<ROWS,COLS,AU_8UP,NPC,WORDWIDTH,(COLS>>NPC)>(ping, dst);
//}

// auWriteIyuv
// template<int ROWS,int COLS,int NPC,int WORDWIDTH>
// void auWriteIyuv(
//		auviz::Mat<ROWS, COLS, AU_8UP, NPC, WORDWIDTH>& y_plane,
//		auviz::Mat<(ROWS>>2), COLS, AU_8UP, NPC, WORDWIDTH>& u_plane,
//		auviz::Mat<(ROWS>>2), COLS, AU_8UP, NPC, WORDWIDTH>& v_plane,
//		XF_SNAME(WORDWIDTH) *dst0,XF_SNAME(WORDWIDTH) *dst1,XF_SNAME(WORDWIDTH) *dst2)
//{
//	auWriteImage<ROWS, COLS, AU_8UP, NPC, WORDWIDTH>(y_plane, dst0);
//	auWriteChroma420<(ROWS>>2),COLS,NPC,WORDWIDTH>(u_plane, dst1, 0);
//	auWriteChroma420<(ROWS>>2),COLS,NPC,WORDWIDTH>(v_plane, dst2, 0);
//}
// auWriteNV12
// template<int WORDWIDTH_Y, int WORDWIDTH_UV, int NPC, int ROWS, int COLS>
// void auWriteNV12(
//		auviz::Mat<ROWS, COLS, AU_8UP, NPC, WORDWIDTH_Y>& y_plane,
//		auviz::Mat<((ROWS+1)>>1), COLS, AU_8UP, NPC, WORDWIDTH_UV>& uv_plane,
//		XF_SNAME(WORDWIDTH_Y)* dst0,XF_SNAME(WORDWIDTH_Y)* dst1,XF_SNAME(WORDWIDTH_Y)* dst2)
//{
//	XF_SNAME(WORDWIDTH_Y) dummy[1];
//	dummy[0] = 0;
//	auWriteImage<ROWS, COLS, AU_8UP, NPC, WORDWIDTH_Y>(y_plane, dst0);
//	auWriteUV420<WORDWIDTH_UV, WORDWIDTH_Y, NPC>(uv_plane, dst1, 0);
//	auWriteDummy<WORDWIDTH_Y>(dummy,dst2);
//}

// auReadIyuv
// template<int WORDWIDTH, int NPC, int ROWS, int COLS>
// void auReadIyuv(
//		XF_SNAME(WORDWIDTH)* src0,
//		XF_SNAME(WORDWIDTH)* src1,
//		XF_SNAME(WORDWIDTH)* src2,
//		auviz::Mat<ROWS, COLS, AU_8UP, NPC, WORDWIDTH>& y_plane,
//		auviz::Mat<(ROWS>>2), COLS, AU_8UP, NPC, WORDWIDTH>& u_plane,
//		auviz::Mat<(ROWS>>2), COLS, AU_8UP, NPC, WORDWIDTH>& v_plane)
//{
//	int off = y_plane.rows & 0x3 ? (y_plane.rows>>2)+1 : (y_plane.rows>>2);
//	auReadImage(src0, y_plane);
//	auReadChroma420<WORDWIDTH, NPC>(src1, u_plane, 0);
//	auReadChroma420<WORDWIDTH, NPC>(src2, v_plane, 0);
//}

// auReadNV12
// template<int WORDWIDTH_Y, int WORDWIDTH_UV, int NPC, int ROWS, int COLS>
// void auReadNV12(
//		XF_SNAME(WORDWIDTH_Y)* src0,XF_SNAME(WORDWIDTH_Y)* src1,XF_SNAME(WORDWIDTH_Y)* src2,
//		auviz::Mat<ROWS, COLS, AU_8UP, NPC, WORDWIDTH_Y>& in_y,
//		auviz::Mat<((ROWS+1)>>1), COLS, AU_8UP, NPC, WORDWIDTH_UV>& in_uv)
//{
//	auReadImage(src0, in_y);
//	auReadUV420<WORDWIDTH_Y,WORDWIDTH_UV,NPC,((ROWS+1)>>1),COLS>(src1, in_uv,0);
//	auReadDummy<WORDWIDTH_Y>(src2);
//}

// auWriteYuv4
// template<int WORDWIDTH, int NPC, int ROWS, int COLS>
// void auWriteYuv4(
//		auviz::Mat<ROWS, COLS, AU_8UP, NPC, WORDWIDTH>& y_plane,
//		auviz::Mat<ROWS, COLS, AU_8UP, NPC, WORDWIDTH>& u_plane,
//		auviz::Mat<ROWS, COLS, AU_8UP, NPC, WORDWIDTH>& v_plane,
//		XF_SNAME(WORDWIDTH)* dst0,XF_SNAME(WORDWIDTH)* dst1,XF_SNAME(WORDWIDTH)* dst2)
//{
//	auWriteImage(y_plane, dst0);
//	auWriteUV444<((ROWS+1)>>1)-1>(u_plane, dst1, 0);
//	auWriteUV444<((ROWS+1)>>1)-1>(v_plane, dst2, 0);
//}

/****************************************************************************
 * 	Function to add the offset and check the saturation
 ***************************************************************************/
static uint8_t saturate_cast(int32_t Value, int32_t offset) {
    // Right shifting Value 15 times to get the integer part
    int Value_int = (Value >> 15) + offset;
    unsigned char Value_uchar = 0;
    if (Value_int > 255)
        Value_uchar = 255;
    else if (Value_int < 0)
        Value_uchar = 0;
    else
        Value_uchar = (uint8_t)Value_int;

    return Value_uchar;
}
static uint8_t saturate_cast(int32_t Value, int32_t offset, int fbits) {
    // Right shifting Value 15 times to get the integer part
    int Value_int = (Value >> fbits) + offset;
    unsigned char Value_uchar = 0;
    if (Value_int > 255)
        Value_uchar = 255;
    else if (Value_int < 0)
        Value_uchar = 0;
    else
        Value_uchar = (uint8_t)Value_int;

    return Value_uchar;
}
/****************************************************************************
 * 	CalculateY - calculates the Y(luma) component using R,G,B values
 * 	Y = (0.257 * R) + (0.504 * G) + (0.098 * B) + 16
 * 	An offset of 16 is added to the resultant value
 ***************************************************************************/
static uint8_t CalculateY(uint8_t R, uint8_t G, uint8_t B) {
// clang-format off
    #pragma HLS INLINE
    // clang-format on
    // 1.15 * 8.0 = 9.15
    int32_t Y = ((short int)R2Y * R) + ((short int)G2Y * G) + ((short int)B2Y * B) + F_05;
    uint8_t Yvalue = saturate_cast(Y, 16);
    return Yvalue;
}

/***********************************************************************
 * CalculateU - calculates the U(Chroma) component using R,G,B values
 * U = -(0.148 * R) - (0.291 * G) + (0.439 * B) + 128
 * an offset of 128 is added to the resultant value
 **********************************************************************/
static uint8_t CalculateU(uint8_t R, uint8_t G, uint8_t B) {
// clang-format off
    #pragma HLS INLINE
    // clang-format on
    int32_t U = ((short int)R2U * R) + ((short int)G2U * G) + ((short int)B2U * B) + F_05;
    uint8_t Uvalue = saturate_cast(U, 128);
    return Uvalue;
}

/***********************************************************************
 * CalculateV - calculates the V(Chroma) component using R,G,B values
 * V = (0.439 * R) - (0.368 * G) - (0.071 * B) + 128
 * an offset of 128 is added to the resultant value
 **********************************************************************/
static uint8_t CalculateV(uint8_t R, uint8_t G, uint8_t B) {
// clang-format off
    #pragma HLS INLINE
    // clang-format on
    int32_t V = ((short int)R2V * R) + ((short int)G2V * G) + ((short int)B2V * B) + F_05;
    uint8_t Vvalue = saturate_cast(V, 128);
    return Vvalue;
}

/***********************************************************************
 * CalculateR - calculates the R(Red) component using Y & V values
 * R = 1.164*Y + 1.596*V = 0.164*Y + 0.596*V + Y + V
 **********************************************************************/
static uint8_t CalculateR(uint8_t Y, int32_t V2Rtemp, int8_t V) {
// clang-format off
    #pragma HLS INLINE
    // clang-format on
    int32_t R = (short int)Y2R * Y + V2Rtemp + F_05;
    uint8_t Rvalue = saturate_cast(R, V + Y);
    return (Rvalue);
}

/***********************************************************************
 * CalculateG - calculates the G(Green) component using Y, U & V values
 * G = 1.164*Y - 0.813*V - 0.391*U = 0.164*Y - 0.813*V - 0.391*U + Y
 **********************************************************************/
static uint8_t CalculateG(uint8_t Y, int32_t U2Gtemp, int32_t V2Gtemp) {
// clang-format off
    #pragma HLS INLINE
    // clang-format on
    int32_t G = (short int)Y2G * Y + U2Gtemp + V2Gtemp + F_05;
    uint8_t Gvalue = saturate_cast(G, Y);
    return (Gvalue);
}

/***********************************************************************
 * CalculateB - calculates the B(Blue) component using Y & U values
 * B = 1.164*Y + 2.018*U = 0.164*Y + Y + 0.018*U + 2*U
 **********************************************************************/
static uint8_t CalculateB(uint8_t Y, int32_t U2Btemp, int8_t U) {
// clang-format off
    #pragma HLS INLINE
    // clang-format on
    int32_t B = (short int)Y2B * Y + U2Btemp + F_05;
    uint8_t Bvalue = saturate_cast(B, 2 * U + Y);
    return (Bvalue);
}

/***********************************************************************
 * CalculateGRAY - calculates the GRAY pixel value using R, G  & B values
 * GRAY = 0.299*R + 0.587*G + 0.114*B
 **********************************************************************/
static uint8_t CalculateGRAY(uint8_t R, uint8_t G, uint8_t B) {
// clang-format off
    #pragma HLS INLINE
    // clang-format on
    int32_t GRAY = (R * (short int)_CVT_WEIGHT1) + (G * (short int)_CVT_WEIGHT2) + (B * (short int)_CVT_WEIGHT3);
    uint8_t sat_GRAY = saturate_cast(GRAY, 0);

    return (sat_GRAY);
}

/***********************************************************************
 * CalculateGRAY - calculates the XYZ pixel value using R, G  & B values
 *	x	0.412453	0.357580	0.180423	R
 *
 *	y	0.212671	0.715160	0.072169	G
 *
 *	z	0.019334	0.119193	0.950257	B
 **********************************************************************/
static uint8_t Calculate_X(uint8_t R, uint8_t G, uint8_t B) {
// clang-format off
    #pragma HLS INLINE
    // clang-format on
    int32_t X = (R * (short int)_CVT_X1) + (G * (short int)_CVT_X2) + (B * (short int)_CVT_X3);
    uint8_t sat_X = saturate_cast(X, 0);

    return (sat_X);
}
static uint8_t Calculate_Y(uint8_t R, uint8_t G, uint8_t B) {
// clang-format off
    #pragma HLS INLINE
    // clang-format on
    int32_t Y = (R * (short int)_CVT_Y1) + (G * (short int)_CVT_Y2) + (B * (short int)_CVT_Y3);
    uint8_t sat_Y = saturate_cast(Y, 0);

    return (sat_Y);
}
static uint8_t Calculate_Z(uint8_t R, uint8_t G, uint8_t B) {
// clang-format off
    #pragma HLS INLINE
    // clang-format on
    int32_t Z = (R * (short int)_CVT_Z1) + (G * (short int)_CVT_Z2) + (B * (short int)_CVT_Z3);
    uint8_t sat_Z = saturate_cast(Z, 0);

    return (sat_Z);
}
/***********************************************************************
 * CalculateRGB - calculates the RGB pixel value using X, Y, Z values
 *	R	3.240479	-1.53715	-0.498535	X
 *
 *	G	-0.969256	1.875991	0.041556	Y
 *
 *	B	0.055648	-0.204043	1.057311	Z
 **********************************************************************/
static uint8_t Calculate_R(uint8_t X, uint8_t Y, uint8_t Z) {
// clang-format off
    #pragma HLS INLINE
    // clang-format on
    int32_t R = (X * (short int)_CVT_R1) + (Y * (short int)_CVT_R2) + (Z * (short int)_CVT_R3);

    uint8_t sat_R = saturate_cast(R, 0, 13);

    return (sat_R);
}
static uint8_t Calculate_G(uint8_t X, uint8_t Y, uint8_t Z) {
// clang-format off
    #pragma HLS INLINE
    // clang-format on
    int32_t G = (X * (short int)_CVT_G1) + (Y * (short int)_CVT_G2) + (Z * (short int)_CVT_G3);
    uint8_t sat_G = saturate_cast(G, 0, 13);

    return (sat_G);
}
static uint8_t Calculate_B(uint8_t X, uint8_t Y, uint8_t Z) {
// clang-format off
    #pragma HLS INLINE
    // clang-format on
    int32_t B = (X * (short int)_CVT_B1) + (Y * (short int)_CVT_B2) + (Z * (short int)_CVT_B3);
    uint8_t sat_B = saturate_cast(B, 0, 13);

    return (sat_B);
}
/***********************************************************************
 * calculates the CRCB pixel value using R,G,B,Y values
 *
 *	Cr <---	(R-Y)*0.713+delta
 *	Cb <---	(R-Y)*0.564+delta
 *
 *
 **********************************************************************/
static uint8_t Calculate_CR(uint8_t R, uint8_t Y) {
// clang-format off
    #pragma HLS INLINE
    // clang-format on
    int32_t CR = ((R - Y) * (short int)CR_WEIGHT);

    uint8_t sat_CR = saturate_cast(CR, 128);

    return (sat_CR);
}
static uint8_t Calculate_CB(uint8_t B, uint8_t Y) {
// clang-format off
    #pragma HLS INLINE
    // clang-format on
    int32_t CB = ((B - Y) * (short int)CB_WEIGHT);

    uint8_t sat_CB = saturate_cast(CB, 128);

    return (sat_CB);
}
/***********************************************************************
 *  calculates the R,G,B pixels value using Cr,Cb,Y values
 *
 *	R <---	Y+1.403*(Cr-delta)
 *	G <---	Y-0.714*Cr-delta-0.334*Cb-delta
 *	B <---	Y+1.773*(Cb-delta)
 *
 **********************************************************************/
#define Ycrcb2R 45974 // 1.403
#define Ycrcb2G 23396 // 0.714
#define W1 11272      // 0.344
#define Ycrcb2B 58098 // 1.773

static uint8_t Calculate_Ycrcb2R(uint8_t Y, uint8_t cr) {
// clang-format off
    #pragma HLS INLINE
    // clang-format on
    int32_t R = Ycrcb2R * (cr - 128);
    uint8_t sat_R = saturate_cast(R, Y);
    return (sat_R);
}
static uint8_t Calculate_Ycrcb2G(uint8_t Y, uint8_t cr, uint8_t cb) {
// clang-format off
    #pragma HLS INLINE
    // clang-format on

    int32_t H_G1 = (Ycrcb2G * (cr - 128));
    int32_t H_G2 = (W1 * (cb - 128));

    int16_t sat_G1 = ((H_G1) >> 15);
    int16_t sat_G2 = ((H_G2) >> 15);

    uint16_t res = ((Y - sat_G1) - sat_G2);
    if (res > 255) {
        res = 255;
    } else if (res < 0) {
        res = 0;
    }
    return (res);
}
static uint8_t Calculate_Ycrcb2B(uint8_t Y, uint8_t cb) {
// clang-format off
    #pragma HLS INLINE
    // clang-format on
    int32_t B = Ycrcb2B * (cb - 128);
    uint8_t sat_R = saturate_cast(B, Y);

    return (sat_R);
}

// static uint8_t min(uint8_t R, uint8_t G,uint8_t B) {
//#pragma HLS INLINE
//	int min=0;
//
//
//	return (min);
//}
#endif // _XF_CVT_COLOR_UTILS_H_
