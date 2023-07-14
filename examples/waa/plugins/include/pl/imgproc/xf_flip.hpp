/*
 * Copyright 2021 Xilinx, Inc.
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

#ifndef _XF_FLIP_HPP_
#define _XF_FLIP_HPP_

#include "hls_stream.h"
#include "common/xf_common.hpp"
#include <iostream>

namespace xf {
namespace cv {

// --------------------------------------------------------------------------------------------
// Function to reverse pixel within a pack of pixels
// --------------------------------------------------------------------------------------------
template <int TYPE, int NPC>
XF_TNAME(TYPE, NPC)
reversePixels(XF_TNAME(TYPE, NPC) InPackPixels) {
// clang-format off
		#pragma HLS INLINE
    // clang-format on

    XF_TNAME(TYPE, NPC) OutPackPixels;
    const int XF_PWIDTH = XF_PIXELWIDTH(TYPE, NPC);

    for (int k = 0; k < NPC; k++) {
// clang-format off
			#pragma HLS UNROLL
        // clang-format on
        OutPackPixels.range((NPC - k) * XF_PWIDTH - 1, (NPC - k - 1) * XF_PWIDTH) =
            InPackPixels.range((k + 1) * XF_PWIDTH - 1, k * XF_PWIDTH);
    }

    return OutPackPixels;
} // reversePixels

template <int PTR_WIDTH, int TYPE, int COLS, int NPC, int TC>
void flip_process(ap_uint<PTR_WIDTH> Src_RowPtr[TC], ap_uint<PTR_WIDTH> Dst_RowPtr[TC], uint16_t Width, int Direction) {
// clang-format off
    #pragma HLS INLINE OFF
// clang-format on		
				  
		int rd_ptr=0;
		XF_TNAME(TYPE, NPC) pxl_pack;
		
		for (int i = 0; i < Width; i++) {
			
	// clang-format off
			#pragma HLS LOOP_TRIPCOUNT min=TC max=TC
			#pragma HLS PIPELINE II=1
        // clang-format on

        if (Direction == 0) // Case of Veritical Flip only
            pxl_pack = Src_RowPtr[i];
        else // Case of Horizontal or both
            pxl_pack = reversePixels<TYPE, NPC>(Src_RowPtr[Width - 1 - i]);

        Dst_RowPtr[i] = pxl_pack;
    }

    return;
}

template <int PTR_WIDTH, int TYPE, int NPC_COLS, int ROWS, int COLS, int NPC>
void _Axi2Mat(ap_uint<PTR_WIDTH>* SrcPtr,
              ap_uint<XF_WORDDEPTH(XF_WORDWIDTH(TYPE, NPC))> SrcRow[NPC_COLS],
              uint16_t Rows,
              uint16_t Cols,
              int Direction,
              int r) {
// clang-format off
		#pragma HLS INLINE OFF
	// clang-format on				

		uint16_t NPCCols = Cols >> XF_BITSHIFT(NPC);
		const uint16_t Pixel_width = XF_WORDDEPTH(XF_WORDWIDTH(TYPE, NPC));
		
		uint32_t ColsPtrwidth = (((NPCCols * Pixel_width) + PTR_WIDTH - 1) / PTR_WIDTH);
	
		uint64_t OffsetSrc;
		
		if(Direction == 1)
			OffsetSrc = r*ColsPtrwidth;
		else
			OffsetSrc = (Rows-1)*ColsPtrwidth - r*ColsPtrwidth;
		
		MMIterIn<PTR_WIDTH, TYPE, 1, COLS, NPC, -1>::Array2xfMat(SrcPtr+OffsetSrc, SrcRow, 1, Cols, -1);
			
			
		return;
}

// Function to process the row

template <int TYPE, int NPC_COLS, int ROWS, int COLS, int NPC>
void _FlipRow(ap_uint<XF_WORDDEPTH(XF_WORDWIDTH(TYPE, NPC))> SrcRow[NPC_COLS],
					ap_uint<XF_WORDDEPTH(XF_WORDWIDTH(TYPE, NPC))> DstRow[NPC_COLS],
					uint16_t Rows,
					uint16_t NPCCols,
					int Direction){
						
// clang-format off
	#pragma HLS INLINE OFF
// clang-format on				
		
	flip_process<XF_WORDDEPTH(XF_WORDWIDTH(TYPE, NPC)), TYPE, COLS, NPC, NPC_COLS>(SrcRow, DstRow, NPCCols, Direction);
		
	return;
}

template <int PTR_WIDTH, int TYPE, int NPC_COLS, int ROWS, int COLS, int NPC>
void _Mat2Axi(ap_uint<XF_WORDDEPTH(XF_WORDWIDTH(TYPE, NPC))> DstRow[NPC_COLS],
					ap_uint<PTR_WIDTH>*DstPtr,
					uint16_t Rows,
					uint16_t Cols,
					int Direction,
					int r){
						
// clang-format off
	#pragma HLS INLINE OFF
// clang-format on				
	
	uint16_t NPCCols = Cols >> XF_BITSHIFT(NPC);
	const uint16_t Pixel_width = XF_WORDDEPTH(XF_WORDWIDTH(TYPE, NPC));
	
	uint32_t ColsPtrwidth = (((NPCCols * Pixel_width) + PTR_WIDTH - 1) / PTR_WIDTH);

	uint64_t OffsetDst;
	
	OffsetDst = r*ColsPtrwidth;
	MMIterOut<PTR_WIDTH, TYPE, 1, COLS, NPC, 1, -1>::xfMat2Array(DstRow, DstPtr+OffsetDst, 1, Cols, -1);
		
	return;
}

template <int PTR_WIDTH, int TYPE, int ROWS, int COLS, int NPC>
void flip(ap_uint<PTR_WIDTH>* SrcPtr,
                      ap_uint<PTR_WIDTH>* DstPtr,
                      int Rows,
                      int Cols,
					  int Direction){
// clang-format off
    #pragma HLS INLINE OFF
// clang-format on		
						  
#ifndef __SYNTHESIS__
    assert(((TYPE == XF_8UC1) || (TYPE == XF_8UC3)) &&
           "Input TYPE must be XF_8UC1 for 1-channel, XF_8UC3 for 3-channel");
    assert(((NPC == XF_NPPC1) || (NPC == XF_NPPC4)) && "NPC must be XF_NPPC1, XF_NPPC4 ");
    assert((Rows <= ROWS) && (Cols <= COLS) && "COLS should be greater than input image size ");
#endif
	const int NPC_COLS = COLS >> XF_BITSHIFT(NPC);
    //const int ROWS = Rows;

    uint16_t NPCCols = Cols >> XF_BITSHIFT(NPC);

    ap_uint<XF_WORDDEPTH(XF_WORDWIDTH(TYPE, NPC))> SrcRow[NPC_COLS];
    ap_uint<XF_WORDDEPTH(XF_WORDWIDTH(TYPE, NPC))> DstRow[NPC_COLS];

// clang-format off
#pragma HLS BIND_STORAGE variable = SrcRow type = ram_s2p impl = bram
#pragma HLS BIND_STORAGE variable = DstRow type = ram_s2p impl = bram
    // clang-format on

    for (int r = 0; r < Rows; r++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=ROWS max=ROWS
		#pragma HLS DATAFLOW
        // clang-format on

        xf::cv::_Axi2Mat<PTR_WIDTH, TYPE, NPC_COLS, ROWS, COLS, NPC>(SrcPtr, SrcRow, Rows, Cols, Direction, r);

        xf::cv::_FlipRow<TYPE, NPC_COLS, ROWS, COLS, NPC>(SrcRow, DstRow, Rows, NPCCols, Direction);

        xf::cv::_Mat2Axi<PTR_WIDTH, TYPE, NPC_COLS, ROWS, COLS, NPC>(DstRow, DstPtr, Rows, Cols, Direction, r);
    }

    return;
}
} // namespace cv
} // namespace xf
#endif //_XF_FLIP_HPP_
