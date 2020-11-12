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

#ifndef _XF_LUT_HPP_
#define _XF_LUT_HPP_

#ifndef __cplusplus
#error C++ is needed to include this header
#endif

#include "hls_stream.h"
#include "common/xf_common.hpp"
#ifndef XF_IN_STEP
#define XF_IN_STEP 8
#endif
/**
 *  xfLUTKernel: The Table Lookup Image Kernel.
 *  This kernel uses each pixel in an image to index into a LUT
 *  and put the indexed LUT value into the output image.
 *	Input		 : _src, _lut
 *	Output		 : _dst
 */
namespace xf {
namespace cv {

template <int SRC_T,
          int ROWS,
          int COLS,
          int PLANES,
          int DEPTH,
          int NPC,
          int WORDWIDTH_SRC,
          int WORDWIDTH_DST,
          int COLS_TRIP>
void xFLUTKernel(xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _src,
                 xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _dst,
                 unsigned char* _lut,
                 uint16_t height,
                 uint16_t width) {
    width = width >> XF_BITSHIFT(NPC);
    ap_uint<13> i, j, k;

    uchar_t lut[XF_NPIXPERCYCLE(NPC) * PLANES][256];

    if ((NPC != 0) || (PLANES != 1)) {
// clang-format off
        #pragma HLS ARRAY_PARTITION variable=lut complete dim=1
        // clang-format on
    }

    // creating a temporary buffers for Resource Optimization and Performance optimization
    if ((NPC != 0) || (PLANES != 3)) {
        for (i = 0; i < (XF_NPIXPERCYCLE(NPC) * PLANES); i++) {
            for (j = 0; j < 256; j++) {
                lut[i][j] = _lut[j];
            }
        }
    }

    XF_SNAME(WORDWIDTH_SRC) val_src;
    XF_SNAME(WORDWIDTH_DST) val_dst;

rowLoop:
    for (i = 0; i < height; i++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=ROWS max=ROWS
        #pragma HLS LOOP_FLATTEN off
    // clang-format on

    colLoop:
        for (j = 0; j < width; j++) {
// clang-format off
            #pragma HLS LOOP_TRIPCOUNT min=COLS_TRIP max=COLS_TRIP
            #pragma HLS pipeline
            // clang-format on

            val_src = (XF_SNAME(WORDWIDTH_SRC))(_src.read(i * width + j)); // read the data from the input stream

            uchar_t l = 0;
            int c = 0;
        procLoop:
            for (k = 0; k < (XF_WORDDEPTH(WORDWIDTH_SRC)); k += XF_IN_STEP) {
// clang-format off
                #pragma HLS unroll
                // clang-format on
                XF_PTNAME(DEPTH) p;
                p = val_src.range(k + (XF_IN_STEP - 1), k); // Get bits from certain range of positions.

                // for Normal operation
                if ((NPC == XF_NPPC1) && (PLANES == 1)) {
                    val_dst.range(k + (XF_IN_STEP - 1), k) = _lut[p]; // Set bits in a range of positions.
                }

                // resource optimization and performance optimization
                else {
                    val_dst.range(k + (XF_IN_STEP - 1), k) = lut[l][p]; // Set bits in a range of positions.
                    l++;
                }
            }
            _dst.write(i * width + j, val_dst); // write the data into the output stream
        }
    }
}
template <int SRC_T, int ROWS, int COLS, int NPC = 1>
void LUT(xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _src, xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _dst, unsigned char* _lut) {
// clang-format off
    #pragma HLS INLINE OFF
	unsigned char height=_src.rows;
	unsigned char width=_src.cols;
	
	 #ifndef __SYNTHESIS__
    	assert((SRC_T == XF_8UC1 ) ||(SRC_T == XF_8UC3 ) && "input type must be XF_8UC1 or XF_8UC3");
    	assert(((NPC == XF_NPPC1) || (NPC == XF_NPPC8)) && "NPC must be XF_NPPC1 or XF_NPPC8");
    	assert(((height <= ROWS ) && (width <= COLS)) && "ROWS and COLS should be greater than input image");
	#endif
	
    xFLUTKernel<SRC_T, ROWS, COLS, XF_CHANNELS(SRC_T, NPC), XF_DEPTH(SRC_T, NPC), NPC, XF_WORDWIDTH(SRC_T, NPC),
               XF_WORDWIDTH(SRC_T, NPC),(COLS >> XF_BITSHIFT(NPC))>(_src, _dst, _lut, _src.rows, _src.cols);
}
} // namespace cv
} // namespace xf
#endif
