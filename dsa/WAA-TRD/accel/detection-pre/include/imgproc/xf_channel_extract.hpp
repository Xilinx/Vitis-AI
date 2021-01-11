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

#ifndef _XF_CHANNEL_EXTRACT_HPP_
#define _XF_CHANNEL_EXTRACT_HPP_

#include "hls_stream.h"
#include "ap_int.h"
#include "common/xf_common.hpp"
#include "common/xf_utility.hpp"

namespace xf {
namespace cv {

/*****************************************************************************
 * 	xFChannelExtract: Extracts one channel from a multiple _channel image
 *
 *	# Parameters
 *	_src	  :	 source image as stream
 *	_dst	  :	 destination image as stream
 * 	_channel :  enumeration specified in < xf_channel_extract_e >
 ****************************************************************************/
template <int ROWS, int COLS, int SRC_T, int DST_T, int NPC, int TC>
void xfChannelExtractKernel(xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _src_mat,
                            xf::cv::Mat<DST_T, ROWS, COLS, NPC>& _dst_mat,
                            uint16_t _channel,
                            uint16_t height,
                            uint16_t width) {
    //#define XF_STEP 8
    const int noofbits = XF_DTPIXELDEPTH(SRC_T, NPC);

    ap_uint<13> i, j, k;
    XF_TNAME(SRC_T, NPC) in_pix;
    XF_TNAME(DST_T, NPC) out_pix;
    ap_uint<XF_PIXELDEPTH(DST_T)> result;
    int shift = 0;
    int bitdepth_src = XF_DTPIXELDEPTH(SRC_T, NPC) / XF_CHANNELS(SRC_T, NPC);
    int bitdepth_dst = XF_DTPIXELDEPTH(DST_T, NPC) / XF_CHANNELS(DST_T, NPC);

    if (_channel == XF_EXTRACT_CH_0 | _channel == XF_EXTRACT_CH_R | _channel == XF_EXTRACT_CH_Y) {
        shift = 0;
    } else if (_channel == XF_EXTRACT_CH_1 | _channel == XF_EXTRACT_CH_G | _channel == XF_EXTRACT_CH_U) {
        shift = noofbits;
    } else if (_channel == XF_EXTRACT_CH_2 | _channel == XF_EXTRACT_CH_B | _channel == XF_EXTRACT_CH_V) {
        shift = noofbits * 2;
    } else if (_channel == XF_EXTRACT_CH_3 | _channel == XF_EXTRACT_CH_A) {
        shift = noofbits * 3;
    }

RowLoop:
    for (i = 0; i < height; i++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=ROWS max=ROWS
        #pragma HLS LOOP_FLATTEN off
    // clang-format on
    ColLoop:
        for (j = 0; j < width; j++) {
// clang-format off
            #pragma HLS LOOP_TRIPCOUNT min=TC max=TC
            #pragma HLS pipeline
            // clang-format on
            int y;
            in_pix = _src_mat.read(i * width + j);

        ProcLoop:
            for (k = 0; k < (noofbits << XF_BITSHIFT(NPC)); k += noofbits) {
// clang-format off
                #pragma HLS unroll
                // clang-format on
                y = k * (XF_CHANNELS(SRC_T, NPC));
                result = in_pix.range(y + shift + noofbits - 1, y + shift);
                out_pix.range(k + (noofbits - 1), k) = result;
            }

            _dst_mat.write(i * width + j, out_pix);
        } // ColLoop
    }     // RowLoop
}

template <int SRC_T, int DST_T, int ROWS, int COLS, int NPC = 1>
void extractChannel(xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _src_mat,
                    xf::cv::Mat<DST_T, ROWS, COLS, NPC>& _dst_mat,
                    uint16_t _channel) {
#ifndef __SYNTHESIS__
    assert(((_channel == XF_EXTRACT_CH_0) || (_channel == XF_EXTRACT_CH_1) || (_channel == XF_EXTRACT_CH_2) ||
            (_channel == XF_EXTRACT_CH_3) || (_channel == XF_EXTRACT_CH_R) || (_channel == XF_EXTRACT_CH_G) ||
            (_channel == XF_EXTRACT_CH_B) || (_channel == XF_EXTRACT_CH_A) || (_channel == XF_EXTRACT_CH_Y) ||
            (_channel == XF_EXTRACT_CH_U) || (_channel == XF_EXTRACT_CH_V)) &&
           "Invalid Channel Value. See xf_channel_extract_e enumerated type");
    assert(((_src_mat.rows <= ROWS) && (_src_mat.cols <= COLS)) && "ROWS and COLS should be greater than input image");
    assert(((_dst_mat.rows <= ROWS) && (_dst_mat.cols <= COLS)) && "ROWS and COLS should be greater than input image");
    assert((SRC_T == XF_8UC4 || SRC_T == XF_8UC3) && (DST_T == XF_8UC1) &&
           "Source image should be of 4 channels and destination image of 1 channel");
//	assert(((NPC == XF_NPPC1)) && "NPC must be XF_NPPC1");
#endif
    short width = _src_mat.cols >> XF_BITSHIFT(NPC);

// clang-format off
    #pragma HLS INLINE OFF
    // clang-format on

    xfChannelExtractKernel<ROWS, COLS, SRC_T, DST_T, NPC, (COLS >> XF_BITSHIFT(NPC))>(_src_mat, _dst_mat, _channel,
                                                                                      _src_mat.rows, width);
}
} // namespace cv
} // namespace xf

#endif //_XF_CHANNEL_EXTRACT_HPP_
