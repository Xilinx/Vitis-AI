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

#ifndef _XF_CHANNEL_COMBINE_HPP_
#define _XF_CHANNEL_COMBINE_HPP_

#include "ap_int.h"
#include "common/xf_common.hpp"
#include "common/xf_utility.hpp"
#include "hls_stream.h"

namespace xf {
namespace cv {

#define XF_STEP_NEXT 8

/********************************************************************
 * 	ChannelCombine: combine multiple 8-bit planes into one
 *******************************************************************/
template <int ROWS, int COLS, int SRC_T, int DST_T, int NPC, int TC>
void xfChannelCombineKernel(xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _in1,
                            xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _in2,
                            xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _in3,
                            xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _in4,
                            xf::cv::Mat<DST_T, ROWS, COLS, NPC>& _out,
                            uint16_t height,
                            uint16_t width) {
    XF_TNAME(SRC_T, NPC) val1, val2, val3, val4;

    width = width >> (XF_BITSHIFT(NPC));
    uchar_t channel1, channel2, channel3, channel4;

    const int noofbits = XF_DTPIXELDEPTH(SRC_T, NPC);
    ap_uint<13> i, j, k;
RowLoop:
    for (i = 0; i < height; i++) {
// clang-format off
        #pragma HLS LOOP_FLATTEN off
        #pragma HLS LOOP_TRIPCOUNT min=ROWS max=ROWS
    // clang-format on
    ColLoop:
        for (j = 0; j < width; j++) {
// clang-format off
            #pragma HLS LOOP_TRIPCOUNT min=TC max=TC
            // clang-format on
            XF_TNAME(DST_T, NPC) res;

            val1 = (XF_TNAME(SRC_T, NPC))(_in1.read(i * width + j));
            val2 = (XF_TNAME(SRC_T, NPC))(_in2.read(i * width + j));
            val3 = (XF_TNAME(SRC_T, NPC))(_in3.read(i * width + j));
            val4 = (XF_TNAME(SRC_T, NPC))(_in4.read(i * width + j));

        ProcLoop:
            for (k = 0; k < (noofbits << XF_BITSHIFT(NPC)); k += noofbits) {
// clang-format off
                #pragma HLS UNROLL
                // clang-format on
                int y = k * XF_CHANNELS(DST_T, NPC);
                channel1 = val1.range(k + (noofbits - 1), k); // B
                channel2 = val2.range(k + (noofbits - 1), k); // G
                channel3 = val3.range(k + (noofbits - 1), k); // R
                channel4 = val4.range(k + (noofbits - 1), k); // A

                uint32_t result = ((uint32_t)channel3 << 0) | ((uint32_t)channel2 << noofbits) |
                                  ((uint32_t)channel1 << noofbits * 2) | ((uint32_t)channel4 << noofbits * 3);

                res.range(y + (XF_PIXELWIDTH(DST_T, NPC) - 1), y) = result;
            } // ProcLoop
            _out.write((i * width + j), (XF_TNAME(DST_T, NPC))res);
        } // ColLoop
    }     // RowLoop
}

template <int ROWS, int COLS, int SRC_T, int DST_T, int NPC, int TC>
void xfChannelCombineKernel(xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _in1,
                            xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _in2,
                            xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _in3,
                            xf::cv::Mat<DST_T, ROWS, COLS, NPC>& _out,
                            uint16_t height,
                            uint16_t width) {
    XF_TNAME(SRC_T, NPC) val1, val2, val3;
    uchar_t channel1, channel2, channel3;
    const int noofbits = XF_DTPIXELDEPTH(SRC_T, NPC);
    width = width >> (XF_BITSHIFT(NPC));
    int rows = height, cols = width;

RowLoop:
    for (int i = 0; i < rows; i++) {
// clang-format off
        #pragma HLS LOOP_FLATTEN off
        #pragma HLS LOOP_TRIPCOUNT min=ROWS max=ROWS
    // clang-format on
    ColLoop:
        for (int j = 0; j < cols; j++) {
// clang-format off
            #pragma HLS LOOP_TRIPCOUNT min=TC max=TC
            // clang-format on
            XF_TNAME(DST_T, NPC) res;

            val1 = (XF_TNAME(SRC_T, NPC))(_in1.read(i * cols + j));
            val2 = (XF_TNAME(SRC_T, NPC))(_in2.read(i * cols + j));
            val3 = (XF_TNAME(SRC_T, NPC))(_in3.read(i * cols + j));

        ProcLoop:
            for (int k = 0; k < (noofbits << XF_BITSHIFT(NPC)); k += noofbits) {
// clang-format off
                #pragma HLS UNROLL
                // clang-format on
                int y = k * XF_CHANNELS(DST_T, NPC);
                channel1 = val1.range(k + (noofbits - 1), k); // B
                channel2 = val2.range(k + (noofbits - 1), k); // G
                channel3 = val3.range(k + (noofbits - 1), k); // R

                uint32_t result = (((uint32_t)channel3 << 0) | ((uint32_t)channel2 << noofbits) |
                                   ((uint32_t)channel1 << noofbits * 2));

                res.range(y + (XF_PIXELWIDTH(DST_T, NPC) - 1), y) = result;
            }
            _out.write((i * cols + j), (XF_TNAME(DST_T, NPC))res);
        }
    }
}

template <int ROWS, int COLS, int SRC_T, int DST_T, int NPC, int TC>
void xfChannelCombineKernel(xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _in1,
                            xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _in2,
                            xf::cv::Mat<DST_T, ROWS, COLS, NPC>& _out,
                            uint16_t height,
                            uint16_t width) {
    XF_TNAME(SRC_T, NPC) val1, val2;
    uchar_t channel1, channel2;
    const int noofbits = XF_DTPIXELDEPTH(SRC_T, NPC);
    width = width >> (XF_BITSHIFT(NPC));
    int rows = height, cols = width;

RowLoop:
    for (int i = 0; i < rows; i++) {
// clang-format off
        #pragma HLS LOOP_FLATTEN off
        #pragma HLS LOOP_TRIPCOUNT min=ROWS max=ROWS
    // clang-format on
    ColLoop:
        for (int j = 0; j < cols; j++) {
// clang-format off
            #pragma HLS LOOP_TRIPCOUNT min=TC max=TC
            // clang-format on
            XF_TNAME(DST_T, NPC) res;

            val1 = (XF_TNAME(SRC_T, NPC))(_in1.read(i * cols + j));
            val2 = (XF_TNAME(SRC_T, NPC))(_in2.read(i * cols + j));

        ProcLoop:
            for (int k = 0; k < (noofbits << XF_BITSHIFT(NPC)); k += noofbits) {
// clang-format off
                #pragma HLS UNROLL
                // clang-format on
                int y = k * XF_CHANNELS(DST_T, NPC);
                channel1 = val1.range(k + (noofbits - 1), k); // B
                channel2 = val2.range(k + (noofbits - 1), k); // G

                uint32_t result = ((uint32_t)channel1 << 0) | ((uint32_t)channel2 << noofbits);
                res.range(y + (XF_PIXELWIDTH(DST_T, NPC) - 1), y) = result;
            }
            _out.write((i * cols + j), (XF_TNAME(DST_T, NPC))res);
        }
    }
}

/*******************************

Kernel for 2 input configuration

*******************************/

template <int SRC_T, int DST_T, int ROWS, int COLS, int NPC = 1>
void merge(xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _src1,
           xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _src2,
           xf::cv::Mat<DST_T, ROWS, COLS, NPC>& _dst) {
#ifndef __SYNTHESIS__
    assert(((_src1.rows <= ROWS) && (_src1.cols <= COLS)) && "ROWS and COLS should be greater than input image");
    assert(((_src2.rows <= ROWS) && (_src2.cols <= COLS)) && "ROWS and COLS should be greater than input image");
    assert(((_dst.rows <= ROWS) && (_dst.cols <= COLS)) && "ROWS and COLS should be greater than input image");
    assert((SRC_T == XF_8UC1) && (DST_T == XF_8UC2) &&
           "Source image should be of 1 channel and destination image of 2 "
           "channels");
//    assert(((NPC == XF_NPPC1)) && "NPC must be XF_NPPC1");
#endif

// clang-format off
    #pragma HLS inline off
    // clang-format on

    xfChannelCombineKernel<ROWS, COLS, SRC_T, DST_T, NPC, (COLS >> (XF_BITSHIFT(NPC)))>(_src1, _src2, _dst, _src1.rows,
                                                                                        _src1.cols);
}

/*******************************

Kernel for 3 input configuration

*******************************/

template <int SRC_T, int DST_T, int ROWS, int COLS, int NPC = 1>
void merge(xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _src1,
           xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _src2,
           xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _src3,
           xf::cv::Mat<DST_T, ROWS, COLS, NPC>& _dst) {
#ifndef __SYNTHESIS__
    assert(((_src1.rows <= ROWS) && (_src1.cols <= COLS)) && "ROWS and COLS should be greater than input image");
    assert(((_src2.rows <= ROWS) && (_src2.cols <= COLS)) && "ROWS and COLS should be greater than input image");
    assert(((_src3.rows <= ROWS) && (_src3.cols <= COLS)) && "ROWS and COLS should be greater than input image");
    assert(((_dst.rows <= ROWS) && (_dst.cols <= COLS)) && "ROWS and COLS should be greater than input image");
    assert((SRC_T == XF_8UC1) && (DST_T == XF_8UC3) &&
           "Source image should be of 1 channel and destination image of 3 "
           "channels");
//    assert(((NPC == XF_NPPC1)) && "NPC must be XF_NPPC1");
#endif

// clang-format off
    #pragma HLS inline off
    // clang-format on

    xfChannelCombineKernel<ROWS, COLS, SRC_T, DST_T, NPC, (COLS >> (XF_BITSHIFT(NPC)))>(_src1, _src2, _src3, _dst,
                                                                                        _src1.rows, _src1.cols);
}

/*******************************

Kernel for 4 input configuration

*******************************/

template <int SRC_T, int DST_T, int ROWS, int COLS, int NPC = 1>
void merge(xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _src1,
           xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _src2,
           xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _src3,
           xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _src4,
           xf::cv::Mat<DST_T, ROWS, COLS, NPC>& _dst) {
#ifndef __SYNTHESIS__
    assert(((_src1.rows <= ROWS) && (_src1.cols <= COLS)) && "ROWS and COLS should be greater than input image");
    assert(((_src2.rows <= ROWS) && (_src2.cols <= COLS)) && "ROWS and COLS should be greater than input image");
    assert(((_src3.rows <= ROWS) && (_src3.cols <= COLS)) && "ROWS and COLS should be greater than input image");
    assert(((_src4.rows <= ROWS) && (_src4.cols <= COLS)) && "ROWS and COLS should be greater than input image");
    assert(((_dst.rows <= ROWS) && (_dst.cols <= COLS)) && "ROWS and COLS should be greater than input image");
    assert((SRC_T == XF_8UC1) && (DST_T == XF_8UC4) &&
           "Source image should be of 1 channel and destination image of 4 "
           "channels");
//    assert(((NPC == XF_NPPC1)) && "NPC must be XF_NPPC1");
#endif

// clang-format off
    #pragma HLS inline off
    // clang-format on

    xfChannelCombineKernel<ROWS, COLS, SRC_T, DST_T, NPC, (COLS >> (XF_BITSHIFT(NPC)))>(_src1, _src2, _src3, _src4,
                                                                                        _dst, _src1.rows, _src1.cols);
}

} // namespace cv
} // namespace xf

#endif //_XF_CHANNEL_COMBINE_HPP_
