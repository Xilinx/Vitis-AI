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

#ifndef _XF_GC_HPP_
#define _XF_GC_HPP_

#include "common/xf_common.hpp"
#include "hls_stream.h"

#ifndef XF_IN_STEP
#define XF_IN_STEP 16
#endif
#ifndef XF_OUT_STEP
#define XF_OUT_STEP 16
#endif

#define R_GAIN 157
#define B_GAIN 140

/**Utility macros and functions**/

#define MAXVAL(pixeldepth) ((1 << pixeldepth) - 1)
#define XF_UCHAR_MAX 255
#define XF_UTENBIT_MAX 1023
#define XF_UTWELVEBIT_MAX 4095
#define XF_USHORT_MAX 65535

template <typename T>
T xf_satcast_gain(int in_val){};

template <>
inline ap_uint<8> xf_satcast_gain<ap_uint<8> >(int v) {
    return (v > MAXVAL(8) ? XF_UCHAR_MAX : v);
};
template <>
inline ap_uint<10> xf_satcast_gain<ap_uint<10> >(int v) {
    return (v > MAXVAL(10) ? XF_UTENBIT_MAX : v);
};
template <>
inline ap_uint<12> xf_satcast_gain<ap_uint<12> >(int v) {
    return (v > MAXVAL(12) ? XF_UTWELVEBIT_MAX : v);
};
template <>
inline ap_uint<16> xf_satcast_gain<ap_uint<16> >(int v) {
    return (v > MAXVAL(16) ? XF_USHORT_MAX : v);
};

namespace xf {

namespace cv {

template <int BFORMAT,
          int SRC_T,
          int ROWS,
          int COLS,
          int NPC,
          int PLANES,
          int DEPTH_SRC,
          int DEPTH_DST,
          int WORDWIDTH_SRC,
          int WORDWIDTH_DST,
          int TC>
void gaincontrolkernel(xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& src1,
                       xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& dst,
                       uint16_t height,
                       uint16_t width) {
    ap_uint<13> i, j, k, l;

    const int STEP = XF_PIXELWIDTH(SRC_T, NPC);

    XF_TNAME(SRC_T, NPC) pxl_pack_out;
    XF_TNAME(SRC_T, NPC) pxl_pack1, pxl_pack2;
RowLoop:
    for (i = 0; i < height; i++) {
#pragma HLS LOOP_TRIPCOUNT min = ROWS max = ROWS
#pragma HLS LOOP_FLATTEN OFF
    ColLoop:
        for (j = 0; j < width; j++) {
#pragma HLS LOOP_TRIPCOUNT min = TC max = TC
#pragma HLS pipeline

            pxl_pack1 = (src1.read(i * width + j)); // reading from 1st input stream

        ProcLoop:
            for (l = 0; l < (XF_NPIXPERCYCLE(NPC) * XF_CHANNELS(SRC_T, NPC)); l++) {
                XF_PTNAME(DEPTH_SRC)
                pxl1 = pxl_pack1.range(l * STEP + STEP - 1, l * STEP); // extracting each pixel in case of 8-pixel mode
                XF_PTNAME(DEPTH_SRC) t;
                bool cond1 = 0, cond2 = 0;

                if (NPC == XF_NPPC1) {
                    cond1 = (j % 2 == 0);
                    cond2 = (j % 2 != 0);
                } else {
                    cond1 = ((l % 2) == 0);
                    cond2 = ((l % 2) != 0);
                }

                if (BFORMAT == XF_BAYER_RG) {
                    if (i % 2 == 0 && cond1) {
                        XF_CTUNAME(SRC_T, NPC) v1 = pxl1;
                        int v2 = (int)((v1 * R_GAIN) >> 7);
                        t = xf_satcast_gain<XF_CTUNAME(SRC_T, NPC)>(v2);
                    } else if (i % 2 != 0 && cond2) {
                        XF_CTUNAME(SRC_T, NPC) v1 = pxl1;
                        int v2 = (int)((v1 * B_GAIN) >> 7);
                        t = xf_satcast_gain<XF_CTUNAME(SRC_T, NPC)>(v2);
                    } else {
                        t = pxl1;
                    }
                }
                if (BFORMAT == XF_BAYER_GR) {
                    if (i % 2 == 0 && cond2) {
                        XF_CTUNAME(SRC_T, NPC) v1 = pxl1;
                        int v2 = (int)((v1 * R_GAIN) >> 7);
                        t = xf_satcast_gain<XF_CTUNAME(SRC_T, NPC)>(v2);
                    } else if (i % 2 != 0 && cond1) {
                        XF_CTUNAME(SRC_T, NPC) v1 = pxl1;
                        int v2 = (int)((v1 * B_GAIN) >> 7);
                        t = xf_satcast_gain<XF_CTUNAME(SRC_T, NPC)>(v2);
                    } else {
                        t = pxl1;
                    }
                }
                if (BFORMAT == XF_BAYER_BG) {
                    if (i % 2 == 0 && cond1) {
                        XF_CTUNAME(SRC_T, NPC) v1 = pxl1;
                        int v2 = (int)((v1 * B_GAIN) >> 7);
                        t = xf_satcast_gain<XF_CTUNAME(SRC_T, NPC)>(v2);
                    } else if (i % 2 == 0 && cond2) {
                        XF_CTUNAME(SRC_T, NPC) v1 = pxl1;
                        int v2 = (int)((v1 * R_GAIN) >> 7);
                        t = xf_satcast_gain<XF_CTUNAME(SRC_T, NPC)>(v2);
                    } else {
                        t = pxl1;
                    }
                }
                if (BFORMAT == XF_BAYER_GB) {
                    if (i % 2 == 0 && cond2) {
                        XF_CTUNAME(SRC_T, NPC) v1 = pxl1;
                        int v2 = (int)((v1 * B_GAIN) >> 7);
                        t = xf_satcast_gain<XF_CTUNAME(SRC_T, NPC)>(v2);
                    } else if (i % 2 != 0 && cond1) {
                        XF_CTUNAME(SRC_T, NPC) v1 = pxl1;
                        int v2 = (int)((v1 * R_GAIN) >> 7);
                        t = xf_satcast_gain<XF_CTUNAME(SRC_T, NPC)>(v2);
                    } else {
                        t = pxl1;
                    }
                }

                pxl_pack_out.range(l * STEP + STEP - 1, l * STEP) = t;
            }

            dst.write(i * width + j, pxl_pack_out); // writing into ouput stream
        }
    }
}

template <int BFORMAT, int SRC_T, int ROWS, int COLS, int NPC = 1>
void gaincontrol(xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& src1, xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& dst) {
#pragma HLS INLINE OFF
#ifndef __SYNTHESIS__
    assert(((src1.rows == dst.rows) && (src1.cols == dst.cols)) && "Input and output image should be of same size");
    assert(((src1.rows <= ROWS) && (src1.cols <= COLS)) && "ROWS and COLS should be greater than input image");
#endif
    short width = src1.cols >> XF_BITSHIFT(NPC);

    gaincontrolkernel<BFORMAT, SRC_T, ROWS, COLS, NPC, XF_CHANNELS(SRC_T, NPC), XF_DEPTH(SRC_T, NPC),
                      XF_DEPTH(SRC_T, NPC), XF_WORDWIDTH(SRC_T, NPC), XF_WORDWIDTH(SRC_T, NPC),
                      (COLS >> XF_BITSHIFT(NPC))>(src1, dst, src1.rows, width);
}
}
}

#endif //_XF_GC_HPP_
