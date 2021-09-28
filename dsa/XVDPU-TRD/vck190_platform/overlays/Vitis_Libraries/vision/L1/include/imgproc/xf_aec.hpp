/*
 * Copyright 2020 Xilinx, Inc.
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
#ifndef _XF_AEC_HPP_
#define _XF_AEC_HPP_

#include "common/xf_common.hpp"
#include "hls_math.h"
#include "hls_stream.h"
#include "xf_bgr2hsv.hpp"
#include "xf_channel_combine.hpp"
#include "xf_channel_extract.hpp"
#include "xf_cvt_color.hpp"
#include "xf_cvt_color_1.hpp"
#include "xf_duplicateimage.hpp"
#include "xf_hist_equalize.hpp"
#include "xf_histogram.hpp"

template <typename T>
T xf_satcast_aec(int in_val){};

template <>
inline ap_uint<8> xf_satcast_aec<ap_uint<8> >(int v) {
    v = (v > 255 ? 255 : v);
    v = (v < 0 ? 0 : v);
    return v;
};
template <>
inline ap_uint<10> xf_satcast_aec<ap_uint<10> >(int v) {
    v = (v > 1023 ? 1023 : v);
    v = (v < 0 ? 0 : v);
    return v;
};
template <>
inline ap_uint<12> xf_satcast_aec<ap_uint<12> >(int v) {
    v = (v > 4095 ? 4095 : v);
    v = (v < 0 ? 0 : v);
    return v;
};
template <>
inline ap_uint<16> xf_satcast_aec<ap_uint<16> >(int v) {
    v = (v > 65535 ? 65535 : v);
    v = (v < 0 ? 0 : v);
    return v;
};

namespace xf {
namespace cv {
template <int SRC_T, int DST_T, int SIN_CHANNEL_TYPE, int ROWS, int COLS, int NPC = 1>
void autoexposurecorrection(xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& src,
                            xf::cv::Mat<DST_T, ROWS, COLS, NPC>& dst,
                            uint32_t hist_array1[1][256],
                            uint32_t hist_array2[1][256]) {
#pragma HLS INLINE OFF

    int rows = src.rows;
    int cols = src.cols;

    uint16_t cols_shifted = cols >> (XF_BITSHIFT(NPC));
    uint16_t rows_shifted = rows;

    xf::cv::Mat<SRC_T, ROWS, COLS, NPC> bgr2hsv(rows, cols);

    xf::cv::Mat<SRC_T, ROWS, COLS, NPC> hsvimg1(rows, cols);
    xf::cv::Mat<SRC_T, ROWS, COLS, NPC> hsvimg2(rows, cols);
    xf::cv::Mat<SRC_T, ROWS, COLS, NPC> hsvimg3(rows, cols);

    xf::cv::Mat<SIN_CHANNEL_TYPE, ROWS, COLS, NPC> himage(rows, cols);
    xf::cv::Mat<SIN_CHANNEL_TYPE, ROWS, COLS, NPC> simage(rows, cols);
    xf::cv::Mat<SIN_CHANNEL_TYPE, ROWS, COLS, NPC> vimage(rows, cols);
    xf::cv::Mat<SIN_CHANNEL_TYPE, ROWS, COLS, NPC> vimage1(rows, cols);
    xf::cv::Mat<SIN_CHANNEL_TYPE, ROWS, COLS, NPC> vimage2(rows, cols);

    xf::cv::Mat<SIN_CHANNEL_TYPE, ROWS, COLS, NPC> vimage_eq(rows, cols);
    xf::cv::Mat<SRC_T, ROWS, COLS, NPC> imgHelper6(rows, cols);

    assert(((rows <= ROWS) && (cols <= COLS)) && "ROWS and COLS should be greater than input image");

// clang-format off
#pragma HLS DATAFLOW
    // clang-format on

    // Convert RGBA to HSV:
    xf::cv::bgr2hsv<SRC_T, ROWS, COLS, NPC>(src, bgr2hsv);

    xf::cv::duplicateimages<SRC_T, ROWS, COLS, NPC>(bgr2hsv, hsvimg1, hsvimg2, hsvimg3);

    xf::cv::extractChannel<SRC_T, SIN_CHANNEL_TYPE, ROWS, COLS, NPC>(hsvimg1, himage, 0);

    xf::cv::extractChannel<SRC_T, SIN_CHANNEL_TYPE, ROWS, COLS, NPC>(hsvimg2, simage, 1);

    xf::cv::extractChannel<SRC_T, SIN_CHANNEL_TYPE, ROWS, COLS, NPC>(hsvimg3, vimage, 2);

    xf::cv::duplicateMat(vimage, vimage1, vimage2);

    // xf::cv::equalizeHist<SIN_CHANNEL_TYPE, ROWS, COLS, NPC>(vimage1, vimage2,
    // vimage_eq);
    xFHistogramKernel<SIN_CHANNEL_TYPE, ROWS, COLS, XF_DEPTH(SIN_CHANNEL_TYPE, NPC), NPC,
                      XF_WORDWIDTH(SIN_CHANNEL_TYPE, NPC), ((COLS >> (XF_BITSHIFT(NPC))) >> 1),
                      XF_CHANNELS(SIN_CHANNEL_TYPE, NPC)>(vimage1, hist_array1, rows_shifted, cols_shifted);

    xFEqualize<SIN_CHANNEL_TYPE, ROWS, COLS, XF_DEPTH(SIN_CHANNEL_TYPE, NPC), NPC, XF_WORDWIDTH(SIN_CHANNEL_TYPE, NPC),
               (COLS >> XF_BITSHIFT(NPC))>(vimage2, hist_array2, vimage_eq, rows_shifted, cols_shifted);

    xf::cv::merge<SIN_CHANNEL_TYPE, SRC_T, ROWS, COLS, NPC>(vimage_eq, simage, himage, imgHelper6);

    xf::cv::hsv2bgr<SRC_T, SRC_T, ROWS, COLS, NPC>(imgHelper6, dst);
}
}
}

#endif
