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
#ifndef _XF_LENSSHDING_CONFIG_HPP_
#define _XF_LENSSHDING_CONFIG_HPP_

#include "common/xf_common.hpp"
#include "hls_math.h"
#include "hls_stream.h"

/**
 * @file xf_lensshadingcorrection.hpp
 * This file is part of Vitis Vision Library.
 */

template <typename T>
T xf_satcast_lsc(int in_val){};

template <>
inline ap_uint<8> xf_satcast_lsc<ap_uint<8> >(int v) {
    v = (v > 255 ? 255 : v);
    v = (v < 0 ? 0 : v);
    return v;
};
template <>
inline ap_uint<10> xf_satcast_lsc<ap_uint<10> >(int v) {
    v = (v > 1023 ? 1023 : v);
    v = (v < 0 ? 0 : v);
    return v;
};
template <>
inline ap_uint<12> xf_satcast_lsc<ap_uint<12> >(int v) {
    v = (v > 4095 ? 4095 : v);
    v = (v < 0 ? 0 : v);
    return v;
};
template <>
inline ap_uint<16> xf_satcast_lsc<ap_uint<16> >(int v) {
    v = (v > 65535 ? 65535 : v);
    v = (v < 0 ? 0 : v);
    return v;
};

namespace xf {
namespace cv {

/**
 * @tparam SRC_T input type
 * @tparam DST_T ouput type
 * @tparam ROWS rows of the input and output image
 * @tparam COLS cols of the input and output image
 * @tparam NPC number of pixels processed per cycle
 * @param src input image
 * @param dst output image
 */
template <int SRC_T, int DST_T, int ROWS, int COLS, int NPC = 1>
void Lscdistancebased(xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& src, xf::cv::Mat<DST_T, ROWS, COLS, NPC>& dst) {
    int rows = src.rows;
    int cols = src.cols >> XF_BITSHIFT(NPC);

    assert(((rows <= ROWS) && (cols <= COLS)) && "ROWS and COLS should be greater than input image");

    short center_pixel_pos_x = (src.cols >> 1);
    short center_pixel_pos_y = (rows >> 1);
    short y_distance = rows - center_pixel_pos_y;
    short x_distance = src.cols - center_pixel_pos_x;
    float y_2 = y_distance * y_distance;
    float x_2 = x_distance * x_distance;

    float max_distance = std::sqrt(y_2 + x_2);
    // ap_fixed<48,24> max_distance_inv = (1/max_distance);
    XF_TNAME(SRC_T, NPC) in_pix, in_pix1, out_pix;
    const int STEP = XF_DTPIXELDEPTH(SRC_T, NPC);
    float a = 0.01759;
    float b = 28.37;
    float c = 13.36;

    for (int i = 0; i < rows; i++) {
// clang-format off
#pragma HLS LOOP_TRIPCOUNT min=ROWS max=ROWS
        // clang-format on
        for (int j = 0; j < cols; j++) {
// clang-format off
#pragma HLS LOOP_TRIPCOUNT min=COLS/NPC max=COLS/NPC
#pragma HLS pipeline II=1
#pragma HLS LOOP_FLATTEN OFF
            // clang-format on
            float y_dist = center_pixel_pos_y - i;

            float y_dist_2 = y_dist * y_dist;

            in_pix = src.read(i * (cols) + j);

            for (ap_uint<9> p = 0; p < XF_NPIXPERCYCLE(NPC); p++) {
                float x_dist = center_pixel_pos_x - (j * NPC + p);
                float x_dist_2 = x_dist * x_dist;

                float xy_2 = std::sqrt(y_dist_2 + x_dist_2);
                float distance = xy_2 / max_distance;

                float gain_val = (a * ((distance + b) * (distance + b))) - c;

                for (ap_uint<9> k = 0; k < XF_CHANNELS(SRC_T, NPC); k++) {
// clang-format off
#pragma HLS unroll
                    // clang-format on

                    XF_CTUNAME(SRC_T, NPC)
                    val = in_pix.range((k + p * 3) * STEP + STEP - 1, (k + p * 3) * STEP);
                    int value = (int)(val * gain_val);

                    out_pix.range((k + p * 3) * STEP + STEP - 1, (k + p * 3) * STEP) =
                        xf_satcast_lsc<XF_CTUNAME(SRC_T, NPC)>(value);
                }
            }

            dst.write(i * cols + j, out_pix);
        }
    }
}
}
}

#endif
