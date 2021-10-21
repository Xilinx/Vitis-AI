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

#ifndef _XF_CCM_HPP_
#define _XF_CCM_HPP_

#ifndef __cplusplus
#error C++ is needed to include this header
#endif

typedef unsigned short uint16_t;
typedef unsigned char uchar;

/**
 * @file xf_colorcorrectionmatrix.hpp
 * This file is part of Vitis Vision Library.
 */

const ap_fixed<32, 4> bt2020_bt709_arr_hls_1[3][3] = {
    {1.6605, -0.5876, -0.0728}, {-0.1246, 1.1329, -0.0083}, {-0.0182, -0.1006, 1.1187}};

const ap_fixed<32, 4> bt2020_bt709_off_hls_1[3] = {0.0, 0.0, 0.0};

const ap_fixed<32, 4> bt709_bt2020_arr_hls_1[3][3] = {
    {0.627, 0.329, 0.0433}, {0.0691, 0.92, 0.0113}, {0.0164, 0.088, 0.896}};

const ap_fixed<32, 4> bt709_bt2020_off_hls_1[3] = {0.0, 0.0, 0.0};

const ap_fixed<32, 4> rgb_yuv_601_arr_hls_1[3][3] = {
    {0.257, 0.504, 0.098}, {-0.148, -0.291, 0.439}, {0.439, -0.368, -0.071}};

const ap_fixed<32, 4> rgb_yuv_601_off_hls_1[3] = {0.0625, 0.500, 0.500};

const ap_fixed<32, 4> rgb_yuv_709_arr_hls_1[3][3] = {
    {0.183, 0.614, 0.062}, {-0.101, -0.338, 0.439}, {0.439, -0.399, -0.040}};

const ap_fixed<32, 4> rgb_yuv_709_off_hls_1[3] = {0.0625, 0.500, 0.500};

const ap_fixed<32, 4> rgb_yuv_2020_arr_hls_1[3][3] = {
    {0.225613, 0.582282, 0.050928}, {-0.119918, -0.309494, 0.429412}, {0.429412, -0.394875, -0.034537}};

const ap_fixed<32, 4> rgb_yuv_2020_off_hls_1[3] = {0.062745, 0.500, 0.500};

const ap_fixed<32, 4> yuv_rgb_601_arr_hls_1[3][3] = {
    {1.164, 0.000, 1.596}, {1.164, -0.813, -0.391}, {1.164, 2.018, 0.000}};

const ap_fixed<32, 4> yuv_rgb_601_off_hls_1[3] = {-0.87075, 0.52925, -1.08175};

const ap_fixed<32, 4> yuv_rgb_709_arr_hls_1[3][3] = {
    {1.164, 0.000, 1.793}, {1.164, -0.213, -0.534}, {1.164, 2.115, 0.000}};

const ap_fixed<32, 4> yuv_rgb_709_off_hls_1[3] = {-0.96925, 0.30075, -1.13025};

const ap_fixed<32, 4> yuv_rgb_2020_arr_hls_1[3][3] = {
    {1.164384, 0.000000, 1.717000}, {1.164384, -0.191603, -0.665274}, {1.164384, 2.190671, 0.000000}};

const ap_fixed<32, 4> yuv_rgb_2020_off_hls_1[3] = {-0.931559, 0.355379, -1.168395};

const ap_fixed<32, 4> full_to_16_235_arr_hls_1[3][3] = {
    {0.856305, 0.000000, 0.000000}, {0.000000, 0.856305, 0.000000}, {0.000000, 0.000000, 0.856305}};

const ap_fixed<32, 4> full_to_16_235_off_hls_1[3] = {0.0625, 0.0625, 0.0625};

const ap_fixed<32, 4> full_from_16_235_arr_hls_1[3][3] = {
    {1.167808, 0.000000, 0.000000}, {0.000000, 1.167808, 0.000000}, {0.000000, 0.000000, 1.167808}};

const ap_fixed<32, 4> full_from_16_235_off_hls_1[3] = {-0.0729880, -0.0729880, -0.0729880};

#include "common/xf_common.hpp"
#include "common/xf_utility.hpp"
#include "hls_stream.h"

template <typename T>
T xf_satcast_ccm(int in_val){};

template <>
inline ap_uint<8> xf_satcast_ccm<ap_uint<8> >(int v) {
    v = (v > 255 ? 255 : v);
    v = (v < 0 ? 0 : v);
    return v;
};
template <>
inline ap_uint<10> xf_satcast_ccm<ap_uint<10> >(int v) {
    v = (v > 1023 ? 1023 : v);
    v = (v < 0 ? 0 : v);
    return v;
};
template <>
inline ap_uint<12> xf_satcast_ccm<ap_uint<12> >(int v) {
    v = (v > 4095 ? 4095 : v);
    v = (v < 0 ? 0 : v);
    return v;
};
template <>
inline ap_uint<16> xf_satcast_ccm<ap_uint<16> >(int v) {
    v = (v > 65535 ? 65535 : v);
    v = (v < 0 ? 0 : v);
    return v;
};

namespace xf {
namespace cv {

template <int SRC_T,
          int ROWS,
          int COLS,
          int DEPTH,
          int NPC,
          int WORDWIDTH_SRC,
          int WORDWIDTH_DST,
          int COLS_TRIP,
          int S_DEPTH>
void xfccmkernel(xf::cv::Mat<SRC_T, ROWS, COLS, NPC, S_DEPTH>& _src_mat,
                 xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _dst_mat,
                 ap_uint<8> _ccm_type,
                 unsigned short height,
                 unsigned short width) {
    ap_fixed<32, 4> ccm_matrix[3][3];
    ap_fixed<32, 4> offsetarray[3];

    switch (_ccm_type) {
        case 0:
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    ccm_matrix[i][j] = bt2020_bt709_arr_hls_1[i][j];
                }
                offsetarray[i] = bt2020_bt709_off_hls_1[i];
            }

            break;
        case 1:
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    ccm_matrix[i][j] = bt709_bt2020_arr_hls_1[i][j];
                }
                offsetarray[i] = bt709_bt2020_off_hls_1[i];
            }

            break;
        case 2:
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    ccm_matrix[i][j] = rgb_yuv_601_arr_hls_1[i][j];
                }
                offsetarray[i] = rgb_yuv_601_off_hls_1[i];
            }

            break;
        case 3:
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    ccm_matrix[i][j] = rgb_yuv_709_arr_hls_1[i][j];
                }
                offsetarray[i] = rgb_yuv_709_off_hls_1[i];
            }

            break;
        case 4:
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    ccm_matrix[i][j] = rgb_yuv_2020_arr_hls_1[i][j];
                }
                offsetarray[i] = rgb_yuv_2020_off_hls_1[i];
            }

            break;
        case 5:
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    ccm_matrix[i][j] = yuv_rgb_601_arr_hls_1[i][j];
                }
                offsetarray[i] = yuv_rgb_601_off_hls_1[i];
            }

            break;
        case 6:
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    ccm_matrix[i][j] = yuv_rgb_709_arr_hls_1[i][j];
                }
                offsetarray[i] = yuv_rgb_709_off_hls_1[i];
            }

            break;
        case 7:
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    ccm_matrix[i][j] = yuv_rgb_2020_arr_hls_1[i][j];
                }
                offsetarray[i] = yuv_rgb_2020_off_hls_1[i];
            }

            break;
        case 8:
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    ccm_matrix[i][j] = full_to_16_235_arr_hls_1[i][j];
                }
                offsetarray[i] = full_to_16_235_off_hls_1[i];
            }

            break;
        case 9:
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    ccm_matrix[i][j] = full_from_16_235_arr_hls_1[i][j];
                }
                offsetarray[i] = full_from_16_235_off_hls_1[i];
            }

            break;
        default:
            break;
    }

    const int STEP = XF_DTPIXELDEPTH(SRC_T, NPC);
    ap_uint<13> i, j, k;
    XF_SNAME(WORDWIDTH_SRC) val_src;
    XF_SNAME(WORDWIDTH_DST) val_dst;

    int value_r = 0, value_g = 0, value_b = 0;

    XF_CTUNAME(SRC_T, NPC) r, g, b;
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

            val_src =
                (XF_SNAME(WORDWIDTH_SRC))(_src_mat.read(i * width + j)); // reading the source stream _src into val_src

            for (int p = 0; p < (XF_NPIXPERCYCLE(NPC) * XF_CHANNELS(SRC_T, NPC)); p = p + XF_CHANNELS(SRC_T, NPC)) {
// clang-format off
#pragma HLS unroll
                // clang-format on

                r = val_src.range(p * STEP + STEP - 1, p * STEP);
                g = val_src.range(p * STEP + (2 * STEP) - 1, p * STEP + STEP);
                b = val_src.range(p * STEP + (3 * STEP) - 1, p * STEP + 2 * STEP);

                ap_fixed<32, 24> value1 = (r * ccm_matrix[0][0]);
                ap_fixed<32, 24> value2 = (g * ccm_matrix[0][1]);
                ap_fixed<32, 24> value3 = (b * ccm_matrix[0][2]);

                ap_fixed<32, 24> value4 = (r * ccm_matrix[1][0]);
                ap_fixed<32, 24> value5 = (g * ccm_matrix[1][1]);
                ap_fixed<32, 24> value6 = (b * ccm_matrix[1][2]);

                ap_fixed<32, 24> value7 = (r * ccm_matrix[2][0]);
                ap_fixed<32, 24> value8 = (g * ccm_matrix[2][1]);
                ap_fixed<32, 24> value9 = (b * ccm_matrix[2][2]);

                value_r = (int)(value1 + value2 + value3 + offsetarray[0]);
                value_g = (int)(value4 + value5 + value6 + offsetarray[1]);
                value_b = (int)(value7 + value8 + value9 + offsetarray[2]);

                val_dst.range(p * STEP + STEP - 1, p * STEP) = xf_satcast_ccm<XF_CTUNAME(SRC_T, NPC)>(value_r);
                val_dst.range(p * STEP + (2 * STEP) - 1, p * STEP + STEP) =
                    xf_satcast_ccm<XF_CTUNAME(SRC_T, NPC)>(value_g);
                val_dst.range(p * STEP + (3 * STEP) - 1, p * STEP + 2 * STEP) =
                    xf_satcast_ccm<XF_CTUNAME(SRC_T, NPC)>(value_b);
            }

            _dst_mat.write(i * width + j, (val_dst)); // writing the val_dst into output stream _dst
        }
    }
}
/**
 * @tparam CCM_TYPE colorcorrection type
 * @tparam SRC_T input type
 * @tparam DST_T ouput type
 * @tparam ROWS rows of the input and output image
 * @tparam COLS cols of the input and output image
 * @tparam NPC number of pixels processed per cycle
 * @param _src_mat input image
 * @param _dst_mat output image
 */
template <int CCM_TYPE, int SRC_T, int DST_T, int ROWS, int COLS, int NPC = 1, int S_DEPTH = 2>
void colorcorrectionmatrix(xf::cv::Mat<SRC_T, ROWS, COLS, NPC, S_DEPTH>& _src_mat,
                           xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _dst_mat) {
    unsigned short width = _src_mat.cols >> XF_BITSHIFT(NPC);
    unsigned short height = _src_mat.rows;
    assert(((height <= ROWS) && (width <= COLS)) && "ROWS and COLS should be greater than input image");

// clang-format off
#pragma HLS INLINE OFF
    // clang-format on

    xfccmkernel<SRC_T, ROWS, COLS, XF_DEPTH(SRC_T, NPC), NPC, XF_WORDWIDTH(SRC_T, NPC), XF_WORDWIDTH(SRC_T, NPC),
                (COLS >> XF_BITSHIFT(NPC)), S_DEPTH>(_src_mat, _dst_mat, CCM_TYPE, height, width);
}
} // namespace cv
} // namespace xf

#endif
