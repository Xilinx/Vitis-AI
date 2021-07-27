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

#ifndef _XF_GAUSSIAN_HPP_
#define _XF_GAUSSIAN_HPP_

#ifndef __cplusplus
#error C++ is needed to include this header
#endif

#include "hls_stream.h"
#include "common/xf_common.hpp"
#include "common/xf_utility.hpp"

namespace xf {
namespace cv {

static void weightsghcalculation3x3(float sigma, unsigned char* weights) {
// clang-format off
    #pragma HLS INLINE
    // clang-format on

    float cf[3];

    float sum = 0;

    if (sigma <= 0) {
        sigma = 0.8;
    }

    int n = 3;

    float scale2X = -(1 / ((sigma * sigma) * 2));

    for (int i = 0; i < n; i++) {
        float x = i - ((n - 1) >> 1);
        float t = expf(scale2X * x * x);

        cf[i] = (float)t;
        sum += cf[i];
    }

    sum = 1. / sum;
    for (int i = 0; i < n; i++) {
        cf[i] = (float)(cf[i] * sum);
        weights[i] = ((cf[i] * 256) + 0.5);
    }
}

static void weightsghcalculation5x5(float sigma, unsigned char weights[5]) {
// clang-format off
    #pragma HLS INLINE
    // clang-format on

    float cf[5];

    float sum = 0;

    if (sigma <= 0) {
        sigma = 1.1f;
    }

    int n = 5;

    float scale2X = -(1 / ((sigma * sigma) * 2));

    for (int i = 0; i < n; i++) {
        float x = i - ((n - 1) >> 1);
        float t = expf(scale2X * x * x);

        cf[i] = (float)t;
        sum += cf[i];
    }

    sum = 1. / sum;
    for (int i = 0; i < n; i++) {
        cf[i] = (float)(cf[i] * sum);
        weights[i] = ((float)(cf[i] * 256) + 0.5);
    }
}

static void weightsghcalculation7x7(float sigma, unsigned char weights[7]) {
// clang-format off
    #pragma HLS INLINE
    // clang-format on

    float kval[7][7];

    float cf[7];

    float sum = 0;

    if (sigma <= 0) {
        sigma = 1.4f;
    }

    int n = 7;

    float scale2X = -(1 / ((sigma * sigma) * 2));

    for (int i = 0; i < n; i++) {
        float x = i - ((n - 1) >> 1);
        float t = expf(scale2X * x * x);

        cf[i] = (float)t;
        sum += cf[i];
    }

    sum = 1. / sum;
    for (int i = 0; i < n; i++) {
        cf[i] = (float)(cf[i] * sum);
        weights[i] = (unsigned char)(((float)cf[i] * 256) + 0.5);
    }
}

/////////////////////////////////////////weights///////////////////////////////////////////////////
template <int DEPTH>
XF_PTNAME(DEPTH)
xFapplygaussian3x3(XF_PTNAME(DEPTH) D1,
                   XF_PTNAME(DEPTH) D2,
                   XF_PTNAME(DEPTH) D3,
                   XF_PTNAME(DEPTH) D4,
                   XF_PTNAME(DEPTH) D5,
                   XF_PTNAME(DEPTH) D6,
                   XF_PTNAME(DEPTH) D7,
                   XF_PTNAME(DEPTH) D8,
                   XF_PTNAME(DEPTH) D9,
                   unsigned char* weights) {
// clang-format off
    #pragma HLS INLINE OFF
    // clang-format on
    XF_PTNAME(DEPTH) out_pix = 0;
    unsigned int sum = 0;

    ap_uint<18> sum1, sum2, sum3;

    sum2 = (D4 + D6) * weights[0] + D5 * weights[1];

    ap_uint<15> sumvalue0 = D1 + D3 + D7 + D9;

    ap_uint<15> sumvalue1 = D2 + D8;

    unsigned int value1 = sumvalue0 * weights[0] + sumvalue1 * weights[1];

    sum = (value1)*weights[0] + sum2 * weights[1];

    unsigned char val = (sum + 32768) >> 16;

    out_pix = (XF_PTNAME(DEPTH))val;

    return out_pix;
}

template <int PLANES, int DEPTH, bool FOR_IMAGE_PYRAMID>
XF_PTNAME(DEPTH)
xfapplygaussian5x5(XF_PTNAME(DEPTH) * src_buf1,
                   XF_PTNAME(DEPTH) * src_buf2,
                   XF_PTNAME(DEPTH) * src_buf3,
                   XF_PTNAME(DEPTH) * src_buf4,
                   XF_PTNAME(DEPTH) * src_buf5,
                   unsigned char weights[5]) {
// clang-format off
    #pragma HLS INLINE OFF
    // clang-format on
    unsigned int sum = 0.0, sumval = 0;
    unsigned char value = 0;

    XF_PTNAME(DEPTH) out_pix = 0;

    ap_uint<10> tmp[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    for (int i = 0, k = 0; i < PLANES; i++, k += 8) {
        tmp[0] = src_buf1[0].range(k + 7, k) + src_buf1[4].range(k + 7, k);
        tmp[1] = src_buf2[0].range(k + 7, k) + src_buf2[4].range(k + 7, k);
        tmp[2] = src_buf3[0].range(k + 7, k) + src_buf3[4].range(k + 7, k);
        tmp[3] = src_buf4[0].range(k + 7, k) + src_buf4[4].range(k + 7, k);
        tmp[4] = src_buf5[0].range(k + 7, k) + src_buf5[4].range(k + 7, k);

        tmp[5] = src_buf1[1].range(k + 7, k) + src_buf1[3].range(k + 7, k);
        tmp[6] = src_buf2[1].range(k + 7, k) + src_buf2[3].range(k + 7, k);
        tmp[7] = src_buf3[1].range(k + 7, k) + src_buf3[3].range(k + 7, k);
        tmp[8] = src_buf4[1].range(k + 7, k) + src_buf4[3].range(k + 7, k);
        tmp[9] = src_buf5[1].range(k + 7, k) + src_buf5[3].range(k + 7, k);

        ap_uint<24> tmp_sum[5] = {0, 0, 0, 0, 0};

        tmp_sum[0] = (ap_uint<24>)(tmp[0] + tmp[4]) * weights[0] + (ap_uint<24>)(tmp[5] + tmp[9]) * weights[1] +
                     (ap_uint<24>)(src_buf1[2].range(k + 7, k) + src_buf5[2].range(k + 7, k)) * weights[2];
        tmp_sum[1] = (ap_uint<24>)(tmp[1] + tmp[3]) * weights[0] + (ap_uint<24>)(tmp[6] + tmp[8]) * weights[1] +
                     (ap_uint<24>)(src_buf2[2].range(k + 7, k) + src_buf4[2].range(k + 7, k)) * weights[2];
        tmp_sum[2] = (ap_uint<24>)tmp[2] * weights[0] + (ap_uint<24>)tmp[7] * weights[1] +
                     (ap_uint<24>)src_buf3[2].range(k + 7, k) * weights[2];

        sumval = (unsigned int)tmp_sum[0] * weights[0] + tmp_sum[1] * weights[1] + tmp_sum[2] * weights[2];

        unsigned short val = ((sumval + 32768) >> 16);

        if (val >= 255) {
            value = 255;
        } else {
            value = val;
        }

        out_pix.range(k + 7, k) = (XF_PTNAME(DEPTH))value;
    }
    return out_pix;
}

template <int PLANES, int DEPTH>
XF_PTNAME(DEPTH)
xfapplygaussian7x7(XF_PTNAME(DEPTH) * src_buf1,
                   XF_PTNAME(DEPTH) * src_buf2,
                   XF_PTNAME(DEPTH) * src_buf3,
                   XF_PTNAME(DEPTH) * src_buf4,
                   XF_PTNAME(DEPTH) * src_buf5,
                   XF_PTNAME(DEPTH) * src_buf6,
                   XF_PTNAME(DEPTH) * src_buf7,
                   unsigned char weights[7]) {
// clang-format off
    #pragma HLS INLINE OFF
    // clang-format on

    XF_PTNAME(DEPTH) out_pix = 0;
    unsigned long long int sum_val = 0;
    unsigned short val = 0;
    for (int c = 0, k = 0; c < PLANES; c++, k += 8) {
        unsigned long long int sum_value = 0;
        unsigned int sum = 0.0, sum1 = 0.0, sum2 = 0.0, sum3 = 0.0, sum4 = 0.0, sum5 = 0.0, sum6 = 0.0;
        sum3 = (unsigned int)(src_buf4[0].range(k + 7, k) + src_buf4[6].range(k + 7, k)) * weights[0] +
               (src_buf4[1].range(k + 7, k) + src_buf4[5].range(k + 7, k)) * weights[1] +
               (src_buf4[2].range(k + 7, k) + src_buf4[4].range(k + 7, k)) * weights[2] +
               src_buf4[3].range(k + 7, k) * weights[3];
        sum = (unsigned int)(src_buf1[0].range(k + 7, k) + src_buf1[6].range(k + 7, k) + src_buf7[0].range(k + 7, k) +
                             src_buf7[6].range(k + 7, k)) *
                  weights[0] +
              (src_buf1[1].range(k + 7, k) + src_buf1[5].range(k + 7, k) + src_buf7[1].range(k + 7, k) +
               src_buf7[5].range(k + 7, k)) *
                  weights[1] +
              (src_buf1[2].range(k + 7, k) + src_buf1[4].range(k + 7, k) + src_buf7[2].range(k + 7, k) +
               src_buf7[4].range(k + 7, k)) *
                  weights[2] +
              (src_buf1[3].range(k + 7, k) + src_buf7[3].range(k + 7, k)) * weights[3];
        sum1 = (unsigned int)(src_buf2[0].range(k + 7, k) + src_buf2[6].range(k + 7, k) + src_buf6[0].range(k + 7, k) +
                              src_buf6[6].range(k + 7, k)) *
                   weights[0] +
               (src_buf2[1].range(k + 7, k) + src_buf2[5].range(k + 7, k) + src_buf6[1].range(k + 7, k) +
                src_buf6[5].range(k + 7, k)) *
                   weights[1] +
               (src_buf2[2].range(k + 7, k) + src_buf2[4].range(k + 7, k) + src_buf6[2].range(k + 7, k) +
                src_buf6[4].range(k + 7, k)) *
                   weights[2] +
               (src_buf2[3].range(k + 7, k) + src_buf6[3].range(k + 7, k)) * weights[3];
        sum2 = (unsigned int)(src_buf3[0].range(k + 7, k) + src_buf3[6].range(k + 7, k) + src_buf5[0].range(k + 7, k) +
                              src_buf5[6].range(k + 7, k)) *
                   weights[0] +
               (src_buf3[1].range(k + 7, k) + src_buf3[5].range(k + 7, k) + src_buf5[1].range(k + 7, k) +
                src_buf5[5].range(k + 7, k)) *
                   weights[1] +
               (src_buf3[2].range(k + 7, k) + src_buf3[4].range(k + 7, k) + src_buf5[2].range(k + 7, k) +
                src_buf5[4].range(k + 7, k)) *
                   weights[2] +
               (src_buf3[3].range(k + 7, k) + src_buf5[3].range(k + 7, k)) * weights[3];

        sum_value = (sum)*weights[0] + (sum1)*weights[1] + (sum2)*weights[2] + (sum3)*weights[3];

        unsigned short val = ((sum_value + 32768) >> 16);

        unsigned char value;

        if (val >= 255) {
            value = 255;
        } else {
            value = val;
        }

        out_pix.range(k + 7, k) = (uchar_t)value;
    }
    return out_pix;
}

template <int NPC, int DEPTH, int PLANES>
void auGaussian3x3(XF_PTNAME(DEPTH) * OutputValues,
                   XF_PTNAME(DEPTH) * src_buf1,
                   XF_PTNAME(DEPTH) * src_buf2,
                   XF_PTNAME(DEPTH) * src_buf3,
                   unsigned char weights[3]) {
// clang-format off
    #pragma HLS INLINE
    // clang-format on
    int p = 0;
    ap_uint<24> val;
Compute_Grad_Loop:
    for (ap_uint<5> j = 0; j < (XF_NPIXPERCYCLE(NPC)); j++) {
// clang-format off
        #pragma HLS UNROLL
        // clang-format on
        for (ap_uint<5> c = 0, k = 0; c < PLANES; c++, k += 8) {
// clang-format off
            #pragma HLS UNROLL
            // clang-format on

            val.range(k + 7, k) = xFapplygaussian3x3<DEPTH>(
                src_buf1[j].range(k + 7, k), src_buf1[j + 1].range(k + 7, k), src_buf1[j + 2].range(k + 7, k),
                src_buf2[j].range(k + 7, k), src_buf2[j + 1].range(k + 7, k), src_buf2[j + 2].range(k + 7, k),
                src_buf3[j].range(k + 7, k), src_buf3[j + 1].range(k + 7, k), src_buf3[j + 2].range(k + 7, k), weights);
        }
        OutputValues[p] = val;
        p++;
    }
}

template <int SRC_T, int ROWS, int COLS, int PLANES, int DEPTH, int NPC, int WORDWIDTH, int TC>
void ProcessGaussian3x3(xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _src_mat,
                        xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _out_mat,
                        XF_SNAME(WORDWIDTH) buf[3][(COLS >> XF_BITSHIFT(NPC))],
                        XF_PTNAME(DEPTH) src_buf1[XF_NPIXPERCYCLE(NPC) + 2],
                        XF_PTNAME(DEPTH) src_buf2[XF_NPIXPERCYCLE(NPC) + 2],
                        XF_PTNAME(DEPTH) src_buf3[XF_NPIXPERCYCLE(NPC) + 2],
                        XF_PTNAME(DEPTH) OutputValues[XF_NPIXPERCYCLE(NPC)],
                        XF_SNAME(WORDWIDTH) & P0,
                        uint16_t img_width,
                        uint16_t img_height,
                        uint16_t& shift_x,
                        ap_uint<2> tp,
                        ap_uint<2> mid,
                        ap_uint<2> bottom,
                        ap_uint<13> row,
                        unsigned char* weights,
                        int& read_index,
                        int& write_index) {
// clang-format off
    #pragma HLS INLINE
    // clang-format on

    XF_SNAME(WORDWIDTH) buf0, buf1, buf2;
    uint16_t npc = XF_NPIXPERCYCLE(NPC);
    ap_uint<5> buf_size = XF_NPIXPERCYCLE(NPC) + 2;

Col_Loop:
    for (ap_uint<13> col = 0; col < img_width; col++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=TC max=TC
        #pragma HLS pipeline
        // clang-format on
        if (row < img_height)
            buf[bottom][col] = _src_mat.read(read_index++); // Read data
        else
            buf[bottom][col] = 0;

        buf0 = buf[tp][col];
        buf1 = buf[mid][col];
        buf2 = buf[bottom][col];

        if (NPC == XF_NPPC8) {
            xfExtractPixels<NPC, WORDWIDTH, DEPTH>(&src_buf1[2], buf0, 0);
            xfExtractPixels<NPC, WORDWIDTH, DEPTH>(&src_buf2[2], buf1, 0);
            xfExtractPixels<NPC, WORDWIDTH, DEPTH>(&src_buf3[2], buf2, 0);

        }

        else {
            src_buf1[2] = buf0;
            src_buf2[2] = buf1;
            src_buf3[2] = buf2;
        }

        auGaussian3x3<NPC, DEPTH, PLANES>(OutputValues, src_buf1, src_buf2, src_buf3, weights);

        if (col == 0) {
            shift_x = 0;
            P0 = 0;
            if (NPC == XF_NPPC8) {
                xfPackPixels<NPC, WORDWIDTH, DEPTH>(&OutputValues[0], P0, 1, (npc - 1), shift_x);
            } else {
                P0 = OutputValues[0];
            }

        } else {
            if (NPC == XF_NPPC8) {
                xfPackPixels<NPC, WORDWIDTH, DEPTH>(&OutputValues[0], P0, 0, 1, shift_x);
            } else {
                P0 = OutputValues[0];
            }

            _out_mat.write(write_index++, P0);

            shift_x = 0;
            P0 = 0;
            if (NPC == XF_NPPC8) {
                xfPackPixels<NPC, WORDWIDTH, DEPTH>(&OutputValues[0], P0, 1, (npc - 1), shift_x);

            } else {
                P0 = OutputValues[0];
            }
        }

        src_buf1[0] = src_buf1[buf_size - 2];
        src_buf1[1] = src_buf1[buf_size - 1];
        src_buf2[0] = src_buf2[buf_size - 2];
        src_buf2[1] = src_buf2[buf_size - 1];
        src_buf3[0] = src_buf3[buf_size - 2];
        src_buf3[1] = src_buf3[buf_size - 1];
    } // Col_Loop
}

template <int SRC_T, int ROWS, int COLS, int PLANES, int DEPTH, int NPC, int WORDWIDTH, int TC>
void xfGaussianFilter3x3(xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _src_mat,
                         xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _out_mat,
                         uint16_t img_height,
                         uint16_t img_width,
                         unsigned char* weights) {
    ap_uint<13> row_ind;
    ap_uint<2> tp, mid, bottom;
    ap_uint<8> buf_size = XF_NPIXPERCYCLE(NPC) + 2;
    uint16_t shift_x = 0;
    ap_uint<13> row, col;
    int read_index = 0, write_index = 0;

    XF_PTNAME(DEPTH) OutputValues[XF_NPIXPERCYCLE(NPC)];

// clang-format off
    #pragma HLS ARRAY_PARTITION variable=OutputValues complete dim=1
    // clang-format on

    XF_PTNAME(DEPTH)
    src_buf1[XF_NPIXPERCYCLE(NPC) + 2] = {0}, src_buf2[XF_NPIXPERCYCLE(NPC) + 2] = {0},
                                    src_buf3[XF_NPIXPERCYCLE(NPC) + 2] = {0};
// clang-format off
    #pragma HLS ARRAY_PARTITION variable=src_buf1 complete dim=1
    #pragma HLS ARRAY_PARTITION variable=src_buf2 complete dim=1
    #pragma HLS ARRAY_PARTITION variable=src_buf3 complete dim=1
    // clang-format on

    XF_SNAME(WORDWIDTH) P0;

    XF_SNAME(WORDWIDTH) buf[3][(COLS >> XF_BITSHIFT(NPC))];
// clang-format off
    #pragma HLS RESOURCE variable=buf core=RAM_S2P_BRAM
    #pragma HLS ARRAY_PARTITION variable=buf complete dim=1
    // clang-format on
    row_ind = 1;

Clear_Row_Loop:
    for (col = 0; col < img_width; col++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=TC max=TC
        #pragma HLS pipeline
        // clang-format on
        //#pragma HLS LOOP_FLATTEN off
        buf[0][col] = 0;
        buf[row_ind][col] = _src_mat.read(read_index++); // data[read_index++];
    }
    row_ind++;

Row_Loop:
    for (row = 1; row < img_height + 1; row++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=ROWS max=ROWS
        // clang-format on
        if (row_ind == 2) {
            tp = 0;
            mid = 1;
            bottom = 2;
        } else if (row_ind == 0) {
            tp = 1;
            mid = 2;
            bottom = 0;
        } else if (row_ind == 1) {
            tp = 2;
            mid = 0;
            bottom = 1;
        }

        src_buf1[0] = src_buf1[1] = 0;
        src_buf2[0] = src_buf2[1] = 0;
        src_buf3[0] = src_buf3[1] = 0;

        P0 = 0;

        ProcessGaussian3x3<SRC_T, ROWS, COLS, PLANES, DEPTH, NPC, WORDWIDTH, TC>(
            _src_mat, _out_mat, buf, src_buf1, src_buf2, src_buf3, OutputValues, P0, img_width, img_height, shift_x, tp,
            mid, bottom, row, weights, read_index, write_index);

        if ((NPC == XF_NPPC8) || (NPC == XF_NPPC16)) {
            OutputValues[0] = xFapplygaussian3x3<DEPTH>(src_buf1[buf_size - 2], src_buf1[buf_size - 1], 0,
                                                        src_buf2[buf_size - 2], src_buf2[buf_size - 1], 0,
                                                        src_buf3[buf_size - 2], src_buf3[buf_size - 1], 0, weights);

        } else {
            ap_uint<24> out_val1;
            for (int i = 0, k = 0; i < PLANES; i++, k += 8) {
                ap_uint<8> srcbuf10 = src_buf1[buf_size - 3].range(k + 7, k);
                ap_uint<8> srcbuf11 = src_buf1[buf_size - 2].range(k + 7, k);
                ap_uint<8> srcbuf20 = src_buf2[buf_size - 3].range(k + 7, k);
                ap_uint<8> srcbuf21 = src_buf2[buf_size - 2].range(k + 7, k);
                ap_uint<8> srcbuf30 = src_buf3[buf_size - 3].range(k + 7, k);
                ap_uint<8> srcbuf31 = src_buf3[buf_size - 2].range(k + 7, k);
                out_val1.range(k + 7, k) = xFapplygaussian3x3<DEPTH>(srcbuf10, srcbuf11, 0, srcbuf20, srcbuf21, 0,
                                                                     srcbuf30, srcbuf31, 0, weights);
            }
            OutputValues[0] = out_val1;
        }

        if (NPC == XF_NPPC8) {
            xfPackPixels<NPC, WORDWIDTH, DEPTH>(&OutputValues[0], P0, 0, 1, shift_x);

        } else {
            P0 = OutputValues[0];
        }

        _out_mat.write(write_index++, P0); // data[write_index++] = (P0);

        shift_x = 0;
        P0 = 0;

        row_ind++;
        if (row_ind == 3) {
            row_ind = 0;
        }
    } // Row_Loop
}

template <int NPC, int DEPTH, int PLANES, bool FOR_IMAGE_PYRAMID>
void xFGaussian5x5(XF_PTNAME(DEPTH) * OutputValues,
                   XF_PTNAME(DEPTH) * src_buf1,
                   XF_PTNAME(DEPTH) * src_buf2,
                   XF_PTNAME(DEPTH) * src_buf3,
                   XF_PTNAME(DEPTH) * src_buf4,
                   XF_PTNAME(DEPTH) * src_buf5,
                   unsigned char weights[5]) {
// clang-format off
    #pragma HLS INLINE
    // clang-format on
    XF_PTNAME(DEPTH) val = 0, p = 0;
Compute_Grad_Loop:
    for (ap_uint<5> j = 0; j < XF_NPIXPERCYCLE(NPC); j++) {
        OutputValues[j] = xfapplygaussian5x5<PLANES, DEPTH, FOR_IMAGE_PYRAMID>(&src_buf1[j], &src_buf2[j], &src_buf3[j],
                                                                               &src_buf4[j], &src_buf5[j], weights);
    }
}

template <int SRC_T, int ROWS, int COLS, int PLANES, int DEPTH, int NPC, int WORDWIDTH, int TC, bool FOR_IMAGE_PYRAMID>
void ProcessGaussian5x5(xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _src_mat,
                        xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _out_mat,
                        XF_SNAME(WORDWIDTH) buf[5][(COLS >> XF_BITSHIFT(NPC))],
                        XF_PTNAME(DEPTH) src_buf1[XF_NPIXPERCYCLE(NPC) + 4],
                        XF_PTNAME(DEPTH) src_buf2[XF_NPIXPERCYCLE(NPC) + 4],
                        XF_PTNAME(DEPTH) src_buf3[XF_NPIXPERCYCLE(NPC) + 4],
                        XF_PTNAME(DEPTH) src_buf4[XF_NPIXPERCYCLE(NPC) + 4],
                        XF_PTNAME(DEPTH) src_buf5[XF_NPIXPERCYCLE(NPC) + 4],
                        XF_PTNAME(DEPTH) OutputValues[XF_NPIXPERCYCLE(NPC)],
                        XF_SNAME(WORDWIDTH) & inter_valx,
                        uint16_t img_width,
                        uint16_t img_height,
                        ap_uint<13> row_ind,
                        uint16_t& shift_x,
                        ap_uint<4> tp1,
                        ap_uint<4> tp2,
                        ap_uint<4> mid,
                        ap_uint<4> bottom1,
                        ap_uint<4> bottom2,
                        ap_uint<13> row,
                        unsigned char weights[5],
                        int& read_index,
                        int& write_index) {
// clang-format off
    #pragma HLS INLINE
    // clang-format on
    XF_SNAME(WORDWIDTH) buf0, buf1, buf2, buf3, buf4;
    ap_uint<8> buf_size = XF_NPIXPERCYCLE(NPC) + 4;
    uint16_t npc = XF_NPIXPERCYCLE(NPC);
    ap_uint<8> max_loop = XF_WORDDEPTH(WORDWIDTH);
    ap_uint<8> step = XF_PIXELDEPTH(DEPTH);

Col_Loop:
    for (ap_uint<13> col = 0; col < img_width; col++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=TC max=TC
        #pragma HLS pipeline
        // clang-format on
        if (row < img_height)
            buf[row_ind][col] = _src_mat.read(read_index++); //.data[read_index++];
        else
            buf[bottom2][col] = 0;

        buf0 = buf[tp1][col];
        buf1 = buf[tp2][col];
        buf2 = buf[mid][col];
        buf3 = buf[bottom1][col];
        buf4 = buf[bottom2][col];

        if (NPC == XF_NPPC8) {
            xfExtractPixels<NPC, WORDWIDTH, DEPTH>(&src_buf1[4], buf0, 0);
            xfExtractPixels<NPC, WORDWIDTH, DEPTH>(&src_buf2[4], buf1, 0);
            xfExtractPixels<NPC, WORDWIDTH, DEPTH>(&src_buf3[4], buf2, 0);
            xfExtractPixels<NPC, WORDWIDTH, DEPTH>(&src_buf4[4], buf3, 0);
            xfExtractPixels<NPC, WORDWIDTH, DEPTH>(&src_buf5[4], buf4, 0);
        } else {
            src_buf1[4] = buf0;
            src_buf2[4] = buf1;
            src_buf3[4] = buf2;
            src_buf4[4] = buf3;
            src_buf5[4] = buf4;
        }

        xFGaussian5x5<NPC, DEPTH, PLANES, FOR_IMAGE_PYRAMID>(OutputValues, src_buf1, src_buf2, src_buf3, src_buf4,
                                                             src_buf5, weights);

        for (ap_uint<4> i = 0; i < 4; i++) {
// clang-format off
            #pragma HLS unroll
            // clang-format on
            src_buf1[i] = src_buf1[buf_size - (4 - i)];
            src_buf2[i] = src_buf2[buf_size - (4 - i)];
            src_buf3[i] = src_buf3[buf_size - (4 - i)];
            src_buf4[i] = src_buf4[buf_size - (4 - i)];
            src_buf5[i] = src_buf5[buf_size - (4 - i)];
        }

        if (col == 0) {
            shift_x = 0;
            inter_valx = 0;

            xfPackPixels<NPC, WORDWIDTH, DEPTH>(&OutputValues[0], inter_valx, 2, (npc - 2), shift_x);

        } else {
            if (NPC == XF_NPPC8) {
                xfPackPixels<NPC, WORDWIDTH, DEPTH>(&OutputValues[0], inter_valx, 0, 2, shift_x);

                _out_mat.write(write_index++, inter_valx);

                shift_x = 0;
                inter_valx = 0;

                xfPackPixels<NPC, WORDWIDTH, DEPTH>(&OutputValues[0], inter_valx, 2, (npc - 2), shift_x);

            } else {
                if (col >= 2) {
                    if (PLANES == 1) {
                        inter_valx((max_loop - 1), (max_loop - 8)) = OutputValues[0];
                        _out_mat.write(write_index++, inter_valx);
                    } else {
                        //						_out_mat.data[write_index++] =
                        //(OutputValues[0]);
                        _out_mat.write(write_index++, OutputValues[0]);
                    }
                }
            }
        }
    } // Col_Loop
}

template <int SRC_T, int ROWS, int COLS, int PLANES, int DEPTH, int NPC, int WORDWIDTH, int TC, bool FOR_IMAGE_PYRAMID>
void xFGaussianFilter5x5(xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _src_mat,
                         xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _out_mat,
                         uint16_t img_height,
                         uint16_t img_width,
                         unsigned char weights[5]) {
    ap_uint<13> row_ind;
    ap_uint<13> row, col;
    ap_uint<4> tp1, tp2, mid, bottom1, bottom2;
    ap_uint<5> i;

    ap_uint<5> buf_size = XF_NPIXPERCYCLE(NPC) + 4;
    ap_uint<9> step = XF_PIXELDEPTH(DEPTH);
    ap_uint<9> max_loop = XF_WORDDEPTH(WORDWIDTH);
    uint16_t shift_x = 0;
    ap_uint<8> npc = XF_NPIXPERCYCLE(NPC);
    int read_index = 0, write_index = 0;

    XF_PTNAME(DEPTH) OutputValues[XF_NPIXPERCYCLE(NPC)];

// clang-format off
    #pragma HLS ARRAY_PARTITION variable=OutputValues complete dim=1
    // clang-format on

    XF_SNAME(WORDWIDTH) buf0, buf1, buf2, buf3, buf4;

    XF_PTNAME(DEPTH)
    src_buf1[XF_NPIXPERCYCLE(NPC) + 4], src_buf2[XF_NPIXPERCYCLE(NPC) + 4], src_buf3[XF_NPIXPERCYCLE(NPC) + 4],
        src_buf4[XF_NPIXPERCYCLE(NPC) + 4], src_buf5[XF_NPIXPERCYCLE(NPC) + 4];
// clang-format off
    #pragma HLS ARRAY_PARTITION variable=src_buf1 complete dim=1
    #pragma HLS ARRAY_PARTITION variable=src_buf2 complete dim=1
    #pragma HLS ARRAY_PARTITION variable=src_buf3 complete dim=1
    #pragma HLS ARRAY_PARTITION variable=src_buf4 complete dim=1
    #pragma HLS ARRAY_PARTITION variable=src_buf5 complete dim=1
    // clang-format on

    XF_SNAME(WORDWIDTH) tmp_in;
    XF_SNAME(WORDWIDTH) inter_valx = 0;

    XF_SNAME(WORDWIDTH) buf[5][(COLS >> XF_BITSHIFT(NPC))];
// clang-format off
    #pragma HLS RESOURCE variable=buf core=RAM_S2P_BRAM
    #pragma HLS ARRAY_PARTITION variable=buf complete dim=1
    // clang-format on

    row_ind = 2;

Clear_Row_Loop:
    for (col = 0; col < img_width; col++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=TC max=TC
        #pragma HLS pipeline
        // clang-format on
        buf[0][col] = 0;
        buf[1][col] = 0;
        buf[row_ind][col] = _src_mat.read(read_index++); //.data[read_index++];
    }

    row_ind++;

Read_Row2_Loop:
    for (col = 0; col < img_width; col++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=TC max=TC
        #pragma HLS pipeline
        // clang-format on

        buf[row_ind][col] = _src_mat.read(read_index++); //_src_mat.data[read_index++];
    }
    row_ind++;

Row_Loop:
    for (row = 2; row < img_height + 2; row++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=ROWS max=ROWS
        // clang-format on

        // modify the buffer indices to re use
        if (row_ind == 4) {
            tp1 = 0;
            tp2 = 1;
            mid = 2;
            bottom1 = 3;
            bottom2 = 4;
        } else if (row_ind == 0) {
            tp1 = 1;
            tp2 = 2;
            mid = 3;
            bottom1 = 4;
            bottom2 = 0;
        } else if (row_ind == 1) {
            tp1 = 2;
            tp2 = 3;
            mid = 4;
            bottom1 = 0;
            bottom2 = 1;
        } else if (row_ind == 2) {
            tp1 = 3;
            tp2 = 4;
            mid = 0;
            bottom1 = 1;
            bottom2 = 2;
        } else if (row_ind == 3) {
            tp1 = 4;
            tp2 = 0;
            mid = 1;
            bottom1 = 2;
            bottom2 = 3;
        }

        src_buf1[0] = src_buf1[1] = src_buf1[2] = src_buf1[3] = 0;
        src_buf2[0] = src_buf2[1] = src_buf2[2] = src_buf2[3] = 0;
        src_buf3[0] = src_buf3[1] = src_buf3[2] = src_buf3[3] = 0;
        src_buf4[0] = src_buf4[1] = src_buf4[2] = src_buf4[3] = 0;
        src_buf5[0] = src_buf5[1] = src_buf5[2] = src_buf5[3] = 0;

        inter_valx = 0;

        ProcessGaussian5x5<SRC_T, ROWS, COLS, PLANES, DEPTH, NPC, WORDWIDTH, TC, FOR_IMAGE_PYRAMID>(
            _src_mat, _out_mat, buf, src_buf1, src_buf2, src_buf3, src_buf4, src_buf5, OutputValues, inter_valx,
            img_width, img_height, row_ind, shift_x, tp1, tp2, mid, bottom1, bottom2, row, weights, read_index,
            write_index);

        if ((NPC == XF_NPPC8) || (NPC == XF_NPPC16)) {
            for (ap_uint<6> i = 4; i < (XF_NPIXPERCYCLE(NPC) + 4); i++) {
                src_buf1[i] = 0;
                src_buf2[i] = 0;
                src_buf3[i] = 0;
                src_buf4[i] = 0;
                src_buf5[i] = 0;
            }
            OutputValues[0] = xfapplygaussian5x5<PLANES, DEPTH, FOR_IMAGE_PYRAMID>(
                &src_buf1[0], &src_buf2[0], &src_buf3[0], &src_buf4[0], &src_buf5[0], weights);

            OutputValues[1] = xfapplygaussian5x5<PLANES, DEPTH, FOR_IMAGE_PYRAMID>(
                &src_buf1[1], &src_buf2[1], &src_buf3[1], &src_buf4[1], &src_buf5[1], weights);

            xfPackPixels<NPC, WORDWIDTH, DEPTH>(&OutputValues[0], inter_valx, 0, 2, shift_x);

            //_out_mat.data[write_index++] = (inter_valx);
            _out_mat.write(write_index++, inter_valx);

        } else {
// clang-format off
            #pragma HLS ALLOCATION function instances=xfapplygaussian5x5<PLANES, DEPTH, FOR_IMAGE_PYRAMID> limit=1
            // clang-format on
            src_buf1[buf_size - 1] = 0;
            src_buf2[buf_size - 1] = 0;
            src_buf3[buf_size - 1] = 0;
            src_buf4[buf_size - 1] = 0;
            src_buf5[buf_size - 1] = 0;
            OutputValues[0] = xfapplygaussian5x5<PLANES, DEPTH, FOR_IMAGE_PYRAMID>(
                &src_buf1[0], &src_buf2[0], &src_buf3[0], &src_buf4[0], &src_buf5[0], weights);

            inter_valx((max_loop - 1), (max_loop - step)) = OutputValues[0];

            _out_mat.write(write_index++, inter_valx);
            for (ap_uint<4> i = 0; i < 4; i++) {
// clang-format off
                #pragma HLS unroll
                // clang-format on
                src_buf1[i] = src_buf1[buf_size - (4 - i)];
                src_buf2[i] = src_buf2[buf_size - (4 - i)];
                src_buf3[i] = src_buf3[buf_size - (4 - i)];
                src_buf4[i] = src_buf4[buf_size - (4 - i)];
                src_buf5[i] = src_buf5[buf_size - (4 - i)];
            }

            OutputValues[0] = xfapplygaussian5x5<PLANES, DEPTH, FOR_IMAGE_PYRAMID>(
                &src_buf1[0], &src_buf2[0], &src_buf3[0], &src_buf4[0], &src_buf5[0], weights);

            inter_valx((max_loop - 1), (max_loop - step)) = OutputValues[0];
            _out_mat.write(write_index++, inter_valx);
        }
        row_ind++;

        if (row_ind == 5) {
            row_ind = 0;
        }
    } // Row_Loop
}

template <int NPC, int DEPTH, int PLANES>
void xFGaussian7x7(XF_PTNAME(DEPTH) * OutputValues,
                   XF_PTNAME(DEPTH) * src_buf1,
                   XF_PTNAME(DEPTH) * src_buf2,
                   XF_PTNAME(DEPTH) * src_buf3,
                   XF_PTNAME(DEPTH) * src_buf4,
                   XF_PTNAME(DEPTH) * src_buf5,
                   XF_PTNAME(DEPTH) * src_buf6,
                   XF_PTNAME(DEPTH) * src_buf7,
                   unsigned char weights[7]) {
// clang-format off
    #pragma HLS INLINE
    // clang-format on
    for (ap_uint<9> j = 0; j < XF_NPIXPERCYCLE(NPC); j++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=8 max=8
        #pragma HLS UNROLL
        // clang-format on
        //		for(ap_uint<8> c=0,k=0;c<PLANES;c++,k+=8)
        //		{
        OutputValues[j] = xfapplygaussian7x7<PLANES, DEPTH>(&src_buf1[j], &src_buf2[j], &src_buf3[j], &src_buf4[j],
                                                            &src_buf5[j], &src_buf6[j], &src_buf7[j], weights);
        //		}
    }
}

template <int SRC_T, int ROWS, int COLS, int PLANES, int DEPTH, int NPC, int WORDWIDTH, int TC>
void ProcessGaussian7x7(xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _src_mat,
                        xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _out_mat,
                        XF_SNAME(WORDWIDTH) buf[7][(COLS >> XF_BITSHIFT(NPC))],
                        XF_PTNAME(DEPTH) src_buf1[XF_NPIXPERCYCLE(NPC) + 6],
                        XF_PTNAME(DEPTH) src_buf2[XF_NPIXPERCYCLE(NPC) + 6],
                        XF_PTNAME(DEPTH) src_buf3[XF_NPIXPERCYCLE(NPC) + 6],
                        XF_PTNAME(DEPTH) src_buf4[XF_NPIXPERCYCLE(NPC) + 6],
                        XF_PTNAME(DEPTH) src_buf5[XF_NPIXPERCYCLE(NPC) + 6],
                        XF_PTNAME(DEPTH) src_buf6[XF_NPIXPERCYCLE(NPC) + 6],
                        XF_PTNAME(DEPTH) src_buf7[XF_NPIXPERCYCLE(NPC) + 6],
                        XF_PTNAME(DEPTH) OutputValues[XF_NPIXPERCYCLE(NPC)],
                        XF_SNAME(WORDWIDTH) & inter_valx,
                        uint16_t img_width,
                        uint16_t img_height,
                        uint16_t& shiftx,
                        ap_uint<4> tp1,
                        ap_uint<4> tp2,
                        ap_uint<4> tp3,
                        ap_uint<4> mid,
                        ap_uint<4> bottom1,
                        ap_uint<4> bottom2,
                        ap_uint<4> bottom3,
                        ap_uint<13> row_index,
                        unsigned char weights[7],
                        int& read_index,
                        int& write_index) {
// clang-format off
    #pragma HLS INLINE
    // clang-format on
    XF_SNAME(WORDWIDTH) buf0, buf1, buf2, buf3, buf4, buf5, buf6;
    uint16_t npc = XF_NPIXPERCYCLE(NPC);
    ap_uint<10> max_loop = XF_WORDDEPTH(WORDWIDTH);

Col_Loop:
    for (ap_uint<13> col = 0; col < img_width; col++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=TC max=TC
        #pragma HLS pipeline
        // clang-format on
        if (row_index < img_height)
            buf[bottom3][col] = _src_mat.read(read_index++); //_src_mat.data[read_index++];
        else
            buf[bottom3][col] = 0;

        buf0 = buf[tp1][col];
        buf1 = buf[tp2][col];
        buf2 = buf[tp3][col];
        buf3 = buf[mid][col];
        buf4 = buf[bottom1][col];
        buf5 = buf[bottom2][col];
        buf6 = buf[bottom3][col];

        if (NPC == XF_NPPC8) {
            xfExtractData<NPC, WORDWIDTH, DEPTH>(src_buf1, src_buf2, src_buf3, src_buf4, src_buf5, src_buf6, src_buf7,
                                                 buf0, buf1, buf2, buf3, buf4, buf5, buf6);
        } else {
            src_buf1[6] = buf0;
            src_buf2[6] = buf1;
            src_buf3[6] = buf2;
            src_buf4[6] = buf3;
            src_buf5[6] = buf4;
            src_buf6[6] = buf5;
            src_buf7[6] = buf6;
        }

        xFGaussian7x7<NPC, DEPTH, PLANES>(OutputValues, src_buf1, src_buf2, src_buf3, src_buf4, src_buf5, src_buf6,
                                          src_buf7, weights);

        xfCopyData<NPC, DEPTH>(src_buf1, src_buf2, src_buf3, src_buf4, src_buf5, src_buf6, src_buf7);

        if (col == 0) {
            shiftx = 0;

            inter_valx = 0;

            xfPackPixels<NPC, WORDWIDTH, DEPTH>(&OutputValues[0], inter_valx, 3, (npc - 3), shiftx);

        } else {
            if ((NPC == XF_NPPC8) || (NPC == XF_NPPC16)) {
                xfPackPixels<NPC, WORDWIDTH, DEPTH>(&OutputValues[0], inter_valx, 0, 3, shiftx);

                //				_out_mat.data[write_index++] = (inter_valx);
                _out_mat.write(write_index++, inter_valx);

                shiftx = 0;

                inter_valx = 0;

                xfPackPixels<NPC, WORDWIDTH, DEPTH>(&OutputValues[0], inter_valx, 3, (npc - 3), shiftx);

            } else {
                if (col >= 3) {
                    if (PLANES == 1) {
                        inter_valx((max_loop - 1), (max_loop - 8)) = OutputValues[0];
                        //_out_mat.data[write_index++] = (inter_valx);
                        _out_mat.write(write_index++, inter_valx);
                    } else {
                        //						_out_mat.data[write_index++] =
                        //(OutputValues[0]);
                        _out_mat.write(write_index++, OutputValues[0]);
                    }
                }
            }
        }
    } // Col_Loop
}

template <int SRC_T, int ROWS, int COLS, int PLANES, int DEPTH, int NPC, int WORDWIDTH, int TC>
void xFGaussianFilter7x7(xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _src_mat,
                         xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _out_mat,
                         uint16_t img_height,
                         uint16_t img_width,
                         unsigned char weights[7]) {
    ap_uint<13> row_ind, row, col;
    ap_uint<4> tp1, tp2, tp3, mid, bottom1, bottom2, bottom3;
    ap_uint<5> i;
    ap_uint<8> buf_size = (XF_NPIXPERCYCLE(NPC) + 6);
    ap_uint<10> max_loop = XF_WORDDEPTH(WORDWIDTH);
    int read_index = 0, write_index = 0;

    XF_PTNAME(DEPTH) OutputValues[XF_NPIXPERCYCLE(NPC)];

// clang-format off
    #pragma HLS ARRAY_PARTITION variable=OutputValues complete dim=1
    // clang-format on

    // Temporary buffers to hold image data from three rows.
    XF_PTNAME(DEPTH)
    src_buf1[XF_NPIXPERCYCLE(NPC) + 6], src_buf2[XF_NPIXPERCYCLE(NPC) + 6], src_buf3[XF_NPIXPERCYCLE(NPC) + 6],
        src_buf4[XF_NPIXPERCYCLE(NPC) + 6], src_buf5[XF_NPIXPERCYCLE(NPC) + 6];
    XF_PTNAME(DEPTH) src_buf6[XF_NPIXPERCYCLE(NPC) + 6], src_buf7[XF_NPIXPERCYCLE(NPC) + 6];
// clang-format off
    #pragma HLS ARRAY_PARTITION variable=src_buf1 complete dim=1
    #pragma HLS ARRAY_PARTITION variable=src_buf2 complete dim=1
    #pragma HLS ARRAY_PARTITION variable=src_buf3 complete dim=1
    #pragma HLS ARRAY_PARTITION variable=src_buf4 complete dim=1
    #pragma HLS ARRAY_PARTITION variable=src_buf5 complete dim=1
    #pragma HLS ARRAY_PARTITION variable=src_buf6 complete dim=1
    #pragma HLS ARRAY_PARTITION variable=src_buf7 complete dim=1
    // clang-format on

    XF_SNAME(WORDWIDTH) inter_valx = 0;
    uint16_t shiftx = 0;

    XF_SNAME(WORDWIDTH) buf[7][(COLS >> XF_BITSHIFT(NPC))];
// clang-format off
    #pragma HLS RESOURCE variable=buf core=RAM_S2P_BRAM
    #pragma HLS ARRAY_PARTITION variable=buf complete dim=1
    // clang-format on

    row_ind = 3;
Clear_Row_Loop:
    for (col = 0; col < img_width; col++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=TC max=TC
        #pragma HLS pipeline
        // clang-format on
        buf[0][col] = 0;
        buf[1][col] = 0;
        buf[2][col] = 0;
        buf[row_ind][col] = _src_mat.read(read_index++); // data[read_index++];
    }
    row_ind++;

Read_Row1_Loop:
    for (col = 0; col < img_width; col++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=TC max=TC
        #pragma HLS pipeline
        // clang-format on

        buf[row_ind][col] = _src_mat.read(read_index++); //_src_mat.data[read_index++];
    }
    row_ind++;

Read_Row2_Loop:
    for (col = 0; col < img_width; col++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=TC max=TC
        #pragma HLS pipeline
        // clang-format on

        buf[row_ind][col] = _src_mat.read(read_index++); //_src_mat.data[read_index++];
    }
    row_ind++;

Row_Loop:
    for (row = 3; row < img_height + 3; row++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=ROWS max=ROWS
        // clang-format on
        // modify the buffer indices to re use
        if (row_ind == 0) {
            tp1 = 1;
            tp2 = 2;
            tp3 = 3;
            mid = 4;
            bottom1 = 5;
            bottom2 = 6;
            bottom3 = 0;
        } else if (row_ind == 1) {
            tp1 = 2;
            tp2 = 3;
            tp3 = 4;
            mid = 5;
            bottom1 = 6;
            bottom2 = 0;
            bottom3 = 1;
        } else if (row_ind == 2) {
            tp1 = 3;
            tp2 = 4;
            tp3 = 5;
            mid = 6;
            bottom1 = 0;
            bottom2 = 1;
            bottom3 = 2;
        } else if (row_ind == 3) {
            tp1 = 4;
            tp2 = 5;
            tp3 = 6;
            mid = 0;
            bottom1 = 1;
            bottom2 = 2;
            bottom3 = 3;
        } else if (row_ind == 4) {
            tp1 = 5;
            tp2 = 6;
            tp3 = 0;
            mid = 1;
            bottom1 = 2;
            bottom2 = 3;
            bottom3 = 4;
        } else if (row_ind == 5) {
            tp1 = 6;
            tp2 = 0;
            tp3 = 1;
            mid = 2;
            bottom1 = 3;
            bottom2 = 4;
            bottom3 = 5;
        } else if (row_ind == 6) {
            tp1 = 0;
            tp2 = 1;
            tp3 = 2;
            mid = 3;
            bottom1 = 4;
            bottom2 = 5;
            bottom3 = 6;
        }

        for (i = 0; i < 6; i++) {
// clang-format off
            #pragma HLS unroll
            // clang-format on
            src_buf1[i] = 0;
            src_buf2[i] = 0;
            src_buf3[i] = 0;
            src_buf4[i] = 0;
            src_buf5[i] = 0;
            src_buf6[i] = 0;
            src_buf7[i] = 0;
        }
        inter_valx = 0;
        ProcessGaussian7x7<SRC_T, ROWS, COLS, PLANES, DEPTH, NPC, WORDWIDTH, TC>(
            _src_mat, _out_mat, buf, src_buf1, src_buf2, src_buf3, src_buf4, src_buf5, src_buf6, src_buf7, OutputValues,
            inter_valx, img_width, img_height, shiftx, tp1, tp2, tp3, mid, bottom1, bottom2, bottom3, row, weights,
            read_index, write_index);

        if ((NPC == XF_NPPC8) || (NPC == XF_NPPC16)) {
            for (i = 0; i < 8; i++) {
// clang-format off
                #pragma HLS LOOP_TRIPCOUNT min=8 max=8
                #pragma HLS unroll
                // clang-format on
                src_buf1[buf_size + i - (XF_NPIXPERCYCLE(NPC))] = 0;
                src_buf2[buf_size + i - (XF_NPIXPERCYCLE(NPC))] = 0;
                src_buf3[buf_size + i - (XF_NPIXPERCYCLE(NPC))] = 0;
                src_buf4[buf_size + i - (XF_NPIXPERCYCLE(NPC))] = 0;
                src_buf5[buf_size + i - (XF_NPIXPERCYCLE(NPC))] = 0;
                src_buf6[buf_size + i - (XF_NPIXPERCYCLE(NPC))] = 0;
                src_buf7[buf_size + i - (XF_NPIXPERCYCLE(NPC))] = 0;
            }

            for (i = 0; i < 3; i++) {
// clang-format off
                #pragma HLS LOOP_TRIPCOUNT min=3 max=3
                #pragma HLS unroll
                // clang-format on

                OutputValues[i] =
                    xfapplygaussian7x7<PLANES, DEPTH>(&src_buf1[i], &src_buf2[i], &src_buf3[i], &src_buf4[i],
                                                      &src_buf5[i], &src_buf6[i], &src_buf7[i], weights);
            }
            xfPackPixels<NPC, WORDWIDTH, DEPTH>(&OutputValues[0], inter_valx, 0, 3, shiftx);

            //_out_mat.data[write_index++] = (inter_valx);
            _out_mat.write(write_index++, inter_valx);

            shiftx = 0;
            inter_valx = 0;
        } else {
            src_buf1[6] = 0;
            src_buf2[6] = 0;
            src_buf3[6] = 0;
            src_buf4[6] = 0;
            src_buf5[6] = 0;
            src_buf6[6] = 0;
            src_buf7[6] = 0;

            for (i = 0; i < 3; i++) {
// clang-format off
                #pragma HLS LOOP_TRIPCOUNT min=3 max=3
                #pragma HLS unroll
                #pragma HLS ALLOCATION function instances=xfapplygaussian7x7<PLANES, DEPTH> limit=1
                // clang-format on

                OutputValues[0] =
                    xfapplygaussian7x7<PLANES, DEPTH>(&src_buf1[0], &src_buf2[0], &src_buf3[0], &src_buf4[0],
                                                      &src_buf5[0], &src_buf6[0], &src_buf7[0], weights);

                xfCopyData<NPC, DEPTH>(src_buf1, src_buf2, src_buf3, src_buf4, src_buf5, src_buf6, src_buf7);
                if (PLANES == 1) {
                    inter_valx((max_loop - 1), (max_loop - 8)) = OutputValues[0];

                    //				_out_mat.data[write_index++] = (inter_valx);
                    _out_mat.write(write_index++, inter_valx);
                } else {
                    //					_out_mat.data[write_index++] = ( OutputValues[0]);
                    _out_mat.write(write_index++, OutputValues[0]);
                }
            }
        }
        row_ind++;
        if (row_ind == 7) {
            row_ind = 0;
        }
    } // Row_Loop ends here

} /*
 template<int ROWS, int COLS,int PLANES, int DEPTH, int NPC, int WORDWIDTH>
 void xFGaussianFilter(hls::stream< XF_SNAME(WORDWIDTH)> &_src, hls::stream< XF_SNAME(WORDWIDTH) > &_dst, int
 _filter_width, int _border_type, uint16_t imgheight, uint16_t imgwidth, float sigma)
 {



 }*/

template <int FILTER_SIZE, int BORDER_TYPE, int SRC_T, int ROWS, int COLS, int NPC = 1>
void GaussianBlur(xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _src, xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _dst, float sigma) {
// clang-format off
    #pragma HLS inline off
    // clang-format on

    int imgwidth = _src.cols >> XF_BITSHIFT(NPC);

    if (FILTER_SIZE == XF_FILTER_3X3) {
        unsigned char weights[3];
// clang-format off
        #pragma HLS ARRAY_PARTITION variable=weights complete dim=1
        // clang-format on
        weightsghcalculation3x3(sigma, weights);
        xfGaussianFilter3x3<SRC_T, ROWS, COLS, XF_CHANNELS(SRC_T, NPC), XF_DEPTH(SRC_T, NPC), NPC,
                            XF_WORDWIDTH(SRC_T, NPC), (COLS >> XF_BITSHIFT(NPC))>(_src, _dst, _src.rows, imgwidth,
                                                                                  weights);
    } else if (FILTER_SIZE == XF_FILTER_5X5) {
        unsigned char weights[5];
// clang-format off
        #pragma HLS ARRAY_PARTITION variable=weights complete dim=1
        // clang-format on
        weightsghcalculation5x5(sigma, weights);
        xFGaussianFilter5x5<SRC_T, ROWS, COLS, XF_CHANNELS(SRC_T, NPC), XF_DEPTH(SRC_T, NPC), NPC,
                            XF_WORDWIDTH(SRC_T, NPC), (COLS >> XF_BITSHIFT(NPC)), false>(_src, _dst, _src.rows,
                                                                                         imgwidth, weights);
    } else if (FILTER_SIZE == XF_FILTER_7X7) {
        unsigned char weights[7];
// clang-format off
        #pragma HLS ARRAY_PARTITION variable=weights complete dim=1
        // clang-format on
        weightsghcalculation7x7(sigma, weights);
        xFGaussianFilter7x7<SRC_T, ROWS, COLS, XF_CHANNELS(SRC_T, NPC), XF_DEPTH(SRC_T, NPC), NPC,
                            XF_WORDWIDTH(SRC_T, NPC), (COLS >> XF_BITSHIFT(NPC))>(_src, _dst, _src.rows, imgwidth,
                                                                                  weights);
    }
}
} // namespace cv
} // namespace xf
#endif //_XF_GAUSSIAN_HPP_
