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

#ifndef _XF_AWB_HPP_
#define _XF_AWB_HPP_

#include "hls_stream.h"
#include "common/xf_common.hpp"

#ifndef XF_IN_STEP
#define XF_IN_STEP 8
#endif
#ifndef XF_OUT_STEP
#define XF_OUT_STEP 8
#endif

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

namespace xf {
namespace cv {

template <int SRC_T,
          int DST_T,
          int ROWS,
          int COLS,
          int NPC,
          int PLANES,
          int DEPTH_SRC,
          int DEPTH_DST,
          int WORDWIDTH_SRC,
          int WORDWIDTH_DST,
          int TC>
void AWBGainUpdateKernel(xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& src1,
                         xf::cv::Mat<DST_T, ROWS, COLS, NPC>& dst,
                         float thresh,
                         int i_gain[3]) {
    int width = src1.cols >> XF_BITSHIFT(NPC);
    int height = src1.rows;

    XF_TNAME(SRC_T, NPC) in_pix, out_pix;

    for (int i = 0; i < height; i++) {
// clang-format off
#pragma HLS LOOP_TRIPCOUNT min=ROWS max=ROWS
#pragma HLS LOOP_FLATTEN OFF
    // clang-format on
    ColLoop1:
        for (int j = 0; j < width; j++) {
// clang-format off
#pragma HLS LOOP_TRIPCOUNT min=TC max=TC
#pragma HLS pipeline
            // clang-format on
            in_pix = src1.read(i * width + j);

            for (int p = 0; p < XF_NPIXPERCYCLE(NPC) * PLANES; p++) {
// clang-format off
#pragma HLS unroll
                // clang-format on
                ap_uint<8> val = in_pix.range(p * 8 + 7, p * 8);
                ap_uint<8> outval = (unsigned char)((val * i_gain[p % 3]) >> 8);

                out_pix.range(p * 8 + 7, p * 8) = outval;
            }

            dst.write(i * width + j, out_pix);
        }
    }
}

template <int SRC_T,
          int DST_T,
          int ROWS,
          int COLS,
          int NPC,
          int PLANES,
          int DEPTH_SRC,
          int DEPTH_DST,
          int WORDWIDTH_SRC,
          int WORDWIDTH_DST,
          int TC>
void AWBChannelGainKernel(xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& src1,
                          xf::cv::Mat<DST_T, ROWS, COLS, NPC>& dst,
                          float thresh,
                          int i_gain[3]) {
    int width = src1.cols >> XF_BITSHIFT(NPC);
    int height = src1.rows;
    ap_uint<13> i = 0, j = 0;

    XF_TNAME(SRC_T, NPC) in_pix, out_pix;
    ap_uint<8> r, g, b, b1 = 0, g1 = 0, r1 = 0;

    XF_SNAME(WORDWIDTH_DST) pxl_pack_out;
    XF_SNAME(WORDWIDTH_SRC) pxl_pack1, pxl_pack2;

    int thresh255 = int(thresh * 255);

    int minRGB, maxRGB;

    ap_ufixed<32, 32> tmpsum_vals[(1 << XF_BITSHIFT(NPC)) * PLANES];

    ap_ufixed<32, 32> sum[PLANES];
// clang-format off
#pragma HLS ARRAY_PARTITION variable=tmpsum_vals complete dim=0
#pragma HLS ARRAY_PARTITION variable=sum complete dim=0
    // clang-format on

    for (j = 0; j < ((1 << XF_BITSHIFT(NPC)) * PLANES); j++) {
// clang-format off
#pragma HLS UNROLL
        // clang-format on
        tmpsum_vals[j] = 0;
    }
    for (j = 0; j < PLANES; j++) {
// clang-format off
#pragma HLS UNROLL
        // clang-format on
        sum[j] = 0;
    }

    int p = 0, read_index = 0;

Row_Loop:
    for (i = 0; i < height; i++) {
// clang-format off
#pragma HLS LOOP_TRIPCOUNT min=ROWS max=ROWS
    // clang-format on
    Col_Loop:
        for (j = 0; j < (width); j++) {
// clang-format off
#pragma HLS LOOP_TRIPCOUNT min=COLS/NPC max=COLS/NPC
#pragma HLS pipeline II=1
#pragma HLS LOOP_FLATTEN OFF
            // clang-format on
            XF_TNAME(SRC_T, NPC) in_buf;
            in_buf = src1.read(i * width + j);

            dst.write(i * width + j, in_buf);

        PLANES_LOOP:
            for (int p = 0; p < XF_NPIXPERCYCLE(NPC) * PLANES; p = p + PLANES) {
// clang-format off
#pragma HLS unroll
                // clang-format on
                ap_uint<8> val1 = in_buf.range(p * 8 + 7, p * 8);
                ap_uint<8> val2 = in_buf.range(p * 8 + 15, p * 8 + 8);
                ap_uint<8> val3 = in_buf.range(p * 8 + 23, p * 8 + 16);

                minRGB = MIN(val1, MIN(val2, val3));
                maxRGB = MAX(val1, MAX(val2, val3));

                if ((maxRGB - minRGB) * 255 > thresh255 * maxRGB) continue;

                tmpsum_vals[p] = tmpsum_vals[p] + val1;
                tmpsum_vals[(p) + 1] = tmpsum_vals[(p) + 1] + val2;
                tmpsum_vals[(p) + 2] = tmpsum_vals[(p) + 2] + val3;
            }
        }
    }

    for (int c = 0; c < PLANES; c++) {
        for (j = 0; j < (1 << XF_BITSHIFT(NPC)); j++) {
// clang-format off
#pragma HLS UNROLL
            // clang-format on
            sum[c] = (sum[c] + tmpsum_vals[j * PLANES + c]);
        }
    }

    ap_ufixed<32, 32> max_sum_fixed = MAX(sum[0], MAX(sum[1], sum[2]));

    ap_ufixed<32, 2> bval = (float)0.1;
    ap_ufixed<32, 32> zero = 0;

    ap_ufixed<40, 32> dinB1;
    ap_ufixed<40, 32> dinG1;
    ap_ufixed<40, 32> dinR1;

    if (sum[0] < bval) {
        dinB1 = 0;
    } else {
        dinB1 = (ap_ufixed<40, 32>)((ap_ufixed<40, 32>)max_sum_fixed / sum[0]);
    }

    if (sum[1] < bval) {
        dinG1 = 0;
    } else {
        dinG1 = (ap_ufixed<40, 32>)((ap_ufixed<40, 32>)max_sum_fixed / sum[1]);
    }
    if (sum[2] < bval) {
        dinR1 = 0;
    } else {
        dinR1 = (ap_ufixed<40, 32>)((ap_ufixed<40, 32>)max_sum_fixed / sum[2]);
    }

    ap_ufixed<40, 32> gain_max1 = MAX(dinB1, MAX(dinG1, dinR1));

    if (gain_max1 > 0) {
        dinB1 /= gain_max1;
        dinG1 /= gain_max1;
        dinR1 /= gain_max1;
    }

    float a1 = dinB1;
    float a2 = dinG1;
    float a3 = dinR1;

    // int i_gain[3] = {0, 0, 0};
    i_gain[0] = (dinB1 * (1 << 8));
    i_gain[1] = (dinG1 * (1 << 8));
    i_gain[2] = (dinR1 * (1 << 8));
}

template <int SRC_T, int DST_T, int ROWS, int COLS, int NPC = 1, int WB_TYPE>
void AWBNormalizationkernel(xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& src,
                            xf::cv::Mat<DST_T, ROWS, COLS, NPC>& dst,
                            uint32_t hist[3][256],
                            float p,
                            float inputMin,
                            float inputMax,
                            float outputMin,
                            float outputMax) {
// clang-format off
#pragma HLS INLINE OFF
    // clang-format on
    short width = dst.cols >> XF_BITSHIFT(NPC);
    short height = dst.rows;

    int writenct = 0;
    int depth = 2; // depth of histogram tree

    int bins = 16; // number of bins at each histogram level

    int nElements = 256; // int(pow((float)bins, (float)depth));

    int total = width * height;
    ap_fixed<16, 12> min_vals = inputMin - 0.5f;
    ap_fixed<16, 12> max_vals = inputMax + 0.5f;
    ap_fixed<16, 12> minValue[3] = {min_vals, min_vals, min_vals};
    ap_fixed<16, 12> maxValue[3] = {max_vals, max_vals, max_vals};

// clang-format off
#pragma HLS ARRAY_PARTITION variable = minValue complete dim = 0
#pragma HLS ARRAY_PARTITION variable = maxValue complete dim = 0
    // clang-format on

    for (int j = 0; j < 3; ++j)
    // searching for s1 and s2
    {
        ap_uint<9> p1 = 0;
        ap_uint<9> p2 = bins - 1;
        ap_uint<24> n1 = 0;
        ap_uint<24> n2 = total;

        ap_fixed<16, 12> s1 = 2.0f;
        ap_fixed<16, 12> s2 = 2.0f;

        ap_fixed<16, 12> interval = (max_vals - min_vals) / bins;

        int value = hist[j][p1];
        int value1 = hist[j][p2];

        int rval = (s1 * total) / 100;

        int rval1 = (100 - s2) * total / 100;

        while (n1 + value < rval && p1 < 256) {
// clang-format off
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=255 max=255
#pragma HLS DEPENDENCE variable=hist array intra false
            // clang-format on
            n1 += hist[j][p1++];
            minValue[j] += interval;
        }
        p1 *= bins;

        while (n2 - value1 > rval1 && p2 != 0) {
// clang-format off
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=255 max=255
#pragma HLS DEPENDENCE variable=hist array intra false
            // clang-format on

            n2 -= hist[j][p2--];
            maxValue[j] -= interval;
        }
        p2 = (p2 + 1) * bins - 1;

        interval /= bins;
    }

    ap_fixed<16, 12> maxmin_diff[3];
// clang-format off
#pragma HLS ARRAY_PARTITION variable = maxmin_diff complete dim = 0
    // clang-format on

    ap_fixed<16, 12> newmax = 255.0f;
    ap_fixed<16, 12> newmin = 0.0f;
    maxmin_diff[0] = maxValue[0] - minValue[0];
    maxmin_diff[1] = maxValue[1] - minValue[1];
    maxmin_diff[2] = maxValue[2] - minValue[2];
    ap_fixed<24, 12> newdiff = newmax - newmin;

    XF_TNAME(SRC_T, NPC) in_buf_n, in_buf_n1, out_buf_n;

    ap_fixed<24, 6> inv_val[3];

    if (maxmin_diff[0] != 0) inv_val[0] = ((ap_fixed<16, 12>)1 / maxmin_diff[0]);
    if (maxmin_diff[1] != 0) inv_val[1] = ((ap_fixed<16, 12>)1 / maxmin_diff[1]);
    if (maxmin_diff[2] != 0) inv_val[2] = ((ap_fixed<16, 12>)1 / maxmin_diff[2]);

    int pval = 0, read_index = 0, write_index = 0;
    ap_uint<13> row, col;

Row_Loop1:
    for (row = 0; row < height; row++) {
// clang-format off
#pragma HLS LOOP_TRIPCOUNT min=ROWS max=ROWS
    // clang-format on

    Col_Loop1:
        for (col = 0; col < width; col++) {
// clang-format off
#pragma HLS LOOP_TRIPCOUNT min=COLS/NPC max=COLS/NPC
#pragma HLS pipeline II=1
#pragma HLS LOOP_FLATTEN OFF
            // clang-format on
            in_buf_n = src.read(read_index++);

            ap_fixed<24, 12> value;
            ap_fixed<24, 12> divval;
            ap_fixed<24, 12> finalmul;
            ap_int<32> dstval;

            for (int p = 0, bit = 0; p < XF_NPIXPERCYCLE(NPC) * XF_CHANNELS(SRC_T, NPC); p++, bit = p % 3) {
// clang-format off
#pragma HLS unroll
                // clang-format on
                ap_uint<8> val = in_buf_n.range(p * 8 + 7, p * 8);
                value = val - minValue[bit];
                divval = value / maxmin_diff[p % 3];
                finalmul = divval * newdiff;
                dstval = (int)(finalmul + newmin);

                if (dstval.range(31, 31) == 1) {
                    dstval = 0;
                }

                if (dstval > 255) {
                    out_buf_n.range((p * 8) + 7, p * 8) = 255;
                } else {
                    out_buf_n.range((p * 8) + 7, p * 8) = (unsigned char)dstval;
                }
            }

            dst.write(row * width + col, out_buf_n);
        }
    }
}

template <int SRC_T, int DST_T, int ROWS, int COLS, int NPC = 1, int WB_TYPE>
void AWBhistogramkernel(xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& src1,
                        xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& src2,
                        uint32_t hist[3][256],
                        float p,
                        float inputMin,
                        float inputMax,
                        float outputMin,
                        float outputMax) {
// clang-format off
#pragma HLS INLINE OFF
    // clang-format on

    int width = src1.cols >> XF_BITSHIFT(NPC);
    int height = src1.rows;

    XF_TNAME(SRC_T, NPC) in_pix, in_pix1, out_pix;
    int writenct = 0;

    //******************** Simple white balance ********************

    int depth = 2; // depth of histogram tree

    int bins = 16; // number of bins at each histogram level

    int nElements = 256; // int(pow((float)bins, (float)depth));

    int val[3];

// histogram initialization

INITIALIZE_HIST:
    for (int k = 0; k < 256; k++) {
// clang-format off
#pragma HLS PIPELINE
#pragma HLS LOOP_TRIPCOUNT min=256 max=256
    // clang-format on
    INITIALIZE:
        for (int hi = 0; hi < 3; hi++) {
// clang-format off
#pragma HLS UNROLL
            // clang-format on
            hist[hi][k] = 0;
        }
    }

    // Temporary array used while computing histogram
    ap_uint<32> tmp_hist[XF_NPIXPERCYCLE(NPC) * XF_CHANNELS(SRC_T, NPC)][256];
    ap_uint<32> tmp_hist1[XF_NPIXPERCYCLE(NPC) * XF_CHANNELS(SRC_T, NPC)][256];
// clang-format off
	#pragma HLS RESOURCE variable=tmp_hist core=RAM_T2P_BRAM
	#pragma HLS RESOURCE variable=tmp_hist1 core=RAM_T2P_BRAM
#pragma HLS ARRAY_PARTITION variable=tmp_hist complete dim=1
#pragma HLS ARRAY_PARTITION variable=tmp_hist1 complete dim=1
    // clang-format on
    XF_TNAME(SRC_T, NPC) in_buf, in_buf1, temp_buf;

    bool flag = 0;

HIST_INITIALIZE_LOOP:
    for (ap_uint<10> i = 0; i < 256; i++) //
    {
// clang-format off
#pragma HLS PIPELINE
        // clang-format on
        for (ap_uint<5> j = 0; j < XF_NPIXPERCYCLE(NPC) * XF_CHANNELS(SRC_T, NPC); j++) {
// clang-format off
#pragma HLS LOOP_TRIPCOUNT min=256 max=256
            // clang-format on
            tmp_hist[j][i] = 0;
            tmp_hist1[j][i] = 0;
        }
    }

    int readcnt = 0;
    ap_fixed<16, 12> min_vals = inputMin - 0.5f;
    ap_fixed<16, 12> max_vals = inputMax + 0.5f;

    ap_fixed<16, 12> minValue = min_vals, minValue1 = min_vals;
    ap_fixed<16, 12> maxValue = max_vals, maxValue1 = max_vals;

    ap_fixed<16, 12> interval = ap_fixed<16, 12>(maxValue - minValue) / bins;

ROW_LOOP:
    for (int row = 0; row != (height); row++) // histogram filling
    {
// clang-format off
#pragma HLS LOOP_TRIPCOUNT min=1 max=ROWS
    // clang-format on
    COL_LOOP:
        for (int col = 0; col < (width); col = col + 2) // histogram filling
        {
// clang-format off
#pragma HLS PIPELINE II=2
#pragma HLS LOOP_FLATTEN OFF
#pragma HLS LOOP_TRIPCOUNT min=1 max=COLS/4
            // clang-format on

            in_pix = src1.read(row * (width) + col);
            in_pix1 = src1.read((row * (width) + col) + 1);

            src2.write(row * (width) + col, in_pix);
            src2.write(row * (width) + col + 1, in_pix1);

        PLANES_LOOP:
            for (ap_uint<9> j = 0; j < XF_NPIXPERCYCLE(NPC) * XF_CHANNELS(SRC_T, NPC); j++) {
// clang-format off
#pragma HLS DEPENDENCE variable=tmp_hist array intra false
#pragma HLS DEPENDENCE variable=tmp_hist1 array intra false
#pragma HLS UNROLL
                // clang-format on

                ap_uint<8> val = 0, val1 = 0;
                val = in_pix.range(j * 8 + 7, j * 8);
                val1 = in_pix1.range(j * 8 + 7, j * 8);

                int pos = 0, pos1 = 0;
                int currentBin = 0, currentBin1 = 0;

                currentBin = int((val - minValue + (ap_fixed<16, 12>)(1e-4f)) / interval);
                currentBin1 = int((val1 - minValue1 + (ap_fixed<16, 12>)(1e-4f)) / interval);

                ++tmp_hist[j][pos + currentBin];
                ++tmp_hist1[j][pos1 + currentBin1];
            }
        }
    }

    //	Now merge computed partial histograms
    const int num_ch = XF_CHANNELS(SRC_T, NPC);

MERGE_HIST_LOOP:
    for (ap_uint<10> i = 0; i < 256; i++) {
// clang-format off
#pragma HLS pipeline
    // clang-format on

    MERGE_HIST_CH_UNROLL:
        for (ap_uint<5> ch = 0; ch < num_ch; ch++) {
// clang-format off
#pragma HLS UNROLL
            // clang-format on

            uint32_t value = 0;

        MERGE_HIST_NPPC_UNROLL:
            for (ap_uint<5> p = 0; p < XF_NPIXPERCYCLE(NPC); p++) {
// clang-format off
#pragma HLS UNROLL
                // clang-format on
                value += tmp_hist[p * num_ch + ch][i] + tmp_hist1[p * num_ch + ch][i];
            }

            hist[ch][i] = value;
        }
    }
}

template <int SRC_T, int DST_T, int ROWS, int COLS, int NPC = 1, int WB_TYPE>
void AWBhistogram(xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& src1,
                  xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& src2,
                  uint32_t histogram[3][256],
                  float thresh,
                  float inputMin,
                  float inputMax,
                  float outputMin,
                  float outputMax) {
// clang-format off
#pragma HLS INLINE OFF
    // clang-format on

    AWBhistogramkernel<XF_8UC3, XF_8UC3, ROWS, COLS, NPC, 1>(src1, src2, histogram, thresh, inputMin, inputMax,
                                                             outputMin, outputMax);
}

template <int SRC_T, int DST_T, int ROWS, int COLS, int NPC = 1, int WB_TYPE>
void AWBNormalization(xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& src,
                      xf::cv::Mat<DST_T, ROWS, COLS, NPC>& dst,
                      uint32_t histogram[3][256],
                      float thresh,
                      float inputMin,
                      float inputMax,
                      float outputMin,
                      float outputMax) {
// clang-format off
#pragma HLS INLINE OFF
    // clang-format on

    AWBNormalizationkernel<XF_8UC3, XF_8UC3, ROWS, COLS, NPC, 1>(src, dst, histogram, thresh, inputMin, inputMax,
                                                                 outputMin, outputMax);
}

template <int SRC_T, int DST_T, int ROWS, int COLS, int NPC = 1, int WB_TYPE>
void AWBGainUpdate(xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& src1,
                   xf::cv::Mat<DST_T, ROWS, COLS, NPC>& src2,
                   float thresh,
                   int i_gain[3]) {
    xf::cv::AWBGainUpdateKernel<SRC_T, SRC_T, ROWS, COLS, NPC, XF_CHANNELS(SRC_T, NPC), XF_DEPTH(SRC_T, NPC),
                                XF_DEPTH(SRC_T, NPC), XF_WORDWIDTH(SRC_T, NPC), XF_WORDWIDTH(SRC_T, NPC),
                                (COLS >> XF_BITSHIFT(NPC))>(src1, src2, thresh, i_gain);
}

template <int SRC_T, int DST_T, int ROWS, int COLS, int NPC = 1, int WB_TYPE>
void AWBChannelGain(xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& src,
                    xf::cv::Mat<DST_T, ROWS, COLS, NPC>& dst,
                    float thresh,
                    int i_gain[3]) {
    xf::cv::AWBChannelGainKernel<SRC_T, SRC_T, ROWS, COLS, NPC, XF_CHANNELS(SRC_T, NPC), XF_DEPTH(SRC_T, NPC),
                                 XF_DEPTH(SRC_T, NPC), XF_WORDWIDTH(SRC_T, NPC), XF_WORDWIDTH(SRC_T, NPC),
                                 (COLS >> XF_BITSHIFT(NPC))>(src, dst, thresh, i_gain);
}
}
}
#endif //_XF_AWB_HPP_
