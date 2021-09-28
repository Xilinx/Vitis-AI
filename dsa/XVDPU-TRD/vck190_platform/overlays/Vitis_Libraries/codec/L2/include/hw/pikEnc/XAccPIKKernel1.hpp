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

/**
 * @file XAccPIKKernel1.hpp
 */

#ifndef _XF_CODEC_XACCPIKKERNEL1_HPP_
#define _XF_CODEC_XACCPIKKERNEL1_HPP_

#include "pik_common.hpp"
#include "resize_mem.hpp"
#include "xf_utils_hw/axi_to_multi_stream.hpp"

#include <ap_int.h>
#include <hls_math.h>
#include <hls_stream.h>
#include <iostream>
#include <stdio.h>
#include <string.h>
#include <vector>

static const int depth_to_buf = MAX_NUM_BLOCK88_W * MAX_NUM_BLOCK88_H * 3;

static const int kRadius = 2;
static const float kScaleR = 1.0f;
static const float kScaleG = 1.0f;    // 2.0f - kScaleR;
static const float kInvScaleR = 1.0f; // 1.0f / kScaleR;
static const float kInvScaleG = 1.0f; // 1.0f / kScaleG;
static const double kGaborishInverse = 0.92718927264540152;

static const float kOpsinAbsorbanceMatrix[9] = {0.00117476669,  0.00248521916,  0.000304727902,
                                                0.000868968258, 0.00268593687,  0.000377467572,
                                                0.000786771008, 0.000275945873, 0.0021850043};

static const float kOpsinAbsorbanceBias[3] = {0.00105043163, 0.000960550329, 0.000559058797};

static const double kGaborish[5] = {-0.092359145662814029, -0.039253623634014627, 0.016176494530216929,
                                    0.00083458437774987476, 0.004512465323949319};

static float qmxlocal[64] = {
    3436.970459, 1844.711548, 1476.212524, 1346.807495, 1294.897095, 1219.305786, 1336.331299, 2710.854736,
    1844.711548, 1194.931519, 1049.940308, 975.210693,  927.143311,  873.855469,  985.372742,  1992.254883,
    1476.212524, 1049.940308, 969.959534,  940.327026,  888.448120,  860.512329,  1072.243286, 2168.390381,
    1346.807495, 975.210693,  940.327026,  879.867188,  833.526062,  832.720642,  1202.815796, 2435.889893,
    1294.897095, 927.143311,  888.448120,  833.526062,  794.444641,  952.046143,  1430.633179, 2310.361572,
    1219.305786, 873.855469,  860.512329,  832.720642,  952.046143,  1227.564819, 1841.159546, 1258.626343,
    1336.331299, 985.372742,  1072.243286, 1202.815796, 1430.633179, 1841.159546, 1102.327393, 693.008972,
    2710.854736, 1992.254883, 2168.390381, 2435.889893, 2310.361572, 1258.626343, 693.008972,  467.984436};

static float qmblocal[64] = {
    270.962311, 168.165771, 68.006966,  141.364029, 296.826141, 222.024902, 202.583298, 710.839050,
    168.165771, 34.349800,  16.236092,  33.422955,  74.581551,  69.456650,  80.322029,  308.227905,
    68.006966,  16.236092,  37.456863,  47.443451,  50.487465,  40.416729,  55.474945,  205.449615,
    141.364029, 33.422955,  47.443451,  80.821915,  49.162575,  33.970627,  74.286789,  250.847351,
    296.826141, 74.581551,  50.487465,  49.162575,  52.696980,  61.840820,  122.950966, 273.765747,
    222.024902, 69.456650,  40.416729,  33.970627,  61.840820,  141.816147, 268.525055, 170.392395,
    202.583298, 80.322029,  55.474945,  74.286789,  122.950966, 268.525055, 203.730957, 111.424103,
    710.839050, 308.227905, 205.449615, 250.847351, 273.765747, 170.392395, 111.424103, 87.561890};

static float qmxglb[64] = {
    3436.970459, 1844.711548, 1476.212524, 1346.807495, 1294.897095, 1219.305786, 1336.331299, 2710.854736,
    1844.711548, 1194.931519, 1049.940308, 975.210693,  927.143311,  873.855469,  985.372742,  1992.254883,
    1476.212524, 1049.940308, 969.959534,  940.327026,  888.448120,  860.512329,  1072.243286, 2168.390381,
    1346.807495, 975.210693,  940.327026,  879.867188,  833.526062,  832.720642,  1202.815796, 2435.889893,
    1294.897095, 927.143311,  888.448120,  833.526062,  794.444641,  952.046143,  1430.633179, 2310.361572,
    1219.305786, 873.855469,  860.512329,  832.720642,  952.046143,  1227.564819, 1841.159546, 1258.626343,
    1336.331299, 985.372742,  1072.243286, 1202.815796, 1430.633179, 1841.159546, 1102.327393, 693.008972,
    2710.854736, 1992.254883, 2168.390381, 2435.889893, 2310.361572, 1258.626343, 693.008972,  467.984436};

static float qmbglb[64] = {
    270.962311, 168.165771, 68.006966,  141.364029, 296.826141, 222.024902, 202.583298, 710.839050,
    168.165771, 34.349800,  16.236092,  33.422955,  74.581551,  69.456650,  80.322029,  308.227905,
    68.006966,  16.236092,  37.456863,  47.443451,  50.487465,  40.416729,  55.474945,  205.449615,
    141.364029, 33.422955,  47.443451,  80.821915,  49.162575,  33.970627,  74.286789,  250.847351,
    296.826141, 74.581551,  50.487465,  49.162575,  52.696980,  61.840820,  122.950966, 273.765747,
    222.024902, 69.456650,  40.416729,  33.970627,  61.840820,  141.816147, 268.525055, 170.392395,
    202.583298, 80.322029,  55.474945,  74.286789,  122.950966, 268.525055, 203.730957, 111.424103,
    710.839050, 308.227905, 205.449615, 250.847351, 273.765747, 170.392395, 111.424103, 87.561890};

inline float hls_SimpleGammaRGB(float v) {
#pragma HLS inline
    int ix;
    ix = fToBits<float, int>(v);
    ix = 0x2a50f200 + ix / 3;

    float x0;
    x0 = bitsToF<int, float>(ix);

    float kOneThird = 0.333333343; // 1.0f / 3.0f;
    float x1 = kOneThird * (2.0f * x0 + v / (x0 * x0));

    float x2 = kOneThird * (2.0f * x1 + v / (x1 * x1));
    return x2;
}

inline void hls_LinerToXyb(const float r, const float g, const float b, float& valx, float& valy, float& valz) {
    float mixed[3];
    const float* mix = &kOpsinAbsorbanceMatrix[0];
    const float* bias = &kOpsinAbsorbanceBias[0];
    ap_uint<3> c;

RGB_TO_MIXED:
    for (c = 0; c < 3; c++) {
#pragma HLS pipeline II = 1
        ap_uint<3> c_tmp = c * 3;
        mixed[c] = mix[c_tmp] * r + mix[c_tmp + 1] * g + mix[c_tmp + 2] * b + bias[c];
        mixed[c] = 0.0f > mixed[c] ? 0.0f : mixed[c];
        mixed[c] = hls_SimpleGammaRGB(mixed[c]);
    }
    float mix0 = kScaleR * mixed[0];
    float mix1 = kScaleG * mixed[1];
    valx = (mix0 - mix1) * 0.5f;
    valy = (mix0 + mix1) * 0.5f;
    valz = mixed[2];
}

inline void hls_OpsinDynamicsImage(
    hls::stream<float> row_in[3], int xsize, int ysize, hls::stream<float> row_out[3], hls::stream<float>& row_y) {
    float row_in0, row_in1, row_in2;
    float row_xyb0, row_xyb1, row_xyb2;

LINEER_TO_XYB:
    for (int y = 0; y < ysize; y++) {
#pragma HLS LOOP_TRIPCOUNT min = 512 max = 512
        for (int x = 0; x < xsize; x++) {
#pragma HLS LOOP_TRIPCOUNT min = 512 max = 512
#pragma HLS pipeline II = 3
            row_in0 = row_in[0].read();
            row_in1 = row_in[1].read();
            row_in2 = row_in[2].read();

            hls_LinerToXyb(row_in0, row_in1, row_in2, row_xyb0, row_xyb1, row_xyb2);

            row_out[0].write(row_xyb0);
            row_out[1].write(row_xyb1);
            row_out[2].write(row_xyb2);

            row_y.write(row_xyb1);
        }
    }
}

inline float FPTwoMul(float in1, float in2) {
#pragma HLS inline
    float r = 0.0;
    r = in1 * in2;
    return r;
}

inline float FPTwoAdd(float in1, float in2) {
#pragma HLS inline
    float r = 0.0;
    r = in1 + in2;
    return r;
}

inline int DivCeil(int a, int b) {
#pragma HLS inline
    return (a + b - 1) / b;
}

inline void hls_GaborishInverse(hls::stream<float> io_strm[3], int xsize, int ysize, hls::stream<float> opsin_strm[3]) {
    float normalized[9] = {
        1.6812343597412109375,           -0.14397151768207550048828125,      0.02521628327667713165283203125,
        -0.14397151768207550048828125,   -0.0611894316971302032470703125,    0.00130096892826259136199951171875,
        0.02521628327667713165283203125, 0.00130096892826259136199951171875, 0.00703413225710391998291015625};

#ifndef __SYNTHESIS__
    std::vector<std::vector<std::vector<float> > > linebuf(
        3, std::vector<std::vector<float> >(5, std::vector<float>(4096)));
#else
    float linebuf[3][5][4096];
#pragma HLS RESOURCE variable = linebuf core = RAM_S2P_URAM
#pragma HLS array_partition variable = linebuf dim = 1 complete
#pragma HLS array_partition variable = linebuf dim = 2 complete
#pragma HLS dependence variable = linebuf inter false
#pragma HLS dependence variable = linebuf intra false
#endif
    Window<5, 5, float> window[3];
#pragma HLS array_partition variable = window dim = 1 complete

    float temp_in[3][5];
    float temp_out[3][5];

    const int x_blocks = (xsize + 7) / 8;
    const int y_blocks = (ysize + 7) / 8;
    int ali_xsize = 8 * x_blocks;
    int ali_ysize = 8 * y_blocks;
    bool is_align = xsize == ali_xsize ? true : false;

Y:
    for (int iy = 0; iy < ali_ysize + 2; iy++) {
#pragma HLS LOOP_TRIPCOUNT min = 514 max = 514
    X:
        for (int ix = 0; ix < ali_xsize + 2; ix++) {
#pragma HLS LOOP_TRIPCOUNT min = 514 max = 514
            for (int c = 0; c < 3; c++) {
#pragma HLS PIPELINE II = 1
                for (int i = 0; i < 5; i++) {
#pragma HLS UNROLL
                    temp_out[c][i] = linebuf[c][i][ix];
                }

                window[c].shift_left();
                if (iy < ysize && ix < xsize) {
                    float tmp = io_strm[c].read();
                    temp_in[c][4] = tmp;
                } else {
                    temp_in[c][4] = temp_out[c][4];
                }

                for (int i = 4; i > 0; i--) {
                    temp_in[c][i - 1] = temp_out[c][i];
                }

            LOAD_FOR_WINDOW:
                for (int i = 0; i < 5; i++) {
#pragma HLS unroll
                    if (iy == 0 && ix == 0) {
                        window[c].val[i][0] = temp_in[c][4];
                        window[c].val[i][1] = temp_in[c][4];
                        window[c].val[i][2] = temp_in[c][4];
                        window[c].val[i][3] = temp_in[c][4];
                        window[c].val[i][4] = temp_in[c][4];
                    } else if (iy > 0 && ix == 0) {
                        window[c].val[i][0] = temp_in[c][i];
                        window[c].val[i][1] = temp_in[c][i];
                        window[c].val[i][2] = temp_in[c][i];
                        window[c].val[i][3] = temp_in[c][i];
                        window[c].val[i][4] = temp_in[c][i];
                    } else if (ix < xsize) {
                        if (iy > 0) {
                            if (ix == 1) window[c].val[i][1] = temp_in[c][i];
                            window[c].val[i][4] = temp_in[c][i];
                        } else {
                            if (ix == 1) window[c].val[i][1] = temp_in[c][4];
                            window[c].val[i][4] = temp_in[c][4];
                        }
                    } else {
                        if (is_align && ix >= (xsize + 1)) window[c].val[i][4] = window[c].val[i][1];
                    }
                }

                for (int i = 0; i < 5; i++) {
#pragma HLS unroll
                    linebuf[c][i][ix] = (iy > 0) ? temp_in[c][i] : temp_in[c][4];
                } // i

                if (iy >= 2 && ix >= 2) {
                    float sum = 0.0;
                SUM_ADD_Y:
                    for (int ky = -kRadius, y = 0; ky <= kRadius; ky++, y++) {
#pragma HLS unroll
                    SUM_ADD_X:
                        for (int kx = -kRadius, x = 0; kx <= kRadius; kx++, x++) {
#pragma HLS unroll
                            const int wy = hls::abs(ky);
                            const int wx = hls::abs(kx);
                            float tmp = normalized[wy * (kRadius + 1) + wx];
                            sum = FPTwoAdd(sum, FPTwoMul(window[c].val[y][x], tmp));
                        } // kx
                    }     // ky
                    opsin_strm[c].write(sum);
                }
            }
        }
    }
}

inline double hls_SimpleGamma(float v) {
#pragma HLS inline
    // A simple HDR compatible gamma function.
    // mul and mul2 represent a scaling difference between pik and butteraugli.
    static const float mul = 103.34350600371506;
    static const float mul2 = 1.0 / (67.797075768826289);

    v *= mul;

    static const float kRetMul = mul2 * 18.6580932135;
    static const float kRetAdd = mul2 * -20.2789020414;
    static const float kVOffset = 7.14672470003;

    if (v < 0) {
        // This should happen rarely, but may lead to a NaN, which is rather
        // undesirable. Since negative photons don't exist we solve the NaNs by
        // clamping here.
        v = 0;
    }
    return kRetMul * hls::log(v + kVOffset) + kRetAdd;
}

inline double hls_RatioOfCubicRootToSimpleGamma(float v) {
#pragma HLS inline
    // The opsin space in pik is the cubic root of photons, i.e., v * v * v
    // is related to the number of photons.
    //
    // SimpleGamma(v * v * v) is the psychovisual space in butteraugli.
    // This ratio allows quantization to move from pik's opsin space to
    // butteraugli's log-gamma space.
    return v / hls_SimpleGamma(v * v * v);
}

inline void hls_RatioOfCubicRootToSimpleGammaStrm(
    int xsize, int ysize, float cutoff, hls::stream<float>& orig, hls::stream<float>& diff, hls::stream<float>& out) {
    static const double mul0 = 0.046650519741099357;

    // PIK's gamma is 3.0 to be able to decode faster with two muls.
    // Butteraugli's gamma is matching the gamma of human eye, around 2.6.
    // We approximate the gamma difference by adding one cubic root into
    // the adaptive quantization. This gives us a total gamma of 2.6666
    // for quantization uses.
    static const double match_gamma_offset = 0.55030107636310233;
    float origin, differe, res, fin;

LOOP_GAMMA:
    for (int i = 0; i < xsize * ysize - 1; i++) {
#pragma HLS pipeline II = 1
        origin = orig.read();
        differe = diff.read();
        res = mul0 * differe * hls_RatioOfCubicRootToSimpleGamma(origin + match_gamma_offset);
        fin = res < cutoff ? res : cutoff;
        out.write(fin);
    }

    origin = orig.read();
    differe = diff.read();
    out.write(fin);
}

inline void hls_ConvolveY(
    int xsize, int ysize, hls::stream<float>& orig_in, hls::stream<float>& orig_out, hls::stream<float>& diff_out) {
    static const float kOverWeightBorders = 1.4;

    float origAbove[2][4096];
#pragma HLS resource variable = origAbove core = RAM_S2P_URAM
#pragma HLS ARRAY_PARTITION variable = origAbove complete dim = 1
#pragma HLS DEPENDENCE variable = origAbove inter false

    float orgL[3];
#pragma HLS ARRAY_PARTITION variable = orgL complete dim = 0

    float diff, orgabv, orgcur, orgnxt;

LOOP_CONVOLVE_Y:
    for (int i = 0; i < xsize; i++) {
#pragma HLS pipeline II = 1
        //		above[1][i] = in.read();
        origAbove[1][i] = orig_in.read();
    }

    orgL[2] = orgL[1];
    orgL[1] = orgL[0];
    orgL[0] = origAbove[1][0];

    ap_uint<10> cnt;
    cnt = 0;

    for (int i = 0; i < xsize; i++) {
#pragma HLS pipeline II = 1
        orgnxt = orig_in.read();

        orgL[2] = orgL[1];
        orgL[1] = orgL[0];
        orgL[0] = origAbove[1][cnt + 1];
        if (i == 0) {
            diff = hls::fabs(orgL[1] - orgnxt) + hls::fabs(orgL[1] - orgnxt) + hls::fabs(orgL[0] - orgL[1]) +
                   hls::fabs(orgL[0] - orgL[1]);
        } else if (i != xsize - 1) {
            diff = hls::fabs(orgL[1] - orgnxt) + hls::fabs(orgL[1] - orgnxt) + hls::fabs(orgL[0] - orgL[1]) +
                   hls::fabs(orgL[1] - orgL[2]) + 3 * hls::fabs(orgL[0] - orgL[2]);
        } else if (i == xsize - 1) {
            diff = kOverWeightBorders * (hls::fabs(orgL[1] - orgnxt) + hls::fabs(orgL[1] - orgnxt));
        }

        origAbove[0][i] = orgnxt;

        diff_out.write(diff);
        orig_out.write(orgL[1]);
        cnt++;
    }

    bool lb;
    lb = 1;

    for (int i = 0; i < ysize - 2; i++) {
        lb = !lb;
        cnt = 0;
        orgL[2] = orgL[1];
        orgL[1] = orgL[0];
        orgL[0] = origAbove[lb][0];

        for (int j = 0; j < xsize; j++) {
#pragma HLS PIPELINE II = 1
            orgabv = origAbove[!lb][j];
            orgnxt = orig_in.read();

            orgL[2] = orgL[1];
            orgL[1] = orgL[0];
            orgL[0] = origAbove[lb][cnt + 1];
            if (j == 0) {
                diff = hls::fabs(orgL[1] - orgnxt) + hls::fabs(orgL[1] - orgabv) + hls::fabs(orgL[0] - orgL[1]) +
                       hls::fabs(orgL[0] - orgL[1]) + 3 * hls::fabs(orgabv - orgnxt);
            } else if (j != xsize - 1) {
                diff = hls::fabs(orgL[1] - orgnxt) + hls::fabs(orgL[1] - orgabv) + hls::fabs(orgL[0] - orgL[1]) +
                       hls::fabs(orgL[1] - orgL[2]) + 3 * (hls::fabs(orgL[0] - orgL[2]) + hls::fabs(orgabv - orgnxt));
            } else if (j == xsize - 1) {
                diff = kOverWeightBorders * (hls::fabs(orgL[1] - orgnxt) + hls::fabs(orgL[1] - orgnxt));
            }

            origAbove[!lb][j] = orgnxt;

            diff_out.write(diff);
            orig_out.write(orgL[1]);
            cnt++;
        }
    }

    orgL[2] = orgL[1];
    orgL[1] = orgL[0];
    orgL[0] = origAbove[!lb][0];
    cnt = 0;

    for (int i = 0; i < xsize; i++) {
#pragma HLS PIPELINE II = 1
        orgL[2] = orgL[1];
        orgL[1] = orgL[0];
        orgL[0] = origAbove[!lb][cnt + 1];
        if (i != xsize - 1) {
            diff = kOverWeightBorders * 2 * hls::fabs(orgL[0] - orgL[1]);
        } else if (i == xsize - 1) {
            diff = kOverWeightBorders * 2 * hls::fabs(orgL[1] - orgL[2]);
        }

        diff_out.write(diff);
        orig_out.write(orgL[1]);
        cnt++;
    }
}

inline void hls_ExpandStrm(ap_uint<32> xsize, ap_uint<32> ysize, hls::stream<float>& in, hls::stream<float>& out) {
    ap_uint<32> out_xsize, out_ysize;
    float res;
    float right_last[3];

    if (xsize.range(2, 0) != 0)
        out_xsize = (xsize | 7) + 1;
    else
        out_xsize = xsize;

    if (ysize.range(2, 0) != 0)
        out_ysize = (ysize | 7) + 1;
    else
        out_ysize = ysize;

    float sum[4096];
#pragma HLS resource variable = sum core = RAM_S2P_URAM
#pragma HLS DEPENDENCE variable = sum inter distance = 8

    float sum_right[7];
#pragma HLS resource variable = sum_right core = RAM_S2P_LUTRAM
#pragma HLS DEPENDENCE variable = sum_right inter distance = 8

    for (int i = 0; i < 3; i++) {
#pragma HLS unroll
        right_last[i] = 0;
    }

    for (int j = 0; j < ysize; j++) {
        for (int i = 0; i < out_xsize; i++) {
#pragma HLS pipeline II = 1
            if (i < xsize) {
                right_last[2] = right_last[1];
                right_last[1] = right_last[0];
                right_last[0] = in.read();
                res = right_last[0];
                float sum_pre = sum[i];
                if (j == 0 || j + 3 == ysize)
                    sum[i] = res;
                else if (j + 3 > ysize)
                    sum[i] = sum_pre + res;
            } else {
                res = (right_last[2] + right_last[1] + right_last[0]) * 0.3333333333333333333333333;
                float sum_pre = sum_right[i - xsize];
                if (j == 0 || j + 3 == ysize)
                    sum_right[i - xsize] = res;
                else if (j + 3 > ysize)
                    sum_right[i - xsize] = sum_pre + res;
            }
            out.write(res);
        }
    }

    for (int j = ysize; j < out_ysize; j++) {
        for (int i = 0; i < out_xsize; i++) {
#pragma HLS pipeline II = 1
            if (i < xsize)
                res = sum[i] * 0.3333333333333333333333333;
            else
                res = sum_right[i - xsize] * 0.3333333333333333333333333;
            out.write(res);
        }
    }
}

inline void hls_ConvolveX35(int xsize, int ysize, hls::stream<float>& in, hls::stream<float>& out) {
    float kernel[35] = {
        0.0060024694539606571197509765625, 0.0076467366889119148254394531250, 0.0095995273441076278686523437500,
        0.0118754766881465911865234375000, 0.0144770387560129165649414062500, 0.0173914544284343719482421875000,
        0.0205882601439952850341796875000, 0.0240176711231470108032226562500, 0.0276102013885974884033203125000,
        0.0312777720391750335693359375000, 0.0349164046347141265869140625000, 0.0384105704724788665771484375000,
        0.0416389256715774536132812500000, 0.0444811210036277770996093750000, 0.0468251816928386688232421875000,
        0.0485747642815113067626953125000, 0.0496557392179965972900390625000, 0.0500213839113712310791015625000,
        0.0496557392179965972900390625000, 0.0485747642815113067626953125000, 0.0468251816928386688232421875000,
        0.0444811210036277770996093750000, 0.0416389256715774536132812500000, 0.0384105704724788665771484375000,
        0.0349164046347141265869140625000, 0.0312777720391750335693359375000, 0.0276102013885974884033203125000,
        0.0240176711231470108032226562500, 0.0205882601439952850341796875000, 0.0173914544284343719482421875000,
        0.0144770387560129165649414062500, 0.0118754766881465911865234375000, 0.0095995273441076278686523437500,
        0.0076467366889119148254394531250, 0.0060024694539606571197509765625};

    float window[35];
#pragma HLS ARRAY_PARTITION variable = window complete dim = 0
    float last[19];
#pragma HLS ARRAY_PARTITION variable = last complete dim = 0
    float res[7];
#pragma HLS ARRAY_PARTITION variable = res complete dim = 0
    float sum, tmp;

    for (int i = 0; i < 7; i++) {
#pragma HLS unroll
        res[i] = 0;
    }

    for (int m = 0; m < ysize; m++) {
        for (int i = 0; i < 18; i++) {
#pragma HLS pipeline II = 1
            if (i < xsize) tmp = in.read();

            for (int j = 0; j < 17; j++) {
#pragma HLS unroll
                window[17 + j] = window[18 + j];
            }
            window[34] = tmp;

            for (int j = 0; j < 16; j++) {
#pragma HLS unroll
                window[16 - j] = window[15 - j];
            }
            if (i < xsize)
                window[0] = tmp;
            else
                window[0] = last[17];

            if (i == 0) {
                for (int j = 0; j < 19; j++) {
#pragma HLS unroll
                    last[j] = tmp;
                }
            } else if (i < xsize) {
                for (int j = 0; j < 18; j++) {
#pragma HLS unroll
                    last[j] = last[j + 1];
                }
                last[18] = tmp;
            } else {
                for (int j = 0; j < 18; j++) {
#pragma HLS unroll
                    last[18 - j] = last[17 - j];
                }
            }
        }

        int state, cnt;
        float reg[35];
#pragma HLS ARRAY_PARTITION variable = reg complete dim = 0
        state = 8;
        cnt = 0;

        for (int i = 0; i < xsize + 4; i++) {
#pragma HLS pipeline II = 1

            if (state == 8) {
                cnt++;
                if (cnt < 4)
                    state = 8;
                else
                    state = 0;
            } else if (state == 0) {
                for (int j = 0; j < 35; j++) {
#pragma HLS unroll
                    reg[j] = window[34 - j];
                }
                res[0] = reg[0] * kernel[0] + reg[1] * kernel[1] + reg[2] * kernel[2] + reg[3] * kernel[3] +
                         reg[4] * kernel[4];
                state = 1;
            } else if (state == 1) {
                for (int j = 0; j < 30; j++) {
#pragma HLS unroll
                    reg[j] = reg[j + 5];
                }
                res[1] = reg[0] * kernel[5] + reg[1] * kernel[6] + reg[2] * kernel[7] + reg[3] * kernel[8] +
                         reg[4] * kernel[9];
                state = 2;
            } else if (state == 2) {
                for (int j = 0; j < 30; j++) {
#pragma HLS unroll
                    reg[j] = reg[j + 5];
                }
                res[2] = reg[0] * kernel[10] + reg[1] * kernel[11] + reg[2] * kernel[12] + reg[3] * kernel[13] +
                         reg[4] * kernel[14];
                state = 3;
            } else if (state == 3) {
                for (int j = 0; j < 30; j++) {
#pragma HLS unroll
                    reg[j] = reg[j + 5];
                }
                res[3] = reg[0] * kernel[15] + reg[1] * kernel[16] + reg[2] * kernel[17] + reg[3] * kernel[18] +
                         reg[4] * kernel[19];
                state = 4;
            } else if (state == 4) {
                for (int j = 0; j < 30; j++) {
#pragma HLS unroll
                    reg[j] = reg[j + 5];
                }
                res[4] = reg[0] * kernel[20] + reg[1] * kernel[21] + reg[2] * kernel[22] + reg[3] * kernel[23] +
                         reg[4] * kernel[24];
                state = 5;
            } else if (state == 5) {
                for (int j = 0; j < 30; j++) {
#pragma HLS unroll
                    reg[j] = reg[j + 5];
                }
                res[5] = reg[0] * kernel[25] + reg[1] * kernel[26] + reg[2] * kernel[27] + reg[3] * kernel[28] +
                         reg[4] * kernel[29];
                state = 6;
            } else if (state == 6) {
                for (int j = 0; j < 30; j++) {
#pragma HLS unroll
                    reg[j] = reg[j + 5];
                }
                res[6] = reg[0] * kernel[30] + reg[1] * kernel[31] + reg[2] * kernel[32] + reg[3] * kernel[33] +
                         reg[4] * kernel[34];
                state = 7;
            } else if (state == 7) {
                for (int j = 0; j < 30; j++) {
#pragma HLS unroll
                    reg[j] = reg[j + 5];
                }
                out.write(res[0] + res[1] + res[2] + res[3] + res[4] + res[5] + res[6]);
                state = 0;
            }

            if (i < xsize - 18) {
                tmp = in.read();
                for (int j = 0; j < 34; j++) {
#pragma HLS unroll
                    window[34 - j] = window[33 - j];
                }
                window[0] = tmp;

                for (int j = 0; j < 18; j++) {
#pragma HLS unroll
                    last[j] = last[j + 1];
                }
                last[18] = tmp;
            } else {
                tmp = last[17];
                for (int j = 0; j < 34; j++) {
#pragma HLS unroll
                    window[34 - j] = window[33 - j];
                }
                window[0] = tmp;

                for (int j = 0; j < 18; j++) {
#pragma HLS unroll
                    last[18 - j] = last[17 - j];
                }
                last[0] = 0;
            }
        }
    }
}

inline void hls_ConvolveY35(int xsize, int ysize, hls::stream<float>& in, hls::stream<float>& out) {
    float kernel[35] = {
        0.0060024694539606571197509765625, 0.0076467366889119148254394531250, 0.0095995273441076278686523437500,
        0.0118754766881465911865234375000, 0.0144770387560129165649414062500, 0.0173914544284343719482421875000,
        0.0205882601439952850341796875000, 0.0240176711231470108032226562500, 0.0276102013885974884033203125000,
        0.0312777720391750335693359375000, 0.0349164046347141265869140625000, 0.0384105704724788665771484375000,
        0.0416389256715774536132812500000, 0.0444811210036277770996093750000, 0.0468251816928386688232421875000,
        0.0485747642815113067626953125000, 0.0496557392179965972900390625000, 0.0500213839113712310791015625000,
        0.0496557392179965972900390625000, 0.0485747642815113067626953125000, 0.0468251816928386688232421875000,
        0.0444811210036277770996093750000, 0.0416389256715774536132812500000, 0.0384105704724788665771484375000,
        0.0349164046347141265869140625000, 0.0312777720391750335693359375000, 0.0276102013885974884033203125000,
        0.0240176711231470108032226562500, 0.0205882601439952850341796875000, 0.0173914544284343719482421875000,
        0.0144770387560129165649414062500, 0.0118754766881465911865234375000, 0.0095995273441076278686523437500,
        0.0076467366889119148254394531250, 0.0060024694539606571197509765625};
#pragma HLS resource variable = kernel core = ROM_1P_LUTRAM

    float lb_ram[5][4096];
#pragma HLS resource variable = lb_ram core = RAM_S2P_URAM
#pragma HLS ARRAY_PARTITION variable = lb_ram complete dim = 1
#pragma HLS DEPENDENCE variable = lb_ram inter distance = 2

    float reg;

    float data[5];
#pragma HLS ARRAY_PARTITION variable = data complete dim = 0

    int r[5][3];
#pragma HLS ARRAY_PARTITION variable = r complete dim = 0

    int p[5];
#pragma HLS ARRAY_PARTITION variable = p complete dim = 0
    p[0] = 4;
    p[1] = 12;
    p[2] = 20;
    p[3] = 28;
    p[4] = 36;

    int cnt[5];
#pragma HLS ARRAY_PARTITION variable = cnt complete dim = 0
    cnt[0] = 0;
    cnt[1] = 0;
    cnt[2] = 0;
    cnt[3] = 0;
    cnt[4] = 0;

    float tmp1[5], tmp2[5], tmp3[5];
#pragma HLS ARRAY_PARTITION variable = tmp1 complete dim = 0
#pragma HLS ARRAY_PARTITION variable = tmp2 complete dim = 0
#pragma HLS ARRAY_PARTITION variable = tmp3 complete dim = 0

    for (int i = 0; i < xsize; i++) {
#pragma HLS pipeline II = 1
        reg = in.read();
        lb_ram[0][i] = reg * kernel[13];
        lb_ram[1][i] = reg * kernel[5];
        lb_ram[2][i] = 0;
        lb_ram[3][i] = 0;
        lb_ram[4][i] = 0;
    }
    cnt[0]++;
    cnt[1]++;

    for (int i = 1; i < ysize - 1; i++) {
        for (int m = 0; m < 5; m++) {
#pragma HLS UNROLL
            r[m][0] = -i - p[m]; //(i - p0) - 2 * i
            r[m][1] = i - p[m];
            r[m][2] = i - p[m] + 2 * (ysize - 1 - i);
        }

        for (int m = 0; m < 5; m++) {
#pragma HLS UNROLL
            if (r[m][0] > -18 && r[m][0] < 18) {
                cnt[m]++;
            }
            if (r[m][1] > -18 && r[m][1] < 18) {
                cnt[m]++;
            }
            if (r[m][2] > -18 && r[m][2] < 18) {
                cnt[m]++;
            }
        }

        for (int j = 0; j < xsize; j++) {
#pragma HLS PIPELINE II = 8
            reg = in.read();
            for (int m = 0; m < 5; m++) {
#pragma HLS UNROLL
                data[m] = lb_ram[m][j];
            }

            for (int m = 0; m < 5; m++) {
#pragma HLS UNROLL
                if (r[m][0] > -18 && r[m][0] < 18) {
                    tmp1[m] = reg * kernel[17 + r[m][0]];
                } else {
                    tmp1[m] = 0;
                }

                if (r[m][1] > -18 && r[m][1] < 18) {
                    tmp2[m] = reg * kernel[17 + r[m][1]];
                } else {
                    tmp2[m] = 0;
                }

                if (r[m][2] > -18 && r[m][2] < 18) {
                    tmp3[m] = reg * kernel[17 + r[m][2]];
                } else {
                    tmp3[m] = 0;
                }
            }

            for (int m = 0; m < 5; m++) {
#pragma HLS UNROLL
                if (cnt[m] == 35) {
                    lb_ram[m][j] = 0;
                } else {
                    lb_ram[m][j] = data[m] + tmp1[m] + tmp2[m] + tmp3[m];
                }
            }

            if (cnt[0] == 35) {
                out.write(data[0] + tmp1[0] + tmp2[0] + tmp3[0]);
            } else if (cnt[1] == 35) {
                out.write(data[1] + tmp1[1] + tmp2[1] + tmp3[1]);
            } else if (cnt[2] == 35) {
                out.write(data[2] + tmp1[2] + tmp2[2] + tmp3[2]);
            } else if (cnt[3] == 35) {
                out.write(data[3] + tmp1[3] + tmp2[3] + tmp3[3]);
            } else if (cnt[4] == 35) {
                out.write(data[4] + tmp1[4] + tmp2[4] + tmp3[4]);
            }
        }

        for (int m = 0; m < 5; m++) {
#pragma HLS unroll
            if (cnt[m] == 35) {
                p[m] = p[m] + 40;
            }
        }

        for (int m = 0; m < 5; m++) {
#pragma HLS unroll
            if (cnt[m] == 35) cnt[m] = 0;
        }
    }

    for (int i = 0; i < xsize; i++) {
#pragma HLS pipeline II = 8
        reg = in.read();
        if (ysize != 8) {
            if (p[0] < p[4]) {
                out.write(lb_ram[0][i] + reg * kernel[28]);
                lb_ram[1][i] = lb_ram[1][i] + reg * kernel[20];
            } else if (p[1] < p[0]) {
                out.write(lb_ram[1][i] + reg * kernel[28]);
                lb_ram[2][i] = lb_ram[2][i] + reg * kernel[20];
            } else if (p[2] < p[1]) {
                out.write(lb_ram[2][i] + reg * kernel[28]);
                lb_ram[3][i] = lb_ram[3][i] + reg * kernel[20];
            } else if (p[3] < p[2]) {
                out.write(lb_ram[3][i] + reg * kernel[28]);
                lb_ram[4][i] = lb_ram[4][i] + reg * kernel[20];
            } else if (p[4] < p[3]) {
                out.write(lb_ram[4][i] + reg * kernel[28]);
                lb_ram[0][i] = lb_ram[0][i] + reg * kernel[20];
            }
        }
    }

    for (int i = 0; i < xsize; i++) {
#pragma HLS pipeline II = 1
        if (p[0] < p[4]) {
            out.write(lb_ram[1][i]);
        } else if (p[1] < p[0]) {
            out.write(lb_ram[2][i]);
        } else if (p[2] < p[1]) {
            out.write(lb_ram[3][i]);
        } else if (p[3] < p[2]) {
            out.write(lb_ram[4][i]);
        } else if (p[4] < p[3]) {
            out.write(lb_ram[0][i]);
        }
    }
}

inline void hls_ComputeMaskStrm(int xsize, int ysize, hls::stream<float>& diff, hls::stream<float>& res) {
    static const float kBase = 1.329262607500535;
    static const float kMul1 = 0.010994306366172898;
    static const float kOffset1 = 0.00683227084849159;
    static const float kMul2 = -0.1949226495025296;
    static const float kOffset2 = 0.075052668223305155;

    for (int y = 0; y < ysize; ++y) {
        for (int x = 0; x < xsize; ++x) {
#pragma HLS pipeline II = 1
            float val = diff.read();
            // Avoid division by zero.
            float div = hls::max<float>(val + kOffset1, 1e-3);
            res.write(kBase + kMul1 / div + kMul2 / (val * val + kOffset2));
        }
    }
}

inline void hls_Exp(int xsize, int ysize, hls::stream<float>& in, hls::stream<float>& out) {
    for (int y = 0; y < ysize; ++y) {
        for (int x = 0; x < xsize; ++x) {
#pragma HLS pipeline II = 1
            out.write(hls::exp(in.read()));
        }
    }
}

inline void hls_scale(
    int xsize, int ysize, int outx, int outy, float lambda, hls::stream<float>& in, hls::stream<float>& out) {
    if (xsize > 1 && ysize > 1) {
        for (size_t y = 0; y < outy; ++y) {
            for (size_t x = 0; x < outx; ++x) {
#pragma HLS pipeline II = 1
                out.write(lambda * in.read());
            }
        }
    } else if (xsize == 1) {
        for (size_t y = 0; y < outy; ++y) {
#pragma HLS pipeline II = 1
            out.write(lambda);
        }
    } else if (ysize == 1) {
        for (size_t x = 0; x < outx; ++x) {
#pragma HLS pipeline II = 1
            out.write(lambda);
        }
    }
}

inline void hls_average(int xsize, int ysize, hls::stream<float>& qfStrm, hls::stream<float>& avgStrm) {
    float sum = 0;

    for (int x = 0; x < xsize * ysize; x++) {
#pragma HLS pipeline II = 8
        sum = sum + qfStrm.read();
    }
    avgStrm.write(sum / (xsize * ysize));
}

inline void duplicateQF(
    int xsize, int ysize, hls::stream<float>& input, hls::stream<float>& output1, hls::stream<float>& output2) {
    for (int i = 0; i < xsize; i++) {
        for (int j = 0; j < ysize; j++) {
#pragma HLS pipeline II = 1
            float reg = input.read();
            output1.write(reg);
            output2.write(reg);
        }
    }
}

inline void initQFStrm(int xsize,
                       int ysize,
                       float cutoff,
                       float lamda,
                       hls::stream<float>& in,
                       hls::stream<float>& qfStrmOut,
                       hls::stream<float>& avgStrmOut) {
#pragma HLS inline

    hls::stream<float> mid("mid");
#pragma HLS STREAM variable = mid depth = 32

    hls::stream<float> diff("diff");
#pragma HLS STREAM variable = diff depth = 32
    hls::stream<float> orig("orig");
#pragma HLS STREAM variable = orig depth = 32
    hls::stream<float> gamma("gamma");
#pragma HLS STREAM variable = orig depth = 32
    hls::stream<float> expd("expd");
#pragma HLS STREAM variable = orig depth = 32
    hls::stream<float> mid35("mid35");
#pragma HLS STREAM variable = mid35 depth = 32
    hls::stream<float> mask("mask");
#pragma HLS STREAM variable = mask depth = 32
    hls::stream<float> exp("exp");
#pragma HLS STREAM variable = exp depth = 32
    hls::stream<float> scale("scale");
#pragma HLS STREAM variable = exp depth = 32
    hls::stream<float> qfStrm("qfStrm");
#pragma HLS STREAM variable = qfStrm depth = 32
    hls::stream<float> avgS1Strm("avgS1Strm");
#pragma HLS STREAM variable = qfStrm depth = 32

    static const int kResolution = 8;
    int out_xsize = (xsize + kResolution - 1) / kResolution;
    int out_ysize = (ysize + kResolution - 1) / kResolution;

    hls_ConvolveY(xsize, ysize, in, orig, diff);
    hls_RatioOfCubicRootToSimpleGammaStrm(xsize, ysize, cutoff, orig, diff, gamma);
    hls_ExpandStrm(xsize, ysize, gamma, expd);
    hls_ConvolveX35(out_xsize << 3, out_ysize << 3, expd, mid35);
    hls_ConvolveY35(out_xsize, out_ysize << 3, mid35, mask);
    hls_ComputeMaskStrm(out_xsize, out_ysize, mask, exp);
    hls_Exp(out_xsize, out_ysize, exp, scale);
    hls_scale(xsize, ysize, out_xsize, out_ysize, lamda, scale, qfStrm);
    duplicateQF(out_xsize, out_ysize, qfStrm, qfStrmOut, avgS1Strm);
    hls_average(out_xsize, out_ysize, avgS1Strm, avgStrmOut);
}

inline void QFwriteCalAddr(int xsize,
                           int ysize,
                           hls::stream<float>& input,
                           hls::stream<float>& dataStrm,
                           hls::stream<ap_uint<32> >& addrStrm) {
    int x4 = (xsize + 4 - 1) / 4;
    int y4 = (ysize + 4 - 1) / 4;
    float y[4];
#pragma HLS ARRAY_PARTITION variable = y complete dim = 1
    bool ping = 0;

    for (int y = 0; y < y4; y++) {
        for (int j = 0; j < 4; j++) {
            for (int x = 0; x < x4; x++) {
                for (int i = 0; i < 4; i++) {
#pragma HLS pipeline II = 1
                    if ((x * 4 + i) < xsize && (y * 4 + j) < ysize)
                        dataStrm.write(input.read());
                    else
                        dataStrm.write(0);
                }
            }
        }
    }
}

inline void QFwriteDDRCtrl(int xsize,
                           int ysize,
                           hls::stream<float>& avgStrm,
                           hls::stream<float>& avg_outStrm,
                           hls::stream<float>& dataStrm,
                           hls::stream<ap_uint<32> >& addrStrm,
                           ap_uint<32>* axi_qf) {
    float reg;
    unsigned int reg_int;
    ap_uint<32> reg_apint;

    int x4 = (xsize + 4 - 1) / 4;
    int y4 = (ysize + 4 - 1) / 4;
    int n = x4 * 4 * y4;
    ap_uint<32> addr;

    for (int y = 0; y < y4; y++) {
        for (int j = 0; j < 4; j++) {
            for (int x = 0; x < x4; x++) {
                for (int i = 0; i < 4; i++) {
#pragma HLS pipeline II = 1
                    reg = dataStrm.read();
                    reg_int = fToBits<float, unsigned int>(reg);
                    reg_apint = reg_int;
                    axi_qf[y * x4 * 16 + x * 16 + j * 4 + 2 + i] = reg_apint;
                }
            }
        }
    }
    reg = avgStrm.read();
    avg_outStrm.write(reg);
    reg_int = fToBits<float, unsigned int>(reg);
    reg_apint = reg_int;
    axi_qf[0] = reg_apint;
}

inline void QFwrite(int xsize,
                    int ysize,
                    hls::stream<float>& qfStrm,
                    hls::stream<float>& avgStrm,
                    hls::stream<float>& avg_outStrm,
                    ap_uint<32>* axi_qf) {
#pragma HLS dataflow
    hls::stream<float> dataStrm;
#pragma HLS STREAM variable = dataStrm depth = 32
    hls::stream<ap_uint<32> > addrStrm;
#pragma HLS STREAM variable = addrStrm depth = 32

    QFwriteCalAddr(xsize, ysize, qfStrm, dataStrm, addrStrm);

    QFwriteDDRCtrl(xsize, ysize, avgStrm, avg_outStrm, dataStrm, addrStrm, axi_qf);
}

inline void QFload(int xsize,
                   int ysize,
                   hls::stream<float>& avgStrm,
                   ap_uint<32>* axi_qf,
                   hls::stream<float>& qfStrm,
                   hls::stream<float>& avg_outStrm) {
    float reg;
    unsigned int reg_int;
    ap_uint<32> reg_apint;

    int x4 = (xsize + 4 - 1) / 4;
    int y4 = (ysize + 4 - 1) / 4;
    int n = x4 * y4;

    reg_apint = axi_qf[0];
    reg_int = reg_apint;
    reg = bitsToF<unsigned int, float>(reg_int);
    avg_outStrm.write(avgStrm.read());
    int cnt = 2;
    for (size_t x = 0; x < n; ++x) {
        for (int i = 0; i < 16; i++) {
#pragma HLS pipeline II = 1
            reg_apint = axi_qf[cnt];
            reg_int = reg_apint;
            reg = bitsToF<unsigned int, float>(reg_int);
            qfStrm.write(reg);
            cnt++;
        }
    }
}

inline void QFcalabsavgS1(
    int xsize, int ysize, hls::stream<float>& qfStrm, hls::stream<float>& avgStrm, hls::stream<float>& sum4Strm) {
    int x4 = (xsize + 4 - 1) / 4;
    int y4 = (ysize + 4 - 1) / 4;
    int n = x4 * y4 * 16;

    float reg[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    float sum = 0;
    int size8 = (x4 * y4 + 8 - 1) / 8;

    float avg = avgStrm.read();

    for (int y = 0; y < y4; y++) {
        for (int x = 0; x < x4; x++) {
            for (int j = 0; j < 2; j++) {
                for (int i = 0; i < 8; i++) {
#pragma HLS pipeline II = 1
                    reg[7] = reg[6];
                    reg[6] = reg[5];
                    reg[5] = reg[4];
                    reg[4] = reg[3];
                    reg[3] = reg[2];
                    reg[2] = reg[1];
                    reg[1] = reg[0];
                    if (x * 4 + i % 4 < xsize && y * 4 + j * 2 + i / 4 < ysize) {
                        reg[0] = hls::abs(qfStrm.read() - avg);
                    } else {
                        qfStrm.read();
                        reg[0] = 0;
                    }
                    sum = reg[0] + reg[1] + reg[2] + reg[3] + reg[4] + reg[5] + reg[6] + reg[7];
                    if (i == 7) sum4Strm.write(sum);
                }
            }
        }
    }
}

inline void QFcalabsavgS2(int xsize, int ysize, hls::stream<float>& sum4Strm, hls::stream<float>& absAvgStrm) {
    int x4 = (xsize + 4 - 1) / 4;
    int y4 = (ysize + 4 - 1) / 4;
    int n = x4 * y4 * 16;

    float sum = 0;
    int size8 = (n + 8 - 1) / 8;

    for (int x = 0; x < size8; x++) {
#pragma HLS pipeline II = 8
        sum = sum + sum4Strm.read();
    }

    absAvgStrm.write(sum / (xsize * ysize));
}

inline void QFWriteOutDataflow(
    int xsize, int ysize, hls::stream<float>& avgStrm, ap_uint<32>* axi_qf, hls::stream<float>& absAvgStrm) {
#pragma HLS dataflow

    hls::stream<float> qfloadStrm("qfloadStrm");
#pragma HLS STREAM variable = qfloadStrm depth = 32

    hls::stream<float> avgloadStrm("qfavgloadStrm");
#pragma HLS STREAM variable = avgloadStrm depth = 32

    hls::stream<float> sum4Strm("qfsum4Strm");
#pragma HLS STREAM variable = sum4Strm depth = 32

    QFload(xsize, ysize, avgStrm, axi_qf, qfloadStrm, avgloadStrm);

    QFcalabsavgS1(xsize, ysize, qfloadStrm, avgloadStrm, sum4Strm);

    QFcalabsavgS2(xsize, ysize, sum4Strm, absAvgStrm);
}

inline void absAvgWrite(hls::stream<float>& absAvgStrm, ap_uint<32>* axi_qf) {
    float reg = absAvgStrm.read();
    unsigned int reg_int = fToBits<float, unsigned int>(reg);
    ap_uint<32> reg_apint = reg_int;
    axi_qf[1] = reg_apint;
}

inline void QFWriteOut(
    int xsize, int ysize, hls::stream<float>& qfStrm, hls::stream<float>& avgStrm, ap_uint<32>* axi_qf) {
    hls::stream<float> avg_outStrm("avg_outStrm");
#pragma HLS STREAM variable = avg_outStrm depth = 2

    hls::stream<float> absAvgStrm("absAvgStrm");
#pragma HLS STREAM variable = absAvgStrm depth = 2

    QFwrite(xsize, ysize, qfStrm, avgStrm, avg_outStrm, axi_qf);

    QFWriteOutDataflow(xsize, ysize, avg_outStrm, axi_qf, absAvgStrm);

    absAvgWrite(absAvgStrm, axi_qf);
}

inline void hls_combineblock(int xsize, int ysize, hls::stream<float>& input, hls::stream<float>& output) {
    float tmp[2][24];
#pragma HLS ARRAY_PARTITION variable = tmp dim = 1
    bool ping = 0;

    for (int j = 0; j < 8; j++) {
        for (int c = 0; c < 3; c++) {
#pragma HLS PIPELINE II = 1
            tmp[ping][c * 8 + j] = input.read();
        }
    }
    ping = !ping;
    for (int i = 0; i < xsize * ysize / 8 - 1; i++) {
        for (int j = 0; j < 8; j++) {
            for (int c = 0; c < 3; c++) {
#pragma HLS PIPELINE II = 1
                tmp[ping][c * 8 + j] = input.read();
                output.write(tmp[!ping][3 * j + c]);
            }
        }
        ping = !ping;
    }

    for (int j = 0; j < 8; j++) {
        for (int c = 0; c < 3; c++) {
#pragma HLS PIPELINE II = 1
            output.write(tmp[!ping][3 * j + c]);
        }
    }
}

inline void hls_splitblock(int xsize, int ysize, hls::stream<float>& input, hls::stream<float>& output) {
    float tmp[2][24];
#pragma HLS ARRAY_PARTITION variable = tmp dim = 1
    bool ping = 0;

    for (int c = 0; c < 3; c++) {
        for (int j = 0; j < 8; j++) {
#pragma HLS PIPELINE II = 1
            tmp[ping][j * 3 + c] = input.read();
        }
    }
    ping = !ping;
    for (int i = 0; i < xsize * ysize / 8 - 1; i++) {
        for (int c = 0; c < 3; c++) {
            for (int j = 0; j < 8; j++) {
#pragma HLS PIPELINE II = 1
                tmp[ping][j * 3 + c] = input.read();
                output.write(tmp[!ping][c * 8 + j] / 64);
            }
        }
        ping = !ping;
    }

    for (int c = 0; c < 3; c++) {
        for (int j = 0; j < 8; j++) {
#pragma HLS PIPELINE II = 1
            output.write(tmp[!ping][c * 8 + j] / 64);
        }
    }
}

inline void hls_DCT8Core(float in[8], float out[8]) {
    const float c1 = 0.707106781186548f; // 1 / sqrt(2)
    const float c2 = 0.382683432365090f; // cos(3 * pi / 8)
    const float c3 = 1.30656296487638f;  // 1 / (2 * cos(3 * pi / 8))
    const float c4 = 0.541196100146197f; // sqrt(2) * cos(3 * pi / 8)

    float i0 = in[0]; // in[i][0];
    float i1 = in[1]; // in[i][1];
    float i2 = in[2];
    float i3 = in[3];
    float i4 = in[4];
    float i5 = in[5];
    float i6 = in[6];
    float i7 = in[7];

    const float t00 = i0 + i7;
    const float t01 = i0 - i7;
    const float t02 = i3 + i4;
    const float t03 = i3 - i4;
    const float t04 = i2 + i5;
    const float t05 = i2 - i5;
    const float t06 = i1 + i6;
    const float t07 = i1 - i6;
    const float t08 = t00 + t02;
    const float t09 = t00 - t02;
    const float t10 = t06 + t04;
    const float t11 = t06 - t04;
    const float t12 = t07 + t05;
    const float t13 = t01 + t07;
    const float t14 = t05 + t03;
    const float t15 = t11 + t09;
    const float t16 = t14 - t13;
    const float t17 = c1 * t15;
    const float t18 = c1 * t12;
    const float t19 = c2 * t16;
    const float t20 = t01 + t18;
    const float t21 = t01 - t18;
    const float t22 = c3 * t13 + t19;
    const float t23 = c4 * t14 + t19;
    out[0] = (t08 + t10);
    out[1] = (t20 + t22);
    out[2] = (t09 + t17);
    out[3] = (t21 - t23);
    out[4] = (t08 - t10);
    out[5] = (t21 + t23);
    out[6] = (t09 - t17);
    out[7] = (t20 - t22);
}

inline void hls_CmapDCT1D(int xsize, int ysize, hls::stream<float>& input, hls::stream<float>& output) {
    float from[8];
    float to[8];

    for (int i = 0; i < 3 * xsize * ysize / 8; i++) {
#pragma HLS PIPELINE II = 8
        from[0] = input.read();
        from[1] = input.read();
        from[2] = input.read();
        from[3] = input.read();
        from[4] = input.read();
        from[5] = input.read();
        from[6] = input.read();
        from[7] = input.read();
        hls_DCT8Core(from, to);
        output.write(to[0]);
        output.write(to[1]);
        output.write(to[2]);
        output.write(to[3]);
        output.write(to[4]);
        output.write(to[5]);
        output.write(to[6]);
        output.write(to[7]);
    }
}

inline void hls_transpose(int xsize, int ysize, hls::stream<float>& input, hls::stream<float>& output) {
#ifndef __SYNTHESIS__
    std::vector<float> linebuffer_ping(98304);
    std::vector<float> linebuffer_pong(98304);
#else
    float linebuffer_ping[98304];
#pragma HLS RESOURCE variable = linebuffer_ping core = RAM_S2P_URAM
    float linebuffer_pong[98304];
#pragma HLS RESOURCE variable = linebuffer_pong core = RAM_S2P_URAM
#endif

    bool ping = 0;

    for (int j = 0; j < 8; j++) {
        for (int i = 0; i < xsize / 8; i++) {
            for (int c = 0; c < 3; c++) {
                for (int k = 0; k < 8; k++) {
#pragma HLS PIPELINE II = 1
                    linebuffer_pong[i * 192 + k * 24 + c * 8 + j] = input.read();
                }
            }
        }
    }
    ping = !ping;

    for (int n = 0; n < ysize / 8 - 1; n++) {
        for (int j = 0; j < 8; j++) {
            for (int i = 0; i < xsize / 8; i++) {
                for (int c = 0; c < 3; c++) {
                    for (int k = 0; k < 8; k++) {
#pragma HLS PIPELINE II = 1
                        if (ping == 1) {
                            linebuffer_ping[i * 192 + k * 24 + c * 8 + j] = input.read();
                            output.write(linebuffer_pong[j * xsize * 3 + i * 24 + c * 8 + k]);
                        } else {
                            linebuffer_pong[i * 192 + k * 24 + c * 8 + j] = input.read();
                            output.write(linebuffer_ping[j * xsize * 3 + i * 24 + c * 8 + k]);
                        }
                    }
                }
            }
        }
        ping = !ping;
    }

    for (int j = 0; j < 8; j++) {
        for (int i = 0; i < xsize / 8; i++) {
            for (int c = 0; c < 3; c++) {
                for (int k = 0; k < 8; k++) {
#pragma HLS PIPELINE II = 1
                    if (ping == 1) {
                        output.write(linebuffer_pong[j * xsize * 3 + i * 24 + c * 8 + k]);
                    } else {
                        output.write(linebuffer_ping[j * xsize * 3 + i * 24 + c * 8 + k]);
                    }
                }
            }
        }
    }
}

inline void hls_dct2DCmap(int xsize, int ysize, hls::stream<float>& input, hls::stream<float>& output) {
#pragma HLS inline
    hls::stream<float> tmp1("tmp1");
#pragma HLS STREAM variable = tmp1 depth = 32
    hls::stream<float> tmp2("tmp2");
#pragma HLS STREAM variable = tmp2 depth = 32
    hls::stream<float> tmp3("tmp3");
#pragma HLS STREAM variable = tmp3 depth = 32
    hls::stream<float> tmp4("tmp4");
#pragma HLS STREAM variable = tmp4 depth = 32

    hls_combineblock(xsize, ysize, input, tmp1);
    hls_CmapDCT1D(xsize, ysize, tmp1, tmp2);
    hls_transpose(xsize, ysize, tmp2, tmp3);
    hls_CmapDCT1D(xsize, ysize, tmp3, tmp4);
    hls_splitblock(xsize, ysize, tmp4, output);
}

inline void FindIndexOfSumMaximum(const int* array, const size_t len, int* idx, int* sum) {
    int maxval = 0;
    int val = 0;
    int maxidx = 0;
    for (size_t i = 1; i < len; ++i) {
        val += array[i];
        if (val > maxval) {
            maxval = val;
            maxidx = i;
        }
    }
    *idx = maxidx;
    *sum = maxval;
}

inline void hls_FindBestCorrelationCntDataFlow(int xsize,
                                               int ysize,
                                               hls::stream<float>& input,
                                               hls::stream<float>& tilemaxStrm,
                                               hls::stream<ap_uint<8> >& tileidxStrm) {
    ap_uint<32> cur_y;

    ap_int<64> d_num_zerosx_add[2][64][256];
#pragma HLS RESOURCE variable = d_num_zerosx_add core = RAM_S2P_URAM
#pragma HLS ARRAY_PARTITION variable = d_num_zerosx_add complete dim = 1

    ap_int<64> d_num_zerosx_sub[2][64][256];
#pragma HLS RESOURCE variable = d_num_zerosx_sub core = RAM_S2P_URAM
#pragma HLS ARRAY_PARTITION variable = d_num_zerosx_sub complete dim = 1

    ap_int<64> d_num_zerosx_pre[2][64][256];
#pragma HLS RESOURCE variable = d_num_zerosx_pre core = RAM_S2P_URAM
#pragma HLS ARRAY_PARTITION variable = d_num_zerosx_pre complete dim = 1

    int ty;
    ty = (ysize + 64 - 1) / 64;

    for (int j = 0; j < (xsize + 64 - 1) / 64; j++) {
        for (int i = 0; i < 256; i++) {
#pragma HLS PIPELINE II = 1
            d_num_zerosx_add[0][j][i] = 0;
            d_num_zerosx_sub[0][j][i] = 0;
            d_num_zerosx_pre[0][j][i] = 0;
            d_num_zerosx_add[1][j][i] = 0;
            d_num_zerosx_sub[1][j][i] = 0;
            d_num_zerosx_pre[1][j][i] = 0;
        }
    }

    int SCALEX = 256;
    int OFFSETX = 128;
    int SCALEB = 128;
    int OFFSETB = 0;

    float acceptancex = -0.625f;
    float acceptanceb = 0.25f;

    float kZeroBiasDefault[3] = {0.65f, 0.6f, 0.7f};
    int N = 8;
    int block_size = N * N;
    float kScalex = SCALEX;
    float kScaleb = SCALEB;
    float kZeroThreshx = kScalex * kZeroBiasDefault[0];
    float kZeroThreshb = kScaleb * kZeroBiasDefault[2];
    size_t kColorTileDimInBlocks = 8;

    ap_uint<32> y_local;
    if (ysize - cur_y > 64)
        y_local = 64;
    else
        y_local = ysize - cur_y;

    ap_uint<9> ycnt = 0;
    ap_uint<15> xcnt = 0;
    ap_uint<6> txcnt = 0;
    ap_uint<3> bxcnt = 0;
    ap_uint<3> bycnt = 0;
    ap_uint<3> bycnt_r = 0;
    ap_uint<6> cnt64 = 0;

    ap_uint<6> sumtxcnt = 0;
    ap_uint<8> sumcnt256 = 0;
    ap_uint<32> sumcnt = 0;
    ap_uint<32> cnt = 0;
    bool ping = true;

    ap_uint<32> d_num_zerosx;
    ap_int<32> tilemax = 0;
    ap_uint<8> tileidx = 0;
    ap_int<16> tilereg = 0;
    ap_uint<32> apxsize = xsize;
    ap_uint<32> apysize = ysize;

    while (cnt < ysize * xsize && cnt < 64 * xsize) {
#pragma HLS PIPELINE II = 3
        if (cnt64 == 0) {
            float colorx = input.read();
            float colory = input.read();
            float colorb = input.read();
        } else {
            float colorx = input.read();
            float colory = input.read();
            float colorb = input.read();

            const float scaled_mx = colory * qmxlocal[cnt64];
            const float scaled_mb = colory * qmblocal[cnt64];
            const float scaled_sx = kScalex * colorx * qmxlocal[cnt64] + OFFSETX * scaled_mx;
            const float scaled_sb = kScaleb * colorb * qmblocal[cnt64] + OFFSETB * scaled_mb;

            // Increment num_zeros[idx] if
            //   std::abs(scaled_s - (idx - OFFSET) *
            //   scaled_m) < kZeroThresh
            if (hls::abs(scaled_mx) >= 1e-8) {
                float from;
                float to;
                if (scaled_mx > 0) {
                    from = (scaled_sx - kZeroThreshx) / scaled_mx;
                    to = (scaled_sx + kZeroThreshx) / scaled_mx;
                } else {
                    from = (scaled_sx + kZeroThreshx) / scaled_mx;
                    to = (scaled_sx - kZeroThreshx) / scaled_mx;
                }
                // Instead of clamping the both values
                // we just check that range is sane.
                if (from < 0.0f) {
                    from = 0.0f;
                }
                if (to > 255.0f) {
                    to = 255.0f;
                }
                if (from <= to) {
                    if (from < 255) {
                        d_num_zerosx_add[ping][txcnt][(int)std::ceil(from)]++;
                    }
                    if (to < 255) {
                        d_num_zerosx_sub[ping][txcnt][(int)std::floor(to + 1)]--;
                    }
                }
            }
        }

        bycnt_r = bycnt;
        if (xcnt != xsize * 8 - 1) {
            xcnt++;
        } else {
            xcnt = 0;
            bycnt++;
        }

        if (bycnt_r == 7 && bycnt == 0) {
            ping = !ping;
        }

        cnt64 = xcnt.range(5, 0);
        bxcnt = xcnt.range(8, 6);
        txcnt = xcnt.range(14, 9);
        cnt++;
    }

    while (cnt < ysize * xsize) {
#pragma HLS pipeline II = 3
        if (cnt64 == 0) {
            float colorx = input.read();
            float colory = input.read();
            float colorb = input.read();
        } else {
            float colorx = input.read();
            float colory = input.read();
            float colorb = input.read();

            const float scaled_mx = colory * qmxlocal[cnt64];
            const float scaled_mb = colory * qmblocal[cnt64];
            const float scaled_sx = kScalex * colorx * qmxlocal[cnt64] + OFFSETX * scaled_mx;
            const float scaled_sb = kScaleb * colorb * qmblocal[cnt64] + OFFSETB * scaled_mb;

            // Increment num_zeros[idx] if
            //   std::abs(scaled_s - (idx - OFFSET) *
            //   scaled_m) < kZeroThresh
            if (hls::abs(scaled_mx) >= 1e-8) {
                float from;
                float to;
                if (scaled_mx > 0) {
                    from = (scaled_sx - kZeroThreshx) / scaled_mx;
                    to = (scaled_sx + kZeroThreshx) / scaled_mx;
                } else {
                    from = (scaled_sx + kZeroThreshx) / scaled_mx;
                    to = (scaled_sx - kZeroThreshx) / scaled_mx;
                }
                // Instead of clamping the both values
                // we just check that range is sane.
                if (from < 0.0f) {
                    from = 0.0f;
                }
                if (to > 255.0f) {
                    to = 255.0f;
                }
                if (from <= to) {
                    if (from < 255) {
                        d_num_zerosx_add[ping][txcnt][(int)std::ceil(from)]++;
                    }
                    if (to < 255) {
                        d_num_zerosx_sub[ping][txcnt][(int)std::floor(to + 1)]--;
                    }
                }
            }
        }

        if (sumcnt < ((xsize + 64 - 1) / 64) * 256) {
            if (sumcnt256 != 0) {
                d_num_zerosx =
                    d_num_zerosx_add[!ping][sumtxcnt][sumcnt256] + d_num_zerosx_sub[!ping][sumtxcnt][sumcnt256];
                tilereg = tilereg + d_num_zerosx - d_num_zerosx_pre[!ping][sumtxcnt][sumcnt256];
                d_num_zerosx_pre[!ping][sumtxcnt][sumcnt256] = d_num_zerosx;
                if (tilemax < tilereg) {
                    tilemax = tilereg;
                    tileidx = sumcnt256;
                }
                if (sumcnt256 == 255 && sumcnt != ((xsize + 64 - 1) / 64) * 256 - 1) {
                    tilemaxStrm.write((float)tilemax / (64 * 64));
                    tileidxStrm.write(tileidx);
                    tilemax = 0;
                    tileidx = 0;
                    tilereg = 0;
                } else if (sumcnt256 == 255 && sumcnt == ((xsize + 64 - 1) / 64) * 256 - 1) {
                    ap_uint<32> tmp = apxsize.range(5, 0) == 0 ? 64 : apxsize.range(5, 0);
                    tilemaxStrm.write((float)tilemax / (64 * tmp));
                    tileidxStrm.write(tileidx);
                    tilemax = 0;
                    tileidx = 0;
                    tilereg = 0;
                }
            }
        }

        bycnt_r = bycnt;
        if (xcnt != xsize * 8 - 1) {
            xcnt++;
        } else {
            xcnt = 0;
            bycnt++;
        }

        if (bycnt_r == 7 && bycnt == 0) {
            ping = !ping;
            sumcnt = 0;
        } else {
            sumcnt++;
        }

        sumcnt256 = sumcnt.range(7, 0);
        sumtxcnt = sumcnt.range(13, 8);
        cnt64 = xcnt.range(5, 0);
        bxcnt = xcnt.range(8, 6);
        txcnt = xcnt.range(14, 9);
        cnt++;
    }

    if (bycnt != 0) {
        ping = !ping;
    }
    sumcnt = 0;
    sumcnt256 = sumcnt.range(7, 0);
    sumtxcnt = sumcnt.range(13, 8);

    while (sumcnt < ((xsize + 64 - 1) / 64) * 256) {
#pragma HLS PIPELINE II = 1

        if (sumcnt256 != 0) {
            d_num_zerosx = d_num_zerosx_add[!ping][sumtxcnt][sumcnt256] + d_num_zerosx_sub[!ping][sumtxcnt][sumcnt256];
            tilereg = tilereg + d_num_zerosx - d_num_zerosx_pre[!ping][sumtxcnt][sumcnt256];
            if (tilemax < tilereg) {
                tilemax = tilereg;
                tileidx = sumcnt256;
            }
            if (sumcnt256 == 255 && sumcnt != ((xsize + 64 - 1) / 64) * 256 - 1) {
                ap_uint<32> tmpy = apysize.range(5, 0) == 0 ? 64 : apysize.range(5, 0);
                tilemaxStrm.write((float)tilemax / (tmpy * 64));
                tileidxStrm.write(tileidx);
                tilemax = 0;
                tileidx = 0;
                tilereg = 0;
            } else if (sumcnt256 == 255 && sumcnt == ((xsize + 64 - 1) / 64) * 256 - 1) {
                ap_uint<32> tmpx = apxsize.range(5, 0) == 0 ? 64 : apxsize.range(5, 0);
                ap_uint<32> tmpy = apysize.range(5, 0) == 0 ? 64 : apysize.range(5, 0);
                tilemaxStrm.write((float)tilemax / (tmpy * tmpx));
                tileidxStrm.write(tileidx);
                tilemax = 0;
                tileidx = 0;
                tilereg = 0;
            }
        }
        sumcnt++;
        sumcnt256 = sumcnt.range(7, 0);
        sumtxcnt = sumcnt.range(13, 8);
    }
}

inline void hls_globalCnt(int xsize,
                          int ysize,
                          hls::stream<float>& input,
                          hls::stream<float>& globalxmaxStrm,
                          hls::stream<ap_uint<8> >& globalxidxStrm,
                          hls::stream<float>& globalbmaxStrm,
                          hls::stream<ap_uint<8> >& globalbidxStrm) {
    ap_int<32> d_num_zeros_globalx_add[256];
    ap_int<32> d_num_zeros_globalx_sub[256];
    ap_int<32> d_num_zeros_globalb_add[256];
    ap_int<32> d_num_zeros_globalb_sub[256];

    int SCALEX = 256;
    int OFFSETX = 128;
    int SCALEB = 128;
    int OFFSETB = 0;

    float acceptancex = -0.625f;
    float acceptanceb = 0.25f;

    float kZeroBiasDefault[3] = {0.65f, 0.6f, 0.7f};
    int N = 8;
    int block_size = N * N;
    float kScalex = SCALEX;
    float kScaleb = SCALEB;
    float kZeroThreshx = kScalex * kZeroBiasDefault[0];
    float kZeroThreshb = kScaleb * kZeroBiasDefault[2];
    size_t kColorTileDimInBlocks = 8;

    ap_uint<6> cnt64 = 0;
    ap_uint<32> cnt = 0;

    for (int i = 0; i < 256; i++) {
#pragma HLS pipeline ii = 1
        d_num_zeros_globalx_add[i] = 0;
        d_num_zeros_globalx_sub[i] = 0;
        d_num_zeros_globalb_add[i] = 0;
        d_num_zeros_globalb_sub[i] = 0;
    }

    while (cnt < xsize * ysize) {
#pragma HLS pipeline II = 3
        if (cnt64 == 0) {
            input.read();
            input.read();
            input.read();
        } else {
            float colorx = input.read();
            float colory = input.read();
            float colorb = input.read();

            const float scaled_mx = colory * qmxglb[cnt64];
            const float scaled_mb = colory * qmbglb[cnt64];
            const float scaled_sx = kScalex * colorx * qmxglb[cnt64] + OFFSETX * scaled_mx;
            const float scaled_sb = kScaleb * colorb * qmbglb[cnt64] + OFFSETB * scaled_mb;

            // Increment num_zeros[idx] if
            //   std::abs(scaled_s - (idx - OFFSET) *
            //   scaled_m) < kZeroThresh
            if (hls::abs(scaled_mx) >= 1e-8) {
                float from;
                float to;
                if (scaled_mx > 0) {
                    from = (scaled_sx - kZeroThreshx) / scaled_mx;
                    to = (scaled_sx + kZeroThreshx) / scaled_mx;
                } else {
                    from = (scaled_sx + kZeroThreshx) / scaled_mx;
                    to = (scaled_sx - kZeroThreshx) / scaled_mx;
                }
                // Instead of clamping the both values
                // we just check that range is sane.
                if (from < 0.0f) {
                    from = 0.0f;
                }
                if (to > 255.0f) {
                    to = 255.0f;
                }
                if (from <= to) {
                    if (from < 255) {
                        d_num_zeros_globalx_add[(int)std::ceil(from)]++;
                    }
                    if (to < 255) {
                        d_num_zeros_globalx_sub[(int)std::floor(to + 1)]--;
                    }
                }
            }

            if (hls::abs(scaled_mb) >= 1e-8) {
                float from;
                float to;
                if (scaled_mb > 0) {
                    from = (scaled_sb - kZeroThreshb) / scaled_mb;
                    to = (scaled_sb + kZeroThreshb) / scaled_mb;
                } else {
                    from = (scaled_sb + kZeroThreshb) / scaled_mb;
                    to = (scaled_sb - kZeroThreshb) / scaled_mb;
                }
                // Instead of clamping the both values
                // we just check that range is sane.
                if (from < 0.0f) {
                    from = 0.0f;
                }
                if (to > 255.0f) {
                    to = 255.0f;
                }
                if (from <= to) {
                    if (from < 255) {
                        d_num_zeros_globalb_add[(int)std::ceil(from)]++;
                    }
                    if (to < 255) {
                        d_num_zeros_globalb_sub[(int)std::floor(to + 1)]--;
                    }
                }
            }
        }
        cnt++;
        cnt64 = cnt.range(5, 0);
    }

    ap_int<32> glbbmax;
    ap_uint<8> glbbidx;
    ap_int<32> glbxmax;
    ap_uint<8> glbxidx;
    ap_int<32> glbbreg;
    ap_int<32> glbxreg;

    glbbmax = 0;
    glbbidx = 0;
    glbxmax = 0;
    glbxidx = 0;
    glbbreg = 0;
    glbxreg = 0;

    for (int i = 1; i < 256; ++i) {
#pragma HLS pipeline ii = 1
        glbbreg = glbbreg + d_num_zeros_globalb_add[i] + d_num_zeros_globalb_sub[i];
        if (glbbmax < glbbreg) {
            glbbmax = glbbreg;
            glbbidx = i;
        }

        glbxreg = glbxreg + d_num_zeros_globalx_add[i] + d_num_zeros_globalx_sub[i];
        if (glbxmax < glbxreg) {
            glbxmax = glbxreg;
            glbxidx = i;
        }
    }

    globalxmaxStrm.write((float)glbxmax / (xsize * ysize));
    globalxidxStrm.write(glbxidx);
    globalbmaxStrm.write((float)glbbmax / (xsize * ysize));
    globalbidxStrm.write(glbbidx);
}

inline void hls_FindBestCorrelationStore(int xsize,
                                         int ysize,
                                         hls::stream<float>& tilemaxStrm,
                                         hls::stream<ap_uint<8> >& tileidxStrm,
                                         hls::stream<float>& globalxmaxStrm,
                                         hls::stream<ap_uint<8> >& globalxidxStrm,
                                         hls::stream<float>& globalbmaxStrm,
                                         hls::stream<ap_uint<8> >& globalbidxStrm,
                                         float tilemax[64][64],
                                         ap_uint<8> tileidx[64][64],
                                         float glbxmax[1],
                                         ap_uint<8> glbxidx[1],
                                         float glbbmax[1],
                                         ap_uint<8> glbbidx[1]) {
    int tx = (xsize + 64 - 1) / 64;
    int ty = (ysize + 64 - 1) / 64;

    for (int j = 0; j < ty; j++) {
        for (int i = 0; i < tx; i++) {
#pragma HLS pipeline ii = 1
            tilemax[j][i] = tilemaxStrm.read();
            tileidx[j][i] = tileidxStrm.read();
        }
    }

    glbxmax[0] = globalxmaxStrm.read();
    glbxidx[0] = globalxidxStrm.read();
    glbbmax[0] = globalbmaxStrm.read();
    glbbidx[0] = globalbidxStrm.read();
}

inline void duplicate(
    int xsize, int ysize, hls::stream<float>& input, hls::stream<float>& output1, hls::stream<float>& output2) {
    for (int c = 0; c < 3; c++) {
        for (int i = 0; i < xsize; i++) {
            for (int j = 0; j < ysize; j++) {
                float reg = input.read();
                output1.write(reg);
                output2.write(reg);
            }
        }
    }
}

inline void hls_FindBestCorrelationDataFlow(int xsize,
                                            int ysize,
                                            hls::stream<float>& input,
                                            float tilemax[64][64],
                                            ap_uint<8> tileidx[64][64],
                                            float glbxmax[1],
                                            ap_uint<8> glbxidx[1],
                                            float glbbmax[1],
                                            ap_uint<8> glbbidx[1]) {
#pragma HLS DATAFLOW
    hls::stream<float> output1;
#pragma HLS STREAM variable = output1 depth = 32

    hls::stream<float> output2("cmap output2");
#pragma HLS STREAM variable = output2 depth = 32

    hls::stream<float> globalxmaxStrm;
#pragma HLS STREAM variable = globalxmaxStrm depth = 2

    hls::stream<ap_uint<8> > globalxidxStrm;
#pragma HLS STREAM variable = globalxidxStrm depth = 2

    hls::stream<float> globalbmaxStrm;
#pragma HLS STREAM variable = globalbmaxStrm depth = 2

    hls::stream<ap_uint<8> > globalbidxStrm;
#pragma HLS STREAM variable = globalbidxStrm depth = 2

    hls::stream<float> tilemaxStrm("cmap tilemaxStrm");
#pragma HLS STREAM variable = tilemaxStrm depth = 32

    hls::stream<ap_uint<8> > tileidxStrm("cmap tileidxStrm");
#pragma HLS STREAM variable = tileidxStrm depth = 32

    duplicate(xsize, ysize, input, output1, output2);
    hls_globalCnt(xsize, ysize, output1, globalxmaxStrm, globalxidxStrm, globalbmaxStrm, globalbidxStrm);

    hls_FindBestCorrelationCntDataFlow(xsize, ysize, output2, tilemaxStrm, tileidxStrm);

    hls_FindBestCorrelationStore(xsize, ysize, tilemaxStrm, tileidxStrm, globalxmaxStrm, globalxidxStrm, globalbmaxStrm,
                                 globalbidxStrm, tilemax, tileidx, glbxmax, glbxidx, glbbmax, glbbidx);
}

inline void hls_FindBestCorrelationforward(int xsize,
                                           int ysize,
                                           float tilemax[64][64],
                                           ap_uint<8> tileidx[64][64],
                                           float glbxmax[1],
                                           ap_uint<8> glbxidx[1],
                                           float glbbmax[1],
                                           ap_uint<8> glbbidx[1],

                                           hls::stream<ap_uint<8> >& cmapxStrm,
                                           hls::stream<ap_uint<8> >& dcxStrm,
                                           hls::stream<ap_uint<8> >& cmapbStrm,
                                           hls::stream<ap_uint<8> >& dcbStrm) {
    int tx = (xsize + 64 - 1) / 64;
    int ty = (ysize + 64 - 1) / 64;

    float acceptancex = -0.625f;
    int N = 8;
    int block_size = N * N;
    size_t kColorTileDimInBlocks = 8;

    float global_normalized_sumx = (float)glbxmax[0] / (xsize * ysize);
    float normalized_acceptance = acceptancex * kColorTileDimInBlocks * kColorTileDimInBlocks * block_size;

    dcxStrm.write(glbxidx[0]);
    dcbStrm.write(glbbidx[0]);
    for (int j = 0; j < ty; j++) {
        for (int i = 0; i < tx; i++) {
#pragma HLS pipeline ii = 1
            if (tilemax[j][i] <= normalized_acceptance + global_normalized_sumx)
                cmapxStrm.write(glbxidx[0]);
            else
                cmapxStrm.write(tileidx[j][i]);
            cmapbStrm.write(glbbidx[0]);
        }
    }
}

inline void hls_FindBestCorrelation_v2(int xsize,
                                       int ysize,
                                       hls::stream<float>& input,
                                       hls::stream<ap_uint<8> >& cmapxStrm,
                                       hls::stream<ap_uint<8> >& dcxStrm,
                                       hls::stream<ap_uint<8> >& cmapbStrm,
                                       hls::stream<ap_uint<8> >& dcbStrm) {
    float tilemax[64][64];
    ap_uint<8> tileidx[64][64];
    float glbxmax[1];
    ap_uint<8> glbxidx[1];
    float glbbmax[1];
    ap_uint<8> glbbidx[1];

    hls_FindBestCorrelationDataFlow(xsize, ysize, input, tilemax, tileidx, glbxmax, glbxidx, glbbmax, glbbidx);

    hls_FindBestCorrelationforward(xsize, ysize, tilemax, tileidx, glbxmax, glbxidx, glbbmax, glbbidx, cmapxStrm,
                                   dcxStrm, cmapbStrm, dcbStrm);
}

inline void k1XYBCalAddr(int xsize,
                         int ysize,
                         hls::stream<float>& input,
                         hls::stream<float>& dataStrm,
                         hls::stream<ap_uint<32> >& addrStrm) {
    int x32 = (xsize + 32 - 1) / 32;
    int y32 = (ysize + 32 - 1) / 32;
    float xyb[2][96];
#pragma HLS ARRAY_PARTITION variable = xyb complete dim = 0
    bool ping = 0;

    for (int y = 0; y < y32; y++) {
        for (int j = 0; j < 32; j++) {
            for (int x = 0; x < x32; x++) {
                for (int i = 0; i < 32; i++) {
                    for (int c = 0; c < 3; c++) {
#pragma HLS pipeline II = 1
                        if ((x * 32 + i) < xsize && (y * 32 + j) < ysize) {
                            xyb[ping][i + c * 32] = input.read();
                        }
                        dataStrm.write(xyb[!ping][i * 3 + c]);
                    }
                }
                ping = !ping;
            }
        }
    }

    for (int i = 0; i < 32; i++) {
        for (int c = 0; c < 3; c++) {
#pragma HLS pipeline II = 1
            dataStrm.write(xyb[!ping][i * 3 + c]);
        }
    }
}

inline void k1XYBDDRCtrl(int xsize,
                         int ysize,
                         hls::stream<float>& dataStrm,
                         hls::stream<ap_uint<32> >& addrStrm,
                         ap_uint<32> axi_out[AXI_OUT]) {
    float reg;
    unsigned int reg_int;
    ap_uint<32> reg_apint;

    int x32 = (xsize + 32 - 1) / 32;
    int y32 = (ysize + 32 - 1) / 32;
    int n = x32 * 32 * y32;
    ap_uint<32> addr;

    for (int i = 0; i < 32; i++) {
        for (int c = 0; c < 3; c++) {
#pragma HLS pipeline II = 1
            dataStrm.read();
        }
    }

    for (int y = 0; y < y32; y++) {
        for (int j = 0; j < 32; j++) {
            for (int x = 0; x < x32; x++) {
                for (int c = 0; c < 3; c++) {
                    for (int i = 0; i < 32; i++) {
#pragma HLS pipeline II = 1
                        reg = dataStrm.read();
                        reg_int = fToBits<float, unsigned int>(reg);
                        reg_apint = reg_int;
                        axi_out[c * x32 * y32 * 1024 + y * x32 * 1024 + x * 1024 + j * 32 + i] = reg_apint;
                    }
                }
            }
        }
    }
}

inline void k1XYBWriteOut(int xsize, int ysize, hls::stream<float>& input, ap_uint<32> axi_out[AXI_OUT]) {
#pragma HLS inline

    hls::stream<float> dataStrm;
#pragma HLS STREAM variable = dataStrm depth = 32
    hls::stream<ap_uint<32> > addrStrm;
#pragma HLS STREAM variable = addrStrm depth = 32

    k1XYBCalAddr(xsize, ysize, input, dataStrm, addrStrm);

    k1XYBDDRCtrl(xsize, ysize, dataStrm, addrStrm, axi_out);
}

inline void k1CmapWriteOut(int xsize,
                           int ysize,
                           hls::stream<ap_uint<8> >& cmapxStrm,
                           hls::stream<ap_uint<8> >& dcxStrm,
                           hls::stream<ap_uint<8> >& cmapbStrm,
                           hls::stream<ap_uint<8> >& dcbStrm,
                           ap_uint<32> axi_cmap[AXI_CMAP]) {
    float reg;
    unsigned int reg_int;

    int cntCmap;
    cntCmap = 2;
    int x64 = (xsize + 64 - 1) / 64;
    int y64 = (ysize + 64 - 1) / 64;
    for (int i = 0; i < y64; i++) {
        for (int j = 0; j < x64; j++) {
#pragma HLS pipeline ii = 2
            axi_cmap[cntCmap] = cmapxStrm.read();
            cntCmap++;
            axi_cmap[cntCmap] = cmapbStrm.read();
            cntCmap++;
        }
    }
    axi_cmap[0] = dcxStrm.read();
    axi_cmap[1] = dcbStrm.read();
}

inline void dupxyb(
    int xsize, int ysize, hls::stream<float> xybGabStrm[3], hls::stream<float>& dctin, hls::stream<float>& output) {
    for (int y = 0; y < ysize; y++) {
        for (int x = 0; x < xsize; x++) {
#pragma HLS pipeline II = 3
            float reg = xybGabStrm[0].read();
            dctin.write(reg);
            output.write(reg);
            reg = xybGabStrm[1].read();
            dctin.write(reg);
            output.write(reg);
            reg = xybGabStrm[2].read();
            dctin.write(reg);
            output.write(reg);
        }
    }
}

inline void loadToStrm(
    int xsize, int ysize, hls::stream<DT> ostrm[3], hls::stream<bool> e_ostrm[3], hls::stream<float> rgbStrm[3]) {
    for (size_t y = 0; y < ysize; ++y) {
        for (size_t x = 0; x < xsize; x++) {
#pragma HLS pipeline II = 1
            e_ostrm[0].read();
            e_ostrm[1].read();
            e_ostrm[2].read();
            rgbStrm[0].write(bitsToF<int, float>(ostrm[0].read()));
            rgbStrm[1].write(bitsToF<int, float>(ostrm[1].read()));
            rgbStrm[2].write(bitsToF<int, float>(ostrm[2].read()));
        }
    }

    e_ostrm[0].read();
    e_ostrm[1].read();
    e_ostrm[2].read();
}

inline void kernel1_core(ap_uint<AXI_WIDTH> rbuf[BUF_DEPTH / 2],
                         const int len[3],
                         const int offsets[3],
                         int xsize,
                         int ysize,
                         float quant_ac,
                         ap_uint<32> axi_out[AXI_OUT],
                         ap_uint<32> axi_cmap[AXI_CMAP],
                         ap_uint<32> axi_qf[AXI_QF]) {
#pragma HLS INLINE off
#pragma HLS DATAFLOW

    hls::stream<DT> ostrm[3];
#pragma HLS STREAM variable = ostrm depth = 1024
#pragma HLS ARRAY_PARTITION variable = ostrm complete dim = 0
#pragma HLS RESOURCE variable = ostrm core = FIFO_BRAM

    hls::stream<bool> e_ostrm[3];
#pragma HLS STREAM variable = e_ostrm depth = 32
#pragma HLS ARRAY_PARTITION variable = e_ostrm complete dim = 0
#pragma HLS RESOURCE variable = e_ostrm core = FIFO_LUTRAM

    hls::stream<float> rgbStrm[3];
#pragma HLS STREAM variable = rgbStrm depth = 32
#pragma HLS ARRAY_PARTITION variable = rgbStrm complete dim = 0
#pragma HLS RESOURCE variable = rgbStrm core = FIFO_LUTRAM

    hls::stream<float> xybStrm[3];
#pragma HLS STREAM variable = xybStrm depth = 32
#pragma HLS ARRAY_PARTITION variable = xybStrm complete dim = 0
#pragma HLS RESOURCE variable = xybStrm core = FIFO_LUTRAM

    hls::stream<float> xybGabStrm[3];
#pragma HLS STREAM variable = xybGabStrm depth = 32
#pragma HLS ARRAY_PARTITION variable = xybGabStrm complete dim = 0
#pragma HLS RESOURCE variable = xybGabStrm core = FIFO_LUTRAM

    hls::stream<float> yOrigStrm("yorig");
#pragma HLS STREAM variable = yOrigStrm depth = 32
#pragma HLS RESOURCE variable = yOrigStrm core = FIFO_LUTRAM

    hls::stream<float> qfinStrm;
#pragma HLS STREAM variable = qfinStrm depth = 32
#pragma HLS RESOURCE variable = qfinStrm core = FIFO_LUTRAM

    hls::stream<float> qfStrm;
#pragma HLS STREAM variable = qfStrm depth = 32
#pragma HLS RESOURCE variable = qfStrm core = FIFO_LUTRAM

    hls::stream<float> avgStrm;
#pragma HLS STREAM variable = avgStrm depth = 32
#pragma HLS RESOURCE variable = avgStrm core = FIFO_LUTRAM

    hls::stream<ap_uint<8> > cmapxStrm("cmapxStrm");
#pragma HLS STREAM variable = cmapxStrm depth = 32
#pragma HLS RESOURCE variable = cmapxStrm core = FIFO_LUTRAM

    hls::stream<ap_uint<8> > dcxStrm("dcxStrm");
#pragma HLS STREAM variable = dcxStrm depth = 32
#pragma HLS RESOURCE variable = dcxStrm core = FIFO_LUTRAM

    hls::stream<ap_uint<8> > cmapbStrm("cmapbStrm");
#pragma HLS STREAM variable = cmapbStrm depth = 32
#pragma HLS RESOURCE variable = cmapbStrm core = FIFO_LUTRAM

    hls::stream<ap_uint<8> > dcbStrm("dcbStrm");
#pragma HLS STREAM variable = dcbStrm depth = 32
#pragma HLS RESOURCE variable = dcbStrm core = FIFO_LUTRAM

    hls::stream<float> dctin("dctin");
#pragma HLS STREAM variable = dctin depth = 32
#pragma HLS RESOURCE variable = dctin core = FIFO_LUTRAM

    hls::stream<float> dctout("dctout");
#pragma HLS STREAM variable = dctout depth = 32
#pragma HLS RESOURCE variable = dctout core = FIFO_LUTRAM

    hls::stream<float> output("output");
#pragma HLS STREAM variable = output depth = 32
#pragma HLS RESOURCE variable = output core = FIFO_LUTRAM

    static const int kResolution = 8;
    const size_t out_xsize = (xsize + kResolution - 1) / kResolution;
    const size_t out_ysize = (ysize + kResolution - 1) / kResolution;

    xf::common::utils_hw::axiToMultiStream<1024, AXI_WIDTH, DT, DT, DT>(rbuf, ostrm[0], e_ostrm[0], ostrm[1],
                                                                        e_ostrm[1], ostrm[2], e_ostrm[2], len, offsets);

    loadToStrm(xsize, ysize, ostrm, e_ostrm, rgbStrm);

    hls_OpsinDynamicsImage(rgbStrm, xsize, ysize, xybStrm, yOrigStrm);

    hls_GaborishInverse(xybStrm, xsize, ysize, xybGabStrm);

    initQFStrm(xsize, ysize, 0.11883287948847132, quant_ac, yOrigStrm, qfStrm, avgStrm);

    QFWriteOut(out_xsize, out_ysize, qfStrm, avgStrm, axi_qf);

    dupxyb(out_xsize * 8, // opsin.xsize(),
           out_ysize * 8, // opsin.ysize(),
           xybGabStrm, dctin, output);

    hls_dct2DCmap(out_xsize * 8, // opsin.xsize(),
                  out_ysize * 8, // opsin.ysize(),
                  dctin, dctout);

    hls_FindBestCorrelation_v2(out_xsize * 8, // opsin.xsize(),
                               out_ysize * 8, // opsin.ysize(),
                               dctout, cmapxStrm, dcxStrm, cmapbStrm, dcbStrm);

    k1CmapWriteOut(out_xsize * 8, // opsin.xsize(),
                   out_ysize * 8, // opsin.ysize(),
                   cmapxStrm, dcxStrm, cmapbStrm, dcbStrm, axi_cmap);

    k1XYBWriteOut(out_xsize * 8, // opsin.xsize(),
                  out_ysize * 8, // opsin.ysize(),
                  output, axi_out);
}

inline void loadConfig(
    ap_uint<32> config[MAX_NUM_CONFIG], int len[3], int offsets[3], int& xsize, int& ysize, float& quant_ac) {
#pragma HLS INLINE off

    len[0] = config[0];
    len[1] = config[1];
    len[2] = config[2];

    offsets[0] = config[3];
    offsets[1] = config[4];
    offsets[2] = config[5];

    xsize = config[6];
    ysize = config[7];
    int32_t quant_ac_tmp = config[8];
    quant_ac = bitsToF<int32_t, float>(quant_ac_tmp);
}

extern "C" void kernel1Top(ap_uint<32> config[MAX_NUM_CONFIG],
                           ap_uint<AXI_WIDTH> rbuf[BUF_DEPTH / 2],
                           ap_uint<32> axi_out[AXI_OUT],
                           ap_uint<32> axi_cmap[AXI_CMAP],
                           ap_uint<32> axi_qf[AXI_QF]);

#endif //_XF_CODEC_XACCPIKKERNEL1_HPP_
