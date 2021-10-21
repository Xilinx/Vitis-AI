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

#include "hls_stream.h"
#include "ap_int.h"
#include "common/xf_common.hpp"

namespace xf {
namespace cv {

#define __MAXVAL(pixeldepth) ((1 << pixeldepth) - 1)

typedef ap_ufixed<9, 1> _FIXED_LUT_TYPE;
typedef ap_ufixed<16, 6> _FIXED_PIXEL_TYPE;
typedef ap_ufixed<9, 8> _FIXED_OUT_PIXEL_TYPE;

typedef struct _cube {
    _FIXED_LUT_TYPE P000;
    _FIXED_LUT_TYPE P001;
    _FIXED_LUT_TYPE P010;
    _FIXED_LUT_TYPE P011;
    _FIXED_LUT_TYPE P100;
    _FIXED_LUT_TYPE P101;
    _FIXED_LUT_TYPE P110;
    _FIXED_LUT_TYPE P111;
} cube;

typedef struct _index {
    unsigned short R;
    unsigned short G;
    unsigned short B;
} pIndex;

/* Linear interpolation */
template <int T = 0>
_FIXED_PIXEL_TYPE interp1(_FIXED_PIXEL_TYPE val1, _FIXED_PIXEL_TYPE val2, _FIXED_PIXEL_TYPE val) {
#pragma HLS INLINE OFF
    _FIXED_PIXEL_TYPE ret = val1 + val * (val2 - val1);
    return ret;
}

/* Tri-linear interpolation */
template <int T = 0>
_FIXED_PIXEL_TYPE interp3(cube vertix, _FIXED_PIXEL_TYPE dist_r, _FIXED_PIXEL_TYPE dist_g, _FIXED_PIXEL_TYPE dist_b) {
#pragma HLS INLINE OFF
    _FIXED_PIXEL_TYPE a = interp1(vertix.P000, vertix.P100, dist_r);
    _FIXED_PIXEL_TYPE b = interp1(vertix.P001, vertix.P101, dist_r);
    _FIXED_PIXEL_TYPE c = interp1(vertix.P010, vertix.P110, dist_r);
    _FIXED_PIXEL_TYPE d = interp1(vertix.P011, vertix.P111, dist_r);

    _FIXED_PIXEL_TYPE e = interp1(a, b, dist_g);
    _FIXED_PIXEL_TYPE f = interp1(c, d, dist_g);

    _FIXED_PIXEL_TYPE g = interp1(e, f, dist_b);

    return g;
}

/**
 * 3DLUT kernel : Applies the given 3dlut on the input image using trilinear interpolation
 * in_img       : input xf::cv::mat
 * lut		: input lut taken as xf::cv::mat
 * out_img      : output xf::cv::mat
 * lutdim	: size of one of the dimensions of the 3d lut.
 */

template <int LUTDIM, int SQLUTDIM, int INTYPE, int OUTTYPE, int ROWS, int COLS, int NPPC = 1, int URAM = 0>
void lut3d(xf::cv::Mat<INTYPE, ROWS, COLS, NPPC>& in_img,
           xf::cv::Mat<XF_32FC3, SQLUTDIM, LUTDIM, NPPC>& lut,
           xf::cv::Mat<OUTTYPE, ROWS, COLS, NPPC>& out_img,
           unsigned char lutdim) {
#ifndef __SYNTHESIS__
    assert(((COLS >= in_img.cols) && (ROWS >= in_img.rows)) &&
           "ROWS and COLS values should be greater than input image rows and columns");
    assert((lutdim <= LUTDIM) && "LUT dimensions should be greater than or equal to lutdim value");
    assert((SQLUTDIM == LUTDIM * LUTDIM) && "SQLUTDIM value should be equal to LUTDIM*LUTDIM");
    assert((INTYPE == XF_8UC3) || (OUTTYPE == XF_8UC3) || (INTYPE == XF_10UC3) || (OUTTYPE == XF_10UC3) ||
           (INTYPE == XF_12UC3) || (OUTTYPE == XF_12UC3) || (INTYPE == XF_16UC3) ||
           (OUTTYPE == XF_16UC3) && "Only XF_8UC3, XF_10UC3, XF_12UC3, XF_16UC3 types are supported");
    assert((NPPC == 1) && "Only 1 pixel parallelism (NPPC=1) is supported");
#endif

#pragma HLS INLINE OFF

    _FIXED_LUT_TYPE lutGrid_r[LUTDIM - 1][LUTDIM - 1][LUTDIM - 1];
    _FIXED_LUT_TYPE lutGrid_g[LUTDIM - 1][LUTDIM - 1][LUTDIM - 1];
    _FIXED_LUT_TYPE lutGrid_b[LUTDIM - 1][LUTDIM - 1][LUTDIM - 1];

    _FIXED_LUT_TYPE borderLutRX[LUTDIM][LUTDIM];
    _FIXED_LUT_TYPE borderLutRY[LUTDIM][LUTDIM - 1];
    _FIXED_LUT_TYPE borderLutRZ[LUTDIM - 1][LUTDIM - 1];

    _FIXED_LUT_TYPE borderLutBX[LUTDIM][LUTDIM];
    _FIXED_LUT_TYPE borderLutBY[LUTDIM][LUTDIM - 1];
    _FIXED_LUT_TYPE borderLutBZ[LUTDIM - 1][LUTDIM - 1];

    _FIXED_LUT_TYPE borderLutGX[LUTDIM][LUTDIM];
    _FIXED_LUT_TYPE borderLutGY[LUTDIM - 1][LUTDIM - 1];
    _FIXED_LUT_TYPE borderLutGZ[LUTDIM - 1][LUTDIM - 1];

    cube cubeBufferR, cubeBufferG, cubeBufferB;

    pIndex pixelIndex;

    if (URAM) {
// clang-format off
#pragma HLS bind_storage variable=lutGrid_r type=ram_t2p impl=uram
#pragma HLS bind_storage variable=lutGrid_g type=ram_t2p impl=uram
#pragma HLS bind_storage variable=lutGrid_b type=ram_t2p impl=uram

#pragma HLS bind_storage variable=borderLutRX type=RAM_T2P impl=uram
#pragma HLS bind_storage variable=borderLutRY type=RAM_T2P impl=uram
#pragma HLS bind_storage variable=borderLutRZ type=RAM_T2P impl=uram

#pragma HLS bind_storage variable=borderLutBX type=RAM_T2P impl=uram
#pragma HLS bind_storage variable=borderLutBY type=RAM_T2P impl=uram
#pragma HLS bind_storage variable=borderLutBZ type=RAM_T2P impl=uram

#pragma HLS bind_storage variable=borderLutGX type=RAM_T2P impl=uram
#pragma HLS bind_storage variable=borderLutGY type=RAM_T2P impl=uram
#pragma HLS bind_storage variable=borderLutGZ type=RAM_T2P impl=uram
        // clang-format on
    } else {
// clang-format off
#pragma HLS bind_storage variable=lutGrid_r type=RAM_T2P impl=bram
#pragma HLS bind_storage variable=lutGrid_g type=RAM_T2P impl=bram
#pragma HLS bind_storage variable=lutGrid_b type=RAM_T2P impl=bram

#pragma HLS bind_storage variable=borderLutRX type=RAM_T2P impl=bram
#pragma HLS bind_storage variable=borderLutRY type=RAM_T2P impl=bram
#pragma HLS bind_storage variable=borderLutRZ type=RAM_T2P impl=bram

#pragma HLS bind_storage variable=borderLutBX type=RAM_T2P impl=bram
#pragma HLS bind_storage variable=borderLutBY type=RAM_T2P impl=bram
#pragma HLS bind_storage variable=borderLutBZ type=RAM_T2P impl=bram

#pragma HLS bind_storage variable=borderLutGX type=RAM_T2P impl=bram
#pragma HLS bind_storage variable=borderLutGY type=RAM_T2P impl=bram
#pragma HLS bind_storage variable=borderLutGZ type=RAM_T2P impl=bram
        // clang-format on
    }
// clang-format off
#pragma HLS ARRAY_PARTITION variable=lutGrid_r dim=1 cyclic factor=2
#pragma HLS ARRAY_PARTITION variable=lutGrid_g dim=1 cyclic factor=2
#pragma HLS ARRAY_PARTITION variable=lutGrid_b dim=1 cyclic factor=2

#pragma HLS ARRAY_PARTITION variable=lutGrid_r dim=2 cyclic factor=2
#pragma HLS ARRAY_PARTITION variable=lutGrid_g dim=2 cyclic factor=2
#pragma HLS ARRAY_PARTITION variable=lutGrid_b dim=2 cyclic factor=2

#pragma HLS ARRAY_PARTITION variable=lutGrid_r dim=3 cyclic factor=2
#pragma HLS ARRAY_PARTITION variable=lutGrid_g dim=3 cyclic factor=2
#pragma HLS ARRAY_PARTITION variable=lutGrid_b dim=3 cyclic factor=2

#pragma HLS ARRAY_PARTITION variable=borderLutRX dim=1 cyclic factor=2
#pragma HLS ARRAY_PARTITION variable=borderLutRY dim=1 cyclic factor=2
#pragma HLS ARRAY_PARTITION variable=borderLutRZ dim=1 cyclic factor=2

#pragma HLS ARRAY_PARTITION variable=borderLutGX dim=1 cyclic factor=2
#pragma HLS ARRAY_PARTITION variable=borderLutGY dim=1 cyclic factor=2
#pragma HLS ARRAY_PARTITION variable=borderLutGZ dim=1 cyclic factor=2

#pragma HLS ARRAY_PARTITION variable=borderLutBX dim=1 cyclic factor=2
#pragma HLS ARRAY_PARTITION variable=borderLutBY dim=1 cyclic factor=2
#pragma HLS ARRAY_PARTITION variable=borderLutBZ dim=1 cyclic factor=2

#pragma HLS ARRAY_PARTITION variable=borderLutRX dim=2 cyclic factor=2
#pragma HLS ARRAY_PARTITION variable=borderLutRY dim=2 cyclic factor=2
#pragma HLS ARRAY_PARTITION variable=borderLutRZ dim=2 cyclic factor=2

#pragma HLS ARRAY_PARTITION variable=borderLutGX dim=2 cyclic factor=2
#pragma HLS ARRAY_PARTITION variable=borderLutGY dim=2 cyclic factor=2
#pragma HLS ARRAY_PARTITION variable=borderLutGZ dim=2 cyclic factor=2

#pragma HLS ARRAY_PARTITION variable=borderLutBX dim=2 cyclic factor=2
#pragma HLS ARRAY_PARTITION variable=borderLutBY dim=2 cyclic factor=2
#pragma HLS ARRAY_PARTITION variable=borderLutBZ dim=2 cyclic factor=2
    // clang-format on

    _FIXED_LUT_TYPE stmp_r, stmp_g, stmp_b;
    int loc_z = 0, loc_y = 0, loc_x = 0, temp = 0;
    int r_int = 0, g_int = 0, b_int = 0;
    int count = 0;
    static constexpr int step = XF_DTPIXELDEPTH(INTYPE, NPPC);
    const ap_ufixed<step + 1, step> __max = (float)(__MAXVAL(step));

z_loop:
    for (unsigned char k = 0; k < lutdim; k++) {
#pragma HLS LOOP_TRIPCOUNT min = LUTDIM max = LUTDIM
    y_loop:
        for (unsigned char l = 0; l < lutdim; l++) {
#pragma HLS LOOP_TRIPCOUNT min = LUTDIM max = LUTDIM
        x_loop:
            for (unsigned char m = 0; m < lutdim; m++) {
#pragma HLS LOOP_TRIPCOUNT min = LUTDIM max = LUTDIM

                ap_uint<96> inLutVal = lut.read(k * lutdim * lutdim + l * lutdim + m);

                r_int = inLutVal.range(31, 0);
                g_int = inLutVal.range(63, 32);
                b_int = inLutVal.range(95, 64);

                stmp_r = *((float*)(&r_int));
                stmp_g = *((float*)(&g_int));
                stmp_b = *((float*)(&b_int));

                loc_x = m;
                loc_y = l;
                loc_z = k;

                /* All border pixels in all three dimensions to be
                 * stored in a separate arrays instead of main 3d array to
                 * get cyclic partition applicable to main 3d array*/
                if (loc_x == lutdim - 1) { // border pixels in x-dim
                    borderLutRX[loc_z][loc_y] = stmp_r;
                    borderLutGX[loc_z][loc_y] = stmp_g;
                    borderLutBX[loc_z][loc_y] = stmp_b;
                } else if (loc_y == lutdim - 1) { // y-dim
                    borderLutRY[loc_z][loc_x] = stmp_r;
                    borderLutGY[loc_z][loc_x] = stmp_g;
                    borderLutBY[loc_z][loc_x] = stmp_b;
                } else if (loc_z == lutdim - 1) { // z-dim
                    borderLutRZ[loc_y][loc_x] = stmp_r;
                    borderLutGZ[loc_y][loc_x] = stmp_g;
                    borderLutBZ[loc_y][loc_x] = stmp_b;
                } else { // Non-border pixels

                    lutGrid_r[loc_z][loc_y][loc_x] = stmp_r;
                    lutGrid_g[loc_z][loc_y][loc_x] = stmp_g;
                    lutGrid_b[loc_z][loc_y][loc_x] = stmp_b;
                }
            }
        }
    }

    _FIXED_PIXEL_TYPE outG = 0;
    _FIXED_PIXEL_TYPE outB = 0;
    _FIXED_PIXEL_TYPE outR = 0;

ROW_LOOP:
    for (short i = 0; i < in_img.rows; ++i) {
#pragma HLS LOOP_TRIPCOUNT min = ROWS max = ROWS
#pragma HLS PIPELINE II = 1
    COL_LOOP:
        for (short j = 0; j < in_img.cols; ++j) {
#pragma HLS LOOP_TRIPCOUNT min = COLS max = COLS
            XF_TNAME(INTYPE, NPPC) inPix = in_img.read(i * in_img.cols + j);
            ap_uint<step> inPixR = inPix.range(step - 1, 0);
            ap_uint<step> inPixG = inPix.range(step * 2 - 1, step);
            ap_uint<step> inPixB = inPix.range(step * 3 - 1, step * 2);

            _FIXED_LUT_TYPE scale_r = (int)inPixR / (float)(__MAXVAL(step));
            _FIXED_LUT_TYPE scale_g = (int)inPixG / (float)(__MAXVAL(step));
            _FIXED_LUT_TYPE scale_b = (int)inPixB / (float)(__MAXVAL(step));

            _FIXED_PIXEL_TYPE index_r = scale_r * (lutdim - 1);
            _FIXED_PIXEL_TYPE index_g = scale_g * (lutdim - 1);
            _FIXED_PIXEL_TYPE index_b = scale_b * (lutdim - 1);

            pixelIndex.R = (int)(index_r.to_float());
            pixelIndex.G = (int)(index_g.to_float());
            pixelIndex.B = (int)(index_b.to_float());

            _FIXED_LUT_TYPE dist_r = index_r - pixelIndex.R;
            _FIXED_LUT_TYPE dist_g = index_g - pixelIndex.G;
            _FIXED_LUT_TYPE dist_b = index_b - pixelIndex.B;

            /* No need to interpolate for border pixels*/
            if (pixelIndex.R == lutdim - 1) {
                outR = borderLutRX[pixelIndex.B][pixelIndex.G];
                outG = borderLutGX[pixelIndex.B][pixelIndex.G];
                outB = borderLutBX[pixelIndex.B][pixelIndex.G];
            } else if (pixelIndex.G == lutdim - 1) {
                outR = borderLutRY[pixelIndex.B][pixelIndex.R];
                outG = borderLutGY[pixelIndex.B][pixelIndex.R];
                outB = borderLutBY[pixelIndex.B][pixelIndex.R];
            } else if (pixelIndex.B == lutdim - 1) {
                outR = borderLutRZ[pixelIndex.G][pixelIndex.R];
                outG = borderLutGZ[pixelIndex.G][pixelIndex.R];
                outB = borderLutBZ[pixelIndex.G][pixelIndex.R];
            } else { // Interpolate for non-border pixels

                /* Special condition for last but one border pixels
                 * as few values of the cube reside in main 3d array
                 * and the rest in the borderLut array(s).
                 */
                if (pixelIndex.R == lutdim - 2) { // x-dimension

                    cubeBufferR.P001 = borderLutRX[pixelIndex.B][pixelIndex.G];
                    cubeBufferG.P001 = borderLutGX[pixelIndex.B][pixelIndex.G];
                    cubeBufferB.P001 = borderLutBX[pixelIndex.B][pixelIndex.G];

                    cubeBufferR.P011 = borderLutRX[pixelIndex.B][pixelIndex.G + 1];
                    cubeBufferG.P011 = borderLutGX[pixelIndex.B][pixelIndex.G + 1];
                    cubeBufferB.P011 = borderLutBX[pixelIndex.B][pixelIndex.G + 1];

                    cubeBufferR.P101 = borderLutRX[pixelIndex.B + 1][pixelIndex.G];
                    cubeBufferG.P101 = borderLutGX[pixelIndex.B + 1][pixelIndex.G];
                    cubeBufferB.P101 = borderLutBX[pixelIndex.B + 1][pixelIndex.G];

                    cubeBufferR.P111 = borderLutRX[pixelIndex.B + 1][pixelIndex.G + 1];
                    cubeBufferG.P111 = borderLutGX[pixelIndex.B + 1][pixelIndex.G + 1];
                    cubeBufferB.P111 = borderLutBX[pixelIndex.B + 1][pixelIndex.G + 1];

                    if (pixelIndex.R == lutdim - 2 && pixelIndex.G == lutdim - 2 &&
                        pixelIndex.B < lutdim - 2) { // x-y border

                        cubeBufferR.P000 = lutGrid_r[pixelIndex.B][pixelIndex.G][pixelIndex.R];
                        cubeBufferG.P000 = lutGrid_g[pixelIndex.B][pixelIndex.G][pixelIndex.R];
                        cubeBufferB.P000 = lutGrid_b[pixelIndex.B][pixelIndex.G][pixelIndex.R];

                        cubeBufferR.P010 = borderLutRY[pixelIndex.B][pixelIndex.R];
                        cubeBufferG.P010 = borderLutGY[pixelIndex.B][pixelIndex.R];
                        cubeBufferB.P010 = borderLutBY[pixelIndex.B][pixelIndex.R];

                        cubeBufferR.P100 = lutGrid_r[pixelIndex.B + 1][pixelIndex.G][pixelIndex.R];
                        cubeBufferG.P100 = lutGrid_g[pixelIndex.B + 1][pixelIndex.G][pixelIndex.R];
                        cubeBufferB.P100 = lutGrid_b[pixelIndex.B + 1][pixelIndex.G][pixelIndex.R];

                        cubeBufferR.P110 = borderLutRY[pixelIndex.B + 1][pixelIndex.R];
                        cubeBufferG.P110 = borderLutGY[pixelIndex.B + 1][pixelIndex.R];
                        cubeBufferB.P110 = borderLutBY[pixelIndex.B + 1][pixelIndex.R];

                    } else if (pixelIndex.R == lutdim - 2 && pixelIndex.B == lutdim - 2 &&
                               pixelIndex.G < lutdim - 2) { // x-z border

                        cubeBufferR.P000 = lutGrid_r[pixelIndex.B][pixelIndex.G][pixelIndex.R];
                        cubeBufferG.P000 = lutGrid_g[pixelIndex.B][pixelIndex.G][pixelIndex.R];
                        cubeBufferB.P000 = lutGrid_b[pixelIndex.B][pixelIndex.G][pixelIndex.R];

                        cubeBufferR.P010 = lutGrid_r[pixelIndex.B][pixelIndex.G + 1][pixelIndex.R];
                        cubeBufferG.P010 = lutGrid_g[pixelIndex.B][pixelIndex.G + 1][pixelIndex.R];
                        cubeBufferB.P010 = lutGrid_b[pixelIndex.B][pixelIndex.G + 1][pixelIndex.R];

                        cubeBufferR.P100 = borderLutRZ[pixelIndex.G][pixelIndex.R];
                        cubeBufferG.P100 = borderLutGZ[pixelIndex.G][pixelIndex.R];
                        cubeBufferB.P100 = borderLutBZ[pixelIndex.G][pixelIndex.R];

                        cubeBufferR.P110 = borderLutRZ[pixelIndex.G + 1][pixelIndex.R];
                        cubeBufferG.P110 = borderLutGZ[pixelIndex.G + 1][pixelIndex.R];
                        cubeBufferB.P110 = borderLutBZ[pixelIndex.G + 1][pixelIndex.R];
                    } else if (pixelIndex.R == lutdim - 2 && pixelIndex.G == lutdim - 2 &&
                               pixelIndex.B == lutdim - 2) { // x-y-z border

                        cubeBufferR.P000 = lutGrid_r[pixelIndex.B][pixelIndex.G][pixelIndex.R];
                        cubeBufferG.P000 = lutGrid_g[pixelIndex.B][pixelIndex.G][pixelIndex.R];
                        cubeBufferB.P000 = lutGrid_b[pixelIndex.B][pixelIndex.G][pixelIndex.R];

                        cubeBufferR.P010 = borderLutRY[pixelIndex.B][pixelIndex.R];
                        cubeBufferG.P010 = borderLutGY[pixelIndex.B][pixelIndex.R];
                        cubeBufferB.P010 = borderLutBY[pixelIndex.B][pixelIndex.R];

                        cubeBufferR.P100 = borderLutRZ[pixelIndex.G][pixelIndex.R];
                        cubeBufferG.P100 = borderLutGZ[pixelIndex.G][pixelIndex.R];
                        cubeBufferB.P100 = borderLutBZ[pixelIndex.G][pixelIndex.R];

                        cubeBufferR.P110 = borderLutRY[pixelIndex.B + 1][pixelIndex.R];
                        cubeBufferG.P110 = borderLutGY[pixelIndex.B + 1][pixelIndex.R];
                        cubeBufferB.P110 = borderLutBY[pixelIndex.B + 1][pixelIndex.R];
                    } else { // only x border
                        cubeBufferR.P000 = lutGrid_r[pixelIndex.B][pixelIndex.G][pixelIndex.R];
                        cubeBufferG.P000 = lutGrid_g[pixelIndex.B][pixelIndex.G][pixelIndex.R];
                        cubeBufferB.P000 = lutGrid_b[pixelIndex.B][pixelIndex.G][pixelIndex.R];

                        cubeBufferR.P010 = lutGrid_r[pixelIndex.B][pixelIndex.G + 1][pixelIndex.R];
                        cubeBufferG.P010 = lutGrid_g[pixelIndex.B][pixelIndex.G + 1][pixelIndex.R];
                        cubeBufferB.P010 = lutGrid_b[pixelIndex.B][pixelIndex.G + 1][pixelIndex.R];

                        cubeBufferR.P100 = lutGrid_r[pixelIndex.B + 1][pixelIndex.G][pixelIndex.R];
                        cubeBufferG.P100 = lutGrid_g[pixelIndex.B + 1][pixelIndex.G][pixelIndex.R];
                        cubeBufferB.P100 = lutGrid_b[pixelIndex.B + 1][pixelIndex.G][pixelIndex.R];

                        cubeBufferR.P110 = lutGrid_r[pixelIndex.B + 1][pixelIndex.G + 1][pixelIndex.R];
                        cubeBufferG.P110 = lutGrid_g[pixelIndex.B + 1][pixelIndex.G + 1][pixelIndex.R];
                        cubeBufferB.P110 = lutGrid_b[pixelIndex.B + 1][pixelIndex.G + 1][pixelIndex.R];
                    }
                } else if (pixelIndex.G == lutdim - 2) { // y-dimension

                    if (pixelIndex.B == lutdim - 2 && pixelIndex.G == lutdim - 2 &&
                        pixelIndex.R < lutdim - 2) { // y-z border

                        cubeBufferR.P000 = lutGrid_r[pixelIndex.B][pixelIndex.G][pixelIndex.R];
                        cubeBufferG.P000 = lutGrid_g[pixelIndex.B][pixelIndex.G][pixelIndex.R];
                        cubeBufferB.P000 = lutGrid_b[pixelIndex.B][pixelIndex.G][pixelIndex.R];

                        cubeBufferR.P010 = borderLutRY[pixelIndex.B][pixelIndex.R];
                        cubeBufferG.P010 = borderLutGY[pixelIndex.B][pixelIndex.R];
                        cubeBufferB.P010 = borderLutBY[pixelIndex.B][pixelIndex.R];

                        cubeBufferR.P100 = borderLutRZ[pixelIndex.G][pixelIndex.R];
                        cubeBufferG.P100 = borderLutGZ[pixelIndex.G][pixelIndex.R];
                        cubeBufferB.P100 = borderLutBZ[pixelIndex.G][pixelIndex.R];

                        cubeBufferR.P110 = borderLutRY[pixelIndex.B + 1][pixelIndex.R];
                        cubeBufferG.P110 = borderLutGY[pixelIndex.B + 1][pixelIndex.R];
                        cubeBufferB.P110 = borderLutBY[pixelIndex.B + 1][pixelIndex.R];

                        cubeBufferR.P001 = lutGrid_r[pixelIndex.B][pixelIndex.G][pixelIndex.R + 1];
                        cubeBufferG.P001 = lutGrid_g[pixelIndex.B][pixelIndex.G][pixelIndex.R + 1];
                        cubeBufferB.P001 = lutGrid_b[pixelIndex.B][pixelIndex.G][pixelIndex.R + 1];

                        cubeBufferR.P011 = borderLutRY[pixelIndex.B][pixelIndex.R + 1];
                        cubeBufferG.P011 = borderLutGY[pixelIndex.B][pixelIndex.R + 1];
                        cubeBufferB.P011 = borderLutBY[pixelIndex.B][pixelIndex.R + 1];

                        cubeBufferR.P101 = borderLutRZ[pixelIndex.G][pixelIndex.R + 1];
                        cubeBufferG.P101 = borderLutGZ[pixelIndex.G][pixelIndex.R + 1];
                        cubeBufferB.P101 = borderLutBZ[pixelIndex.G][pixelIndex.R + 1];

                        cubeBufferR.P111 = borderLutRY[pixelIndex.B + 1][pixelIndex.R + 1];
                        cubeBufferG.P111 = borderLutGY[pixelIndex.B + 1][pixelIndex.R + 1];
                        cubeBufferB.P111 = borderLutBY[pixelIndex.B + 1][pixelIndex.R + 1];

                    } else { // only y-border
                        cubeBufferR.P000 = lutGrid_r[pixelIndex.B][pixelIndex.G][pixelIndex.R];
                        cubeBufferG.P000 = lutGrid_g[pixelIndex.B][pixelIndex.G][pixelIndex.R];
                        cubeBufferB.P000 = lutGrid_b[pixelIndex.B][pixelIndex.G][pixelIndex.R];

                        cubeBufferR.P001 = lutGrid_r[pixelIndex.B][pixelIndex.G][pixelIndex.R + 1];
                        cubeBufferG.P001 = lutGrid_g[pixelIndex.B][pixelIndex.G][pixelIndex.R + 1];
                        cubeBufferB.P001 = lutGrid_b[pixelIndex.B][pixelIndex.G][pixelIndex.R + 1];

                        cubeBufferR.P100 = lutGrid_r[pixelIndex.B + 1][pixelIndex.G][pixelIndex.R];
                        cubeBufferG.P100 = lutGrid_g[pixelIndex.B + 1][pixelIndex.G][pixelIndex.R];
                        cubeBufferB.P100 = lutGrid_b[pixelIndex.B + 1][pixelIndex.G][pixelIndex.R];

                        cubeBufferR.P101 = lutGrid_r[pixelIndex.B + 1][pixelIndex.G][pixelIndex.R + 1];
                        cubeBufferG.P101 = lutGrid_g[pixelIndex.B + 1][pixelIndex.G][pixelIndex.R + 1];
                        cubeBufferB.P101 = lutGrid_b[pixelIndex.B + 1][pixelIndex.G][pixelIndex.R + 1];

                        cubeBufferR.P010 = borderLutRY[pixelIndex.B][pixelIndex.R];
                        cubeBufferG.P010 = borderLutGY[pixelIndex.B][pixelIndex.R];
                        cubeBufferB.P010 = borderLutBY[pixelIndex.B][pixelIndex.R];

                        cubeBufferR.P011 = borderLutRY[pixelIndex.B][pixelIndex.R + 1];
                        cubeBufferG.P011 = borderLutGY[pixelIndex.B][pixelIndex.R + 1];
                        cubeBufferB.P011 = borderLutBY[pixelIndex.B][pixelIndex.R + 1];

                        cubeBufferR.P110 = borderLutRY[pixelIndex.B + 1][pixelIndex.R];
                        cubeBufferG.P110 = borderLutGY[pixelIndex.B + 1][pixelIndex.R];
                        cubeBufferB.P110 = borderLutBY[pixelIndex.B + 1][pixelIndex.R];

                        cubeBufferR.P111 = borderLutRY[pixelIndex.B + 1][pixelIndex.R + 1];
                        cubeBufferG.P111 = borderLutGY[pixelIndex.B + 1][pixelIndex.R + 1];
                        cubeBufferB.P111 = borderLutBY[pixelIndex.B + 1][pixelIndex.R + 1];
                    }
                } else if (pixelIndex.B == lutdim - 2) { // z-dimension

                    cubeBufferR.P000 = lutGrid_r[pixelIndex.B][pixelIndex.G][pixelIndex.R];
                    cubeBufferG.P000 = lutGrid_g[pixelIndex.B][pixelIndex.G][pixelIndex.R];
                    cubeBufferB.P000 = lutGrid_b[pixelIndex.B][pixelIndex.G][pixelIndex.R];

                    cubeBufferR.P001 = lutGrid_r[pixelIndex.B][pixelIndex.G][pixelIndex.R + 1];
                    cubeBufferG.P001 = lutGrid_g[pixelIndex.B][pixelIndex.G][pixelIndex.R + 1];
                    cubeBufferB.P001 = lutGrid_b[pixelIndex.B][pixelIndex.G][pixelIndex.R + 1];

                    cubeBufferR.P010 = lutGrid_r[pixelIndex.B][pixelIndex.G + 1][pixelIndex.R];
                    cubeBufferG.P010 = lutGrid_g[pixelIndex.B][pixelIndex.G + 1][pixelIndex.R];
                    cubeBufferB.P010 = lutGrid_b[pixelIndex.B][pixelIndex.G + 1][pixelIndex.R];

                    cubeBufferR.P011 = lutGrid_r[pixelIndex.B][pixelIndex.G + 1][pixelIndex.R + 1];
                    cubeBufferG.P011 = lutGrid_g[pixelIndex.B][pixelIndex.G + 1][pixelIndex.R + 1];
                    cubeBufferB.P011 = lutGrid_b[pixelIndex.B][pixelIndex.G + 1][pixelIndex.R + 1];

                    cubeBufferR.P100 = borderLutRZ[pixelIndex.G][pixelIndex.R];
                    cubeBufferG.P100 = borderLutGZ[pixelIndex.G][pixelIndex.R];
                    cubeBufferB.P100 = borderLutBZ[pixelIndex.G][pixelIndex.R];

                    cubeBufferR.P101 = borderLutRZ[pixelIndex.G][pixelIndex.R + 1];
                    cubeBufferG.P101 = borderLutGZ[pixelIndex.G][pixelIndex.R + 1];
                    cubeBufferB.P101 = borderLutBZ[pixelIndex.G][pixelIndex.R + 1];

                    cubeBufferR.P110 = borderLutRZ[pixelIndex.G + 1][pixelIndex.R];
                    cubeBufferG.P110 = borderLutGZ[pixelIndex.G + 1][pixelIndex.R];
                    cubeBufferB.P110 = borderLutBZ[pixelIndex.G + 1][pixelIndex.R];

                    cubeBufferR.P111 = borderLutRZ[pixelIndex.G + 1][pixelIndex.R + 1];
                    cubeBufferG.P111 = borderLutGZ[pixelIndex.G + 1][pixelIndex.R + 1];
                    cubeBufferB.P111 = borderLutBZ[pixelIndex.G + 1][pixelIndex.R + 1];
                } else { // For all other pixels use only main 3d array
                    cubeBufferR.P000 = lutGrid_r[pixelIndex.B][pixelIndex.G][pixelIndex.R];
                    cubeBufferG.P000 = lutGrid_g[pixelIndex.B][pixelIndex.G][pixelIndex.R];
                    cubeBufferB.P000 = lutGrid_b[pixelIndex.B][pixelIndex.G][pixelIndex.R];

                    cubeBufferR.P001 = lutGrid_r[pixelIndex.B][pixelIndex.G][pixelIndex.R + 1];
                    cubeBufferG.P001 = lutGrid_g[pixelIndex.B][pixelIndex.G][pixelIndex.R + 1];
                    cubeBufferB.P001 = lutGrid_b[pixelIndex.B][pixelIndex.G][pixelIndex.R + 1];

                    cubeBufferR.P010 = lutGrid_r[pixelIndex.B][pixelIndex.G + 1][pixelIndex.R];
                    cubeBufferG.P010 = lutGrid_g[pixelIndex.B][pixelIndex.G + 1][pixelIndex.R];
                    cubeBufferB.P010 = lutGrid_b[pixelIndex.B][pixelIndex.G + 1][pixelIndex.R];

                    cubeBufferR.P011 = lutGrid_r[pixelIndex.B][pixelIndex.G + 1][pixelIndex.R + 1];
                    cubeBufferG.P011 = lutGrid_g[pixelIndex.B][pixelIndex.G + 1][pixelIndex.R + 1];
                    cubeBufferB.P011 = lutGrid_b[pixelIndex.B][pixelIndex.G + 1][pixelIndex.R + 1];

                    cubeBufferR.P100 = lutGrid_r[pixelIndex.B + 1][pixelIndex.G][pixelIndex.R];
                    cubeBufferG.P100 = lutGrid_g[pixelIndex.B + 1][pixelIndex.G][pixelIndex.R];
                    cubeBufferB.P100 = lutGrid_b[pixelIndex.B + 1][pixelIndex.G][pixelIndex.R];

                    cubeBufferR.P101 = lutGrid_r[pixelIndex.B + 1][pixelIndex.G][pixelIndex.R + 1];
                    cubeBufferG.P101 = lutGrid_g[pixelIndex.B + 1][pixelIndex.G][pixelIndex.R + 1];
                    cubeBufferB.P101 = lutGrid_b[pixelIndex.B + 1][pixelIndex.G][pixelIndex.R + 1];

                    cubeBufferR.P110 = lutGrid_r[pixelIndex.B + 1][pixelIndex.G + 1][pixelIndex.R];
                    cubeBufferG.P110 = lutGrid_g[pixelIndex.B + 1][pixelIndex.G + 1][pixelIndex.R];
                    cubeBufferB.P110 = lutGrid_b[pixelIndex.B + 1][pixelIndex.G + 1][pixelIndex.R];

                    cubeBufferR.P111 = lutGrid_r[pixelIndex.B + 1][pixelIndex.G + 1][pixelIndex.R + 1];
                    cubeBufferG.P111 = lutGrid_g[pixelIndex.B + 1][pixelIndex.G + 1][pixelIndex.R + 1];
                    cubeBufferB.P111 = lutGrid_b[pixelIndex.B + 1][pixelIndex.G + 1][pixelIndex.R + 1];
                }

                outR = interp3(cubeBufferR, dist_r, dist_g, dist_b);

                outG = interp3(cubeBufferG, dist_r, dist_g, dist_b);

                outB = interp3(cubeBufferB, dist_r, dist_g, dist_b);
            }

            XF_TNAME(OUTTYPE, NPPC) outPix = 0;

            _FIXED_OUT_PIXEL_TYPE _outR = outR * __max;
            _FIXED_OUT_PIXEL_TYPE _outG = outG * __max;
            _FIXED_OUT_PIXEL_TYPE _outB = outB * __max;

            outPix.range(step - 1, 0) = _outR;
            outPix.range(step * 2 - 1, step) = _outG;
            outPix.range(step * 3 - 1, step * 2) = _outB;

            out_img.write(i * in_img.cols + j, outPix);
        }
    }
}
}
}
