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

#ifndef __XF_DEMOSAICING_HPP__
#define __XF_DEMOSAICING_HPP__

#include "ap_int.h"
#include "common/xf_common.hpp"
#include "hls_stream.h"

/**Utility macros and functions**/

#define MAXVAL(pixeldepth) ((1 << pixeldepth) - 1)
#define XF_UCHAR_MAX 255
#define XF_UTENBIT_MAX 1023
#define XF_UTWELVEBIT_MAX 4095
#define XF_USHORT_MAX 65535

template <typename T>
T xf_satcast(int in_val){};

template <>
inline ap_uint<8> xf_satcast<ap_uint<8> >(int v) {
    return (v > MAXVAL(8) ? XF_UCHAR_MAX : v);
};
template <>
inline ap_uint<10> xf_satcast<ap_uint<10> >(int v) {
    return (v > MAXVAL(10) ? XF_UTENBIT_MAX : v);
};
template <>
inline ap_uint<12> xf_satcast<ap_uint<12> >(int v) {
    return (v > MAXVAL(12) ? XF_UTWELVEBIT_MAX : v);
};
template <>
inline ap_uint<16> xf_satcast<ap_uint<16> >(int v) {
    return (v > MAXVAL(16) ? XF_USHORT_MAX : v);
};

/*  */
namespace xf {
namespace cv {

template <typename T, int buf_size>
int g_kernel(T imgblock[5][buf_size], int loop) {
// clang-format off
    #pragma HLS inline off
    // clang-format on
    int res = -(imgblock[0][2 + loop] + imgblock[2][0 + loop] + imgblock[2][4 + loop] + imgblock[4][2 + loop]) +
              (imgblock[1][2 + loop] + imgblock[2][1 + loop] + imgblock[2][3 + loop] + imgblock[3][2 + loop]) * 2 +
              (imgblock[2][2 + loop]) * 4;
    res /= 8;
    if (res < 0) return 0;
    return res;
}
// R at a B location and B at a R location
template <typename T, int buf_size>
int rb_kernel(T imgblock[5][buf_size], int loop) {
// clang-format off
    #pragma HLS inline off
    // clang-format on
    int t1 = (imgblock[0][2 + loop] + imgblock[2][0 + loop] + imgblock[2][4 + loop] + imgblock[4][2 + loop]);
    t1 = (t1 * 3) / 2;
    int t2 = (imgblock[1][1 + loop] + imgblock[1][3 + loop] + imgblock[3][1 + loop] + imgblock[3][3 + loop]);
    t2 = t2 * 2;
    int t3 = (imgblock[2][2 + loop]) * 6;
    int res = (-t1) + (t2) + (t3);
    res /= 8;
    if (res < 0) return 0;
    return res;
}
//**** R at a G location in R row ****** B at a Green location in a B row
//*******
template <typename T, int buf_size>
int rgr_bgb_kernel(T imgblock[5][buf_size], int loop) {
// clang-format off
    #pragma HLS inline off
    // clang-format on
    int t1 = imgblock[0][2 + loop] + imgblock[4][2 + loop];
    int t2 = imgblock[1][1 + loop] + imgblock[1][3 + loop] + imgblock[2][0 + loop] + imgblock[2][4 + loop] +
             imgblock[3][1 + loop] + imgblock[3][3 + loop];
    int t3 = imgblock[2][1 + loop] + imgblock[2][3 + loop];
    t3 *= 4;
    int t4 = (imgblock[2][2 + loop]) * 5;
    int res = ((t1) >> 1) - (t2) + (t3) + (t4);
    res /= 8;
    if (res < 0) return 0;
    return res;
}
// R at a G location in a B row and B at a G location in R row
template <typename T, int buf_size>
int rgb_bgr_kernel(T imgblock[5][buf_size], int loop) {
// clang-format off
    #pragma HLS inline off
    // clang-format on
    int t1 = (imgblock[2][0 + loop] + imgblock[2][4 + loop]);
    t1 /= 2;
    int t2 = imgblock[0][2 + loop] + imgblock[1][1 + loop] + imgblock[1][3 + loop] + imgblock[3][1 + loop] +
             imgblock[3][3 + loop] + imgblock[4][2 + loop];
    int t3 = imgblock[1][2 + loop] + imgblock[3][2 + loop];
    t3 *= 4;
    int t4 = (imgblock[2][2 + loop]) * 5;
    int res = (t1) - (t2) + (t3) + (t4);
    res /= 8;
    if (res < 0) return 0;
    return res;
}

template <int BFORMAT, int SRC_T, int NPC, int DEPTH, int buf_size>
void Core_Process(XF_DTUNAME(SRC_T, NPC) imgblock[5][buf_size], int& b, int& g, int& r, int row, int col, int loop) {
    if (BFORMAT == XF_BAYER_RG) {
        if ((row & 0x00000001) == 0) {     // if R row
            if ((col & 0x00000001) == 0) { // We already have R value at this location
                b = rb_kernel<XF_DTUNAME(SRC_T, NPC), buf_size>(imgblock, loop);
                g = g_kernel<XF_DTUNAME(SRC_T, NPC), buf_size>(imgblock, loop);
                r = imgblock[2][2 + loop];
            } else { // We already have G value at this location
                b = rgb_bgr_kernel<XF_DTUNAME(SRC_T, NPC), buf_size>(imgblock, loop);
                g = imgblock[2][2 + loop];
                r = rgr_bgb_kernel<XF_DTUNAME(SRC_T, NPC), buf_size>(imgblock, loop);
            }
        } else {                           // B row
            if ((col & 0x00000001) == 0) { // We have already G value at this location
                b = rgr_bgb_kernel<XF_DTUNAME(SRC_T, NPC), buf_size>(imgblock, loop);
                g = imgblock[2][2 + loop];
                r = rgb_bgr_kernel<XF_DTUNAME(SRC_T, NPC), buf_size>(imgblock, loop);
            } else { // We already have B value at this location
                b = imgblock[2][2 + loop];
                g = g_kernel<XF_DTUNAME(SRC_T, NPC), buf_size>(imgblock, loop);
                r = rb_kernel<XF_DTUNAME(SRC_T, NPC), buf_size>(imgblock, loop);
            }
        }
    } else if (BFORMAT == XF_BAYER_BG) {
        if ((row & 0x00000001) == 0) {     // if B row
            if ((col & 0x00000001) == 0) { // Even row, even column - We already have
                                           // B value at this location
                r = rb_kernel<XF_DTUNAME(SRC_T, NPC), buf_size>(imgblock, loop);
                g = g_kernel<XF_DTUNAME(SRC_T, NPC), buf_size>(imgblock, loop);
                b = imgblock[2][2 + loop];
            } else { // Even row, odd column - We already have G value at this
                     // location
                b = rgr_bgb_kernel<XF_DTUNAME(SRC_T, NPC), buf_size>(imgblock, loop);
                g = imgblock[2][2 + loop];
                r = rgb_bgr_kernel<XF_DTUNAME(SRC_T, NPC), buf_size>(imgblock, loop);
            }
        } else {                           // if R row
            if ((col & 0x00000001) == 0) { // Odd row, even column - We have G value at this location
                b = rgb_bgr_kernel<XF_DTUNAME(SRC_T, NPC), buf_size>(imgblock, loop);
                g = imgblock[2][2 + loop];
                r = rgr_bgb_kernel<XF_DTUNAME(SRC_T, NPC), buf_size>(imgblock, loop);
            } else { // Odd row, odd column - We already have R value at this location
                r = imgblock[2][2 + loop];
                g = g_kernel<XF_DTUNAME(SRC_T, NPC), buf_size>(imgblock, loop);
                b = rb_kernel<XF_DTUNAME(SRC_T, NPC), buf_size>(imgblock, loop);
            }
        }
    } else if (BFORMAT == XF_BAYER_GB) {
        if ((row & 0x00000001) == 0) {     // if B row
            if ((col & 0x00000001) == 0) { // Even row, even column - We already have
                                           // G value at this location
                b = rgr_bgb_kernel<XF_DTUNAME(SRC_T, NPC), buf_size>(imgblock, loop);
                g = imgblock[2][2 + loop];
                r = rgb_bgr_kernel<XF_DTUNAME(SRC_T, NPC), buf_size>(imgblock, loop);
            } else { // Even row, odd column - We already have B value at this
                     // location
                b = imgblock[2][2 + loop];
                g = g_kernel<XF_DTUNAME(SRC_T, NPC), buf_size>(imgblock, loop);
                r = rb_kernel<XF_DTUNAME(SRC_T, NPC), buf_size>(imgblock, loop);
            }
        } else {                           // if R row
            if ((col & 0x00000001) == 0) { // Odd row, even column - We have R value at this location
                r = imgblock[2][2 + loop];
                g = g_kernel<XF_DTUNAME(SRC_T, NPC), buf_size>(imgblock, loop);
                b = rb_kernel<XF_DTUNAME(SRC_T, NPC), buf_size>(imgblock, loop);
            } else { // Odd row, odd column - We already have G value at this location
                b = rgb_bgr_kernel<XF_DTUNAME(SRC_T, NPC), buf_size>(imgblock, loop);
                g = imgblock[2][2 + loop];
                r = rgr_bgb_kernel<XF_DTUNAME(SRC_T, NPC), buf_size>(imgblock, loop);
            }
        }
    } else if (BFORMAT == XF_BAYER_GR) {
        if ((row & 0x00000001) == 0) {     // if R row
            if ((col & 0x00000001) == 0) { // Even row, even column - We already have
                                           // G value at this location
                b = rgb_bgr_kernel<XF_DTUNAME(SRC_T, NPC), buf_size>(imgblock, loop);
                g = imgblock[2][2 + loop];
                r = rgr_bgb_kernel<XF_DTUNAME(SRC_T, NPC), buf_size>(imgblock, loop);
            } else { // Even row, odd column - We already have R value at this
                     // location
                r = imgblock[2][2 + loop];
                g = g_kernel<XF_DTUNAME(SRC_T, NPC), buf_size>(imgblock, loop);
                b = rb_kernel<XF_DTUNAME(SRC_T, NPC), buf_size>(imgblock, loop);
            }
        } else {                           // if B row
            if ((col & 0x00000001) == 0) { // Odd row, even column - We have B value at this location
                b = imgblock[2][2 + loop];
                g = g_kernel<XF_DTUNAME(SRC_T, NPC), buf_size>(imgblock, loop);
                r = rb_kernel<XF_DTUNAME(SRC_T, NPC), buf_size>(imgblock, loop);
            } else { // Odd row, odd column - We already have G value at this location
                b = rgr_bgb_kernel<XF_DTUNAME(SRC_T, NPC), buf_size>(imgblock, loop);
                g = imgblock[2][2 + loop];
                r = rgb_bgr_kernel<XF_DTUNAME(SRC_T, NPC), buf_size>(imgblock, loop);
            }
        }
    }
}

template <int BFORMAT, int SRC_T, int DST_T, int ROWS, int COLS, int NPC, bool USE_URAM>
void demosaicing(xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& src_mat, xf::cv::Mat<DST_T, ROWS, COLS, NPC>& dst_mat) {
#ifndef __SYNTHESIS__
    assert(((BFORMAT == XF_BAYER_BG) || (BFORMAT == XF_BAYER_GB) || (BFORMAT == XF_BAYER_GR) ||
            (BFORMAT == XF_BAYER_RG)) &&
           ("Unsupported Bayer pattern. Use anyone among: "
            "XF_BAYER_BG;XF_BAYER_GB;XF_BAYER_GR;XF_BAYER_RG"));
    assert(((src_mat.rows <= ROWS) && (src_mat.cols <= COLS)) && "ROWS and COLS should be greater than input image");
    assert(((NPC == 1) || (NPC == 2) || (NPC == 4)) && "Only 1, 2 and 4 pixel-parallelism are supported");
    assert(((SRC_T == XF_8UC1) || (SRC_T == XF_10UC1) || (SRC_T == XF_12UC1) || (SRC_T == XF_16UC1)) &&
           "Only 8, 10, 12 and 16 bit, single channel images are supported");
    assert(((DST_T == XF_8UC3) || (DST_T == XF_10UC3) || (DST_T == XF_12UC3) || (DST_T == XF_16UC3) ||
            (DST_T == XF_8UC4) || (DST_T == XF_10UC4) || (DST_T == XF_12UC4) || (DST_T == XF_16UC4)) &&
           "Only 8, 10, 12 and 16 bit, 3 and 4 channel images are supported");
#endif
    const int __BHEIGHT = 5;
    const int __BHEIGHTMINUSONE = __BHEIGHT - 1;
    const int __BWIDTH = NPC + __BHEIGHTMINUSONE + (((NPC - 1) >> 1) << 1);

// clang-format off
    #pragma HLS INLINE OFF
    // clang-format on
    XF_TNAME(SRC_T, NPC) linebuffer[__BHEIGHTMINUSONE][COLS >> XF_BITSHIFT(NPC)];
    if (USE_URAM) {
// clang-format off
        #pragma HLS RESOURCE variable=linebuffer core=RAM_T2P_URAM
        #pragma HLS array_reshape variable=linebuffer dim=1 factor=4 cyclic
        // clang-format on
    } else {
// clang-format off
        #pragma HLS RESOURCE variable=linebuffer core=RAM_T2P_BRAM
        #pragma HLS array_partition variable=linebuffer complete dim=1
        // clang-format on
    }
    XF_CTUNAME(SRC_T, NPC) imgblock[__BHEIGHT][__BWIDTH];
    const int pre_read_count = (2 / NPC) + ((NPC * NPC) >> 2);  // 2-2-4
    const int post_read_count = pre_read_count + 2;             // 4-4-6
    const int end_read_count = ((NPC << 1) >> (NPC * NPC)) + 1; // 2-1-1

// clang-format off
    #pragma HLS array_partition variable=imgblock complete dim=0
    // clang-format on

    int lineStore = 3, read_index = 0, write_index = 0;
LineBuffer:
    for (int i = 0; i < 2; i++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=2 max=2
        // clang-format on
        for (int j = 0; j<src_mat.cols>> XF_BITSHIFT(NPC); j++) {
// clang-format off
            #pragma HLS LOOP_TRIPCOUNT min=COLS/NPC max=COLS/NPC
            #pragma HLS pipeline ii=1
            // clang-format on
            XF_TNAME(SRC_T, NPC) tmp = src_mat.read(read_index++);
            linebuffer[i][j] = 0;
            linebuffer[i + 2][j] = tmp;
        }
    }
    ap_uint<3> line0 = 3, line1 = 0, line2 = 1, line3 = 2;
    int step = XF_DTPIXELDEPTH(SRC_T, NPC);
    int out_step = XF_DTPIXELDEPTH(DST_T, NPC);
    XF_TNAME(SRC_T, NPC) tmp;

Row_Loop:
    for (int i = 0; i < src_mat.rows; i++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=ROWS max=ROWS
        // clang-format on
        int bram_read_count = 0;
        lineStore++;
        if (lineStore > 3) {
            lineStore = 0;
        }
        if (line0 == 0) {
            line0 = 1;
            line1 = 2;
            line2 = 3;
            line3 = 0;
        } else if (line0 == 1) {
            line0 = 2;
            line1 = 3;
            line2 = 0;
            line3 = 1;
        } else if (line0 == 2) {
            line0 = 3;
            line1 = 0;
            line2 = 1;
            line3 = 2;
        } else {
            line0 = 0;
            line1 = 1;
            line2 = 2;
            line3 = 3;
        }

    /*Image left corner case */
    Zero:
        for (int p = 0; p < 4; ++p) {
// clang-format off
            #pragma HLS PIPELINE ii=1
            // clang-format on
            for (int k = 0; k < NPC + 2; k++) {
                imgblock[p][k] = 0;
            }
        }

    /*Filling the data in the first four rows of 5x5/5x6/5x10 window from
     * linebuffer */
    Datafill:
        for (int n = 0, w = 0, v = 0; n < pre_read_count; ++n, ++v) {
// clang-format off
            #pragma HLS UNROLL
            // clang-format on
            imgblock[0][2 + NPC + n] = linebuffer[line0][w].range((step + step * v) - 1, step * v);
            imgblock[1][2 + NPC + n] = linebuffer[line1][w].range((step + step * v) - 1, step * v);
            imgblock[2][2 + NPC + n] = linebuffer[line2][w].range((step + step * v) - 1, step * v);
            imgblock[3][2 + NPC + n] = linebuffer[line3][w].range((step + step * v) - 1, step * v);
            (NPC == 1) ? (bram_read_count++, w++, v = -1) : bram_read_count; // Read twice (for 3rd and 4th locations of
                                                                             // imgblock) for NPPC1
        }
        (NPC == 2 || NPC == 4) ? (bram_read_count++) : bram_read_count;

    Col_Loop:
        for (int j = 0; j < ((src_mat.cols) >> XF_BITSHIFT(NPC)); j++) {
// clang-format off
            #pragma HLS PIPELINE ii=1
            #pragma HLS dependence variable=linebuffer inter false
            #pragma HLS LOOP_TRIPCOUNT min=COLS/NPC max=COLS/NPC
            #pragma HLS LOOP_FLATTEN OFF
            // clang-format on

            if (i < src_mat.rows - 2) {
                tmp = src_mat.read(read_index++); // Reading 5th row element
            } else {
                tmp = 0;
            }

            for (int z = 0; z < NPC; ++z) {
                imgblock[4][2 + NPC + z] = tmp.range((step + step * z) - 1, step * z);
            }

            // Shift the elements in imgblock by NPC
            for (int k = 0; k < 5; k++) {
                for (int m = 0; m < NPC; ++m) {
                    for (int l = 0; l < (__BWIDTH - 1); l++) {
                        imgblock[k][l] = imgblock[k][l + 1];
                    }
                }
            }
            XF_TNAME(SRC_T, NPC)
            packed_read1, packed_read2, packed_read3, packed_read4, packed_store;

            if (j < (src_mat.cols >> XF_BITSHIFT(NPC)) - end_read_count) { // for each element being processed that is
                                                                           // not at borders

                packed_read1 = linebuffer[line0][bram_read_count];
                packed_read2 = linebuffer[line1][bram_read_count];
                packed_read3 = linebuffer[line2][bram_read_count];
                packed_read4 = linebuffer[line3][bram_read_count];

                for (int q = 0; q < NPC; ++q) {
                    imgblock[0][post_read_count + q] = packed_read1.range((step + step * q) - 1, step * q);
                    imgblock[1][post_read_count + q] = packed_read2.range((step + step * q) - 1, step * q);
                    imgblock[2][post_read_count + q] = packed_read3.range((step + step * q) - 1, step * q);
                    imgblock[3][post_read_count + q] = packed_read4.range((step + step * q) - 1, step * q);
                    imgblock[4][NPC + 2 + q] = tmp.range((step + step * q) - 1, step * q);
                    packed_store.range((step + step * q) - 1, step * q) = imgblock[4][2 + q];
                }
                linebuffer[lineStore][j] = packed_store;

            } else { // For processing elements at the end of the line.
                for (int r = 0; r < NPC; ++r) {
                    if (NPC == 1) {
                        imgblock[4][post_read_count + r - 1] = tmp.range((step + step * r) - 1, step * r);
                    }
                    linebuffer[lineStore][j].range((step + step * r) - 1, step * r) = imgblock[4][2 + r];

                    imgblock[0][post_read_count + r] = 0;
                    imgblock[1][post_read_count + r] = 0;
                    imgblock[2][post_read_count + r] = 0;
                    imgblock[3][post_read_count + r] = 0;
                    imgblock[4][post_read_count + r] = 0;
                }
            }

            bram_read_count++;

            // Calculate the resultant intensities at each pixel
            int r, g, b;
            XF_TNAME(DST_T, NPC) res_pixel[NPC];

            for (int loop = 0; loop < NPC; loop++) {
                Core_Process<BFORMAT, SRC_T, NPC, XF_DEPTH(SRC_T, NPC), __BWIDTH>(imgblock, b, g, r, i, j * NPC + loop,
                                                                                  loop);

                b = xf_satcast<XF_CTUNAME(DST_T, NPC)>(b);
                g = xf_satcast<XF_CTUNAME(DST_T, NPC)>(g);
                r = xf_satcast<XF_CTUNAME(DST_T, NPC)>(r);

                if (XF_CHANNELS(DST_T, NPC) == 4) {
                    res_pixel[loop].range(4 * out_step - 1, 3 * out_step) = MAXVAL(out_step);
                }
                res_pixel[loop].range(3 * out_step - 1, 2 * out_step) = r; // xf_satcast<XF_CTUNAME(SRC_T, NPC)>(r);
                res_pixel[loop].range(2 * out_step - 1, out_step) = g;     // xf_satcast<XF_CTUNAME(SRC_T, NPC)>(g);
                res_pixel[loop].range(out_step - 1, 0) = b;                // xf_satcast<XF_CTUNAME(SRC_T, NPC)>(b);
            }
            XF_TNAME(DST_T, NPC) packed_res_pixel;
            int pstep = XF_PIXELWIDTH(DST_T, NPC);
            for (int ploop = 0; ploop < NPC; ploop++) {
                packed_res_pixel.range(pstep + pstep * ploop - 1, pstep * ploop) = res_pixel[ploop];
            }

            dst_mat.write(write_index++, packed_res_pixel);

        } // end COL loop
    }     // end ROW loop
}

} // namespace cv
}; // namespace xf
#endif
