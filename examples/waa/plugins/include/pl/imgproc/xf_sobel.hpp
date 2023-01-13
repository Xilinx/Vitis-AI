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

#ifndef _XF_SOBEL_HPP_
#define _XF_SOBEL_HPP_

typedef unsigned short uint16_t;

typedef unsigned int uint32_t;

#include "common/xf_common.hpp"
#include "common/xf_utility.hpp"
#include "hls_stream.h"

namespace xf {
namespace cv {

/*****************************************************************
 * 		                 SobelFilter3x3
 *****************************************************************
 * X-Gradient Computation
 *
 * -------------
 * |-1	0 	1|
 * |-2	0	2|
 * |-1	0	1|
 * -------------
 *****************************************************************/
template <int DEPTH_SRC, int DEPTH_DST>
XF_PTNAME(DEPTH_DST)
xFGradientX3x3(XF_PTNAME(DEPTH_SRC) t0,
               XF_PTNAME(DEPTH_SRC) t1,
               XF_PTNAME(DEPTH_SRC) t2,
               XF_PTNAME(DEPTH_SRC) m0,
               XF_PTNAME(DEPTH_SRC) m1,
               XF_PTNAME(DEPTH_SRC) m2,
               XF_PTNAME(DEPTH_SRC) b0,
               XF_PTNAME(DEPTH_SRC) b1,
               XF_PTNAME(DEPTH_SRC) b2) {
// clang-format off
#pragma HLS INLINE off
    // clang-format on

    XF_PTNAME(DEPTH_DST) g_x = 0;
    // ap_uint<8> g_x = 0;
    short int M00 = ((short int)m0 << 1);
    short int M01 = ((short int)m2 << 1);
    short int A00 = (t2 + b2);
    short int S00 = (t0 + b0);

    short int out_pix;
    out_pix = M01 - M00;
    out_pix = out_pix + A00;
    out_pix = out_pix - S00;

    g_x = (XF_PTNAME(DEPTH_DST))out_pix;

    if ((DEPTH_DST == XF_8UP) || (DEPTH_DST == XF_24UP)) {
        if (out_pix < 0) {
            g_x = 0;
        } else if (out_pix > 255) {
            g_x = 255;
        }
    }

    return g_x;
}

/**********************************************************************
 * Y-Gradient Computation
 * -------------
 * | 1	 2 	 1|
 * | 0	 0	 0|
 * |-1	-2	-1|
 * -------------
 **********************************************************************/
template <int DEPTH_SRC, int DEPTH_DST>
XF_PTNAME(DEPTH_DST)
xFGradientY3x3(XF_PTNAME(DEPTH_SRC) t0,
               XF_PTNAME(DEPTH_SRC) t1,
               XF_PTNAME(DEPTH_SRC) t2,
               XF_PTNAME(DEPTH_SRC) m0,
               XF_PTNAME(DEPTH_SRC) m1,
               XF_PTNAME(DEPTH_SRC) m2,
               XF_PTNAME(DEPTH_SRC) b0,
               XF_PTNAME(DEPTH_SRC) b1,
               XF_PTNAME(DEPTH_SRC) b2) {
// clang-format off
#pragma HLS INLINE off
    // clang-format on

    XF_PTNAME(DEPTH_DST) g_y = 0;

    short int M00 = ((short int)t1 << 1);
    short int M01 = ((short int)b1 << 1);
    short int A00 = (b0 + b2);
    short int S00 = (t0 + t2);

    short int out_pix;
    out_pix = M01 - M00;
    out_pix = out_pix + A00;
    out_pix = out_pix - S00;

    g_y = (XF_PTNAME(DEPTH_DST))out_pix;

    if ((DEPTH_DST == XF_8UP) || (DEPTH_DST == XF_24UP)) {
        if (out_pix < 0) {
            g_y = 0;
        } else if (out_pix > 255) {
            g_y = 255;
        }
    }
    return g_y;
}

/**
 * xFSobel3x3 : Applies the mask and Computes the gradient values
 *
 */
template <int PLANES, int NPC, int DEPTH_SRC, int DEPTH_DST>
void xFSobel3x3(XF_PTNAME(DEPTH_DST) * GradientvaluesX,
                XF_PTNAME(DEPTH_DST) * GradientvaluesY,
                XF_PTNAME(DEPTH_SRC) * src_buf1,
                XF_PTNAME(DEPTH_SRC) * src_buf2,
                XF_PTNAME(DEPTH_SRC) * src_buf3) {
// clang-format off
#pragma HLS INLINE off
    // clang-format on

    int STEP, STEP_OUT;
    if ((DEPTH_DST == XF_48SP) || (DEPTH_DST == XF_16SP)) {
        STEP_OUT = 16;
        STEP = 8;
    } else {
        STEP_OUT = 8;
        STEP = 8;
    }

Compute_Grad_Loop:
    for (ap_uint<5> j = 0; j < XF_NPIXPERCYCLE(NPC); j++) {
        int p = 0;
// clang-format off
#pragma HLS UNROLL
        // clang-format on
        for (ap_uint<5> c = 0, k = 0; c < PLANES; c++, k += STEP) {
            GradientvaluesX[j].range(p + (STEP_OUT - 1), p) = xFGradientX3x3<DEPTH_SRC, DEPTH_DST>(
                src_buf1[j].range(k + STEP - 1, k), src_buf1[j + 1].range(k + STEP - 1, k),
                src_buf1[j + 2].range(k + STEP - 1, k), src_buf2[j].range(k + STEP - 1, k),
                src_buf2[j + 1].range(k + STEP - 1, k), src_buf2[j + 2].range(k + STEP - 1, k),
                src_buf3[j].range(k + STEP - 1, k), src_buf3[j + 1].range(k + STEP - 1, k),
                src_buf3[j + 2].range(k + STEP - 1, k));

            GradientvaluesY[j].range(p + (STEP_OUT - 1), p) = xFGradientY3x3<DEPTH_SRC, DEPTH_DST>(
                src_buf1[j].range(k + STEP - 1, k), src_buf1[j + 1].range(k + STEP - 1, k),
                src_buf1[j + 2].range(k + STEP - 1, k), src_buf2[j].range(k + STEP - 1, k),
                src_buf2[j + 1].range(k + STEP - 1, k), src_buf2[j + 2].range(k + STEP - 1, k),
                src_buf3[j].range(k + STEP - 1, k), src_buf3[j + 1].range(k + STEP - 1, k),
                src_buf3[j + 2].range(k + STEP - 1, k));
            p += STEP_OUT;
        }
    }
}

/**************************************************************************************
 * ProcessSobel3x3 : Computes gradients for the column input data
 **************************************************************************************/
template <int SRC_T,
          int DST_T,
          int ROWS,
          int COLS,
          int PLANES,
          int DEPTH_SRC,
          int DEPTH_DST,
          int NPC,
          int WORDWIDTH_SRC,
          int WORDWIDTH_DST,
          int TC>
void ProcessSobel3x3(xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _src_mat,
                     xf::cv::Mat<DST_T, ROWS, COLS, NPC>& _gradx_mat,
                     xf::cv::Mat<DST_T, ROWS, COLS, NPC>& _grady_mat,
                     XF_SNAME(WORDWIDTH_SRC) buf[3][(COLS >> XF_BITSHIFT(NPC))],
                     XF_PTNAME(DEPTH_SRC) src_buf1[XF_NPIXPERCYCLE(NPC) + 2],
                     XF_PTNAME(DEPTH_SRC) src_buf2[XF_NPIXPERCYCLE(NPC) + 2],
                     XF_PTNAME(DEPTH_SRC) src_buf3[XF_NPIXPERCYCLE(NPC) + 2],
                     XF_PTNAME(DEPTH_DST) GradientValuesX[XF_NPIXPERCYCLE(NPC)],
                     XF_PTNAME(DEPTH_DST) GradientValuesY[XF_NPIXPERCYCLE(NPC)],
                     XF_SNAME(WORDWIDTH_DST) & P0,
                     XF_SNAME(WORDWIDTH_DST) & P1,
                     uint16_t img_width,
                     uint16_t img_height,
                     ap_uint<13> row_ind,
                     uint16_t& shift_x,
                     uint16_t& shift_y,
                     ap_uint<2> tp,
                     ap_uint<2> mid,
                     ap_uint<2> bottom,
                     ap_uint<13> row,
                     int& read_index,
                     int& write_index) {
// clang-format off
#pragma HLS INLINE
    // clang-format on

    XF_SNAME(WORDWIDTH_SRC) buf0, buf1, buf2;
    uint16_t npc = XF_NPIXPERCYCLE(NPC);
    ap_uint<5> buf_size = XF_NPIXPERCYCLE(NPC) + 2;

Col_Loop:
    for (ap_uint<13> col = 0; col < img_width; col++) {
// clang-format off
#pragma HLS LOOP_TRIPCOUNT min=TC max=TC
#pragma HLS pipeline
        // clang-format on
        if (row < img_height)
            buf[row_ind][col] = _src_mat.read(read_index++); // Read data
        else
            buf[bottom][col] = 0;
        buf0 = buf[tp][col];
        buf1 = buf[mid][col];
        buf2 = buf[bottom][col];

        if (NPC == XF_NPPC8) {
            xfExtractPixels<NPC, WORDWIDTH_SRC, DEPTH_SRC>(&src_buf1[2], buf0, 0);
            xfExtractPixels<NPC, WORDWIDTH_SRC, DEPTH_SRC>(&src_buf2[2], buf1, 0);
            xfExtractPixels<NPC, WORDWIDTH_SRC, DEPTH_SRC>(&src_buf3[2], buf2, 0);
        } else {
            src_buf1[2] = buf0;
            src_buf2[2] = buf1;
            src_buf3[2] = buf2;
        }

        xFSobel3x3<PLANES, NPC, DEPTH_SRC, DEPTH_DST>(GradientValuesX, GradientValuesY, src_buf1, src_buf2, src_buf3);

        if (col == 0) {
            shift_x = 0;
            shift_y = 0;
            P0 = 0;
            P1 = 0;

            xfPackPixels<NPC, WORDWIDTH_DST, DEPTH_DST>(&GradientValuesX[0], P0, 1, (npc - 1), shift_x);
            xfPackPixels<NPC, WORDWIDTH_DST, DEPTH_DST>(&GradientValuesY[0], P1, 1, (npc - 1), shift_y);

        } else {
            xfPackPixels<NPC, WORDWIDTH_DST, DEPTH_DST>(&GradientValuesX[0], P0, 0, 1, shift_x);
            xfPackPixels<NPC, WORDWIDTH_DST, DEPTH_DST>(&GradientValuesY[0], P1, 0, 1, shift_y);

            _gradx_mat.write(write_index, P0);
            _grady_mat.write(write_index++, P1);

            shift_x = 0;
            shift_y = 0;
            P0 = 0;
            P1 = 0;

            xfPackPixels<NPC, WORDWIDTH_DST, DEPTH_DST>(&GradientValuesX[0], P0, 1, (npc - 1), shift_x);
            xfPackPixels<NPC, WORDWIDTH_DST, DEPTH_DST>(&GradientValuesY[0], P1, 1, (npc - 1), shift_y);
        }

        src_buf1[0] = src_buf1[buf_size - 2];
        src_buf1[1] = src_buf1[buf_size - 1];
        src_buf2[0] = src_buf2[buf_size - 2];
        src_buf2[1] = src_buf2[buf_size - 1];
        src_buf3[0] = src_buf3[buf_size - 2];
        src_buf3[1] = src_buf3[buf_size - 1];
    } // Col_Loop
}

/**
 * xFSobelFilter3x3 : Computes Sobel gradient of the input image
 *                    for filtersize 3x3
 * _src_mat		: Input image
 * _gradx_mat	: GradientX output
 * _grady_mat	: GradientY output
 */
template <int SRC_T,
          int DST_T,
          int ROWS,
          int COLS,
          int PLANES,
          int DEPTH_SRC,
          int DEPTH_DST,
          int NPC,
          int WORDWIDTH_SRC,
          int WORDWIDTH_DST,
          int TC,
          bool USE_URAM>
void xFSobelFilter3x3(xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _src_mat,
                      xf::cv::Mat<DST_T, ROWS, COLS, NPC>& _dst_matx,
                      xf::cv::Mat<DST_T, ROWS, COLS, NPC>& _dst_maty,
                      uint16_t img_height,
                      uint16_t img_width) {
    ap_uint<13> row_ind;
    ap_uint<2> tp, mid, bottom;
    ap_uint<8> buf_size = XF_NPIXPERCYCLE(NPC) + 2;
    uint16_t shift_x = 0, shift_y = 0;
    ap_uint<13> row, col;
    int read_index = 0, write_index = 0;

    XF_PTNAME(DEPTH_DST)
    GradientValuesX[XF_NPIXPERCYCLE(NPC)]; // X-Gradient result buffer
    XF_PTNAME(DEPTH_DST)
    GradientValuesY[XF_NPIXPERCYCLE(NPC)]; // Y-Gradient result buffer
                                           // clang-format off
#pragma HLS ARRAY_PARTITION variable=GradientValuesX complete dim=1
#pragma HLS ARRAY_PARTITION variable=GradientValuesY complete dim=1
    // clang-format on

    XF_PTNAME(DEPTH_SRC)
    src_buf1[XF_NPIXPERCYCLE(NPC) + 2],
        src_buf2[XF_NPIXPERCYCLE(NPC) + 2], // Temporary buffers to hold input data for processing
        src_buf3[XF_NPIXPERCYCLE(NPC) + 2];
// clang-format off
#pragma HLS ARRAY_PARTITION variable=src_buf1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=src_buf2 complete dim=1
#pragma HLS ARRAY_PARTITION variable=src_buf3 complete dim=1
    // clang-format on

    XF_SNAME(WORDWIDTH_DST) P0, P1; // Output data is packed
    // Line buffer to hold image data
    XF_SNAME(WORDWIDTH_SRC) buf[3][(COLS >> XF_BITSHIFT(NPC))]; // Line buffer
    if (USE_URAM) {
// clang-format off
#pragma HLS array reshape variable=buf dim=1 factor=3 cyclic
#pragma HLS RESOURCE variable=buf core=RAM_S2P_URAM
        // clang-format on
    } else {
// clang-format off
#pragma HLS RESOURCE variable=buf core=RAM_S2P_BRAM
#pragma HLS ARRAY_PARTITION variable=buf complete dim=1
        // clang-format on
    }
    row_ind = 1;

Clear_Row_Loop:
    for (col = 0; col < img_width; col++) // Top row border care
    {
// clang-format off
#pragma HLS LOOP_TRIPCOUNT min=TC max=TC
#pragma HLS pipeline
        // clang-format on
        buf[0][col] = 0;
        buf[row_ind][col] = _src_mat.read(read_index++); // Read data
    }
    row_ind++;

Row_Loop: // Process complete image
    for (row = 1; row < img_height + 1; row++) {
// clang-format off
#pragma HLS LOOP_TRIPCOUNT min=ROWS max=ROWS
        // clang-format on
        if (row_ind == 2) // Indexes to hold maintain the row index
        {
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

        /***********		Process complete row
         * **********/
        P0 = P1 = 0;
        ProcessSobel3x3<SRC_T, DST_T, ROWS, COLS, PLANES, DEPTH_SRC, DEPTH_DST, NPC, WORDWIDTH_SRC, WORDWIDTH_DST, TC>(
            _src_mat, _dst_matx, _dst_maty, buf, src_buf1, src_buf2, src_buf3, GradientValuesX, GradientValuesY, P0, P1,
            img_width, img_height, row_ind, shift_x, shift_y, tp, mid, bottom, row, read_index, write_index);

        /*			Last column border care	for RO & PO Case
         */
        if ((NPC == XF_NPPC8)) {
            //	Compute gradient at last column
            int STEP, STEP_OUT, p = 0;
            if ((DEPTH_DST == XF_48SP) || (DEPTH_DST == XF_16SP)) {
                STEP_OUT = 16;
                STEP = 8;
            } else {
                STEP_OUT = 8;
                STEP = 8;
            }
            for (ap_uint<5> c = 0, k = 0; c < PLANES; c++, k += STEP) {
                GradientValuesX[0].range(p + (STEP_OUT - 1), p) = xFGradientX3x3<DEPTH_SRC, DEPTH_DST>(
                    src_buf1[buf_size - 2].range(k + STEP - 1, k), src_buf1[buf_size - 1].range(k + STEP - 1, k), 0,
                    src_buf2[buf_size - 2].range(k + STEP - 1, k), src_buf2[buf_size - 1].range(k + STEP - 1, k), 0,
                    src_buf3[buf_size - 2].range(k + STEP - 1, k), src_buf3[buf_size - 1].range(k + STEP - 1, k), 0);

                GradientValuesY[0].range(p + (STEP_OUT - 1), p) = xFGradientY3x3<DEPTH_SRC, DEPTH_DST>(
                    src_buf1[buf_size - 2].range(k + STEP - 1, k), src_buf1[buf_size - 1].range(k + STEP - 1, k), 0,
                    src_buf2[buf_size - 2].range(k + STEP - 1, k), src_buf2[buf_size - 1].range(k + STEP - 1, k), 0,
                    src_buf3[buf_size - 2].range(k + STEP - 1, k), src_buf3[buf_size - 1].range(k + STEP - 1, k), 0);
                p += STEP_OUT;
            }
        } else /*			Last column border care	for NO Case
         */
        {
            int STEP, STEP_OUT, q = 0;
            if ((DEPTH_DST == XF_48SP) || (DEPTH_DST == XF_16SP)) {
                STEP_OUT = 16;
                STEP = 8;
            } else {
                STEP_OUT = 8;
                STEP = 8;
            }
            for (ap_uint<7> i = 0, k = 0; i < PLANES; i++, k += STEP) {
                GradientValuesX[0].range(q + (STEP_OUT - 1), q) = xFGradientX3x3<DEPTH_SRC, DEPTH_DST>(
                    src_buf1[buf_size - 3].range(k + STEP - 1, k), src_buf1[buf_size - 2].range(k + STEP - 1, k), 0,
                    src_buf2[buf_size - 3].range(k + STEP - 1, k), src_buf2[buf_size - 2].range(k + STEP - 1, k), 0,
                    src_buf3[buf_size - 3].range(k + STEP - 1, k), src_buf3[buf_size - 2].range(k + STEP - 1, k), 0);

                GradientValuesY[0].range(q + (STEP_OUT - 1), q) = xFGradientY3x3<DEPTH_SRC, DEPTH_DST>(
                    src_buf1[buf_size - 3].range(k + STEP - 1, k), src_buf1[buf_size - 2].range(k + STEP - 1, k), 0,
                    src_buf2[buf_size - 3].range(k + STEP - 1, k), src_buf2[buf_size - 2].range(k + STEP - 1, k), 0,
                    src_buf3[buf_size - 3].range(k + STEP - 1, k), src_buf3[buf_size - 2].range(k + STEP - 1, k), 0);
                q += STEP_OUT;
            }
        }

        xfPackPixels<NPC, WORDWIDTH_DST, DEPTH_DST>(&GradientValuesX[0], P0, 0, 1, shift_x);
        xfPackPixels<NPC, WORDWIDTH_DST, DEPTH_DST>(&GradientValuesY[0], P1, 0, 1, shift_y);

        _dst_matx.write(write_index, P0);
        _dst_maty.write(write_index++, P1);

        shift_x = 0;
        shift_y = 0;
        P0 = 0;
        P1 = 0;

        row_ind++;
        if (row_ind == 3) {
            row_ind = 0;
        }
    } // Row_Loop
}
// xFSobelFilter3x3

/*****************************************************************
 * 		                 SobelFilter5x5
 *****************************************************************/
/**
 *  Sobel Filter X-Gradient used is 5x5
 *
 *       --- ---- ---- ---- ---
 *      | -1 |  -2 | 0 |  2 | 1 |
 *       --- ---- ---- ---- ---
 *      | -4 |  -8 | 0 |  8 | 4 |
 *       --- ---- ---- ---- ---
 *      | -6 | -12 | 0 | 12 | 6 |
 *       --- ---- ---- ---- ---
 *      | -4 |  -8 | 0 |  8 | 4 |
 *       --- ---- ---- ---- ---
 *      | -1 |  -2 | 0 |  2 | 1 |
 *       --- ---- ---- ---- ---
 ****************************************************************/

template <int PLANES, int DEPTH_SRC, int DEPTH_DST>
XF_PTNAME(DEPTH_DST)
xFGradientX5x5(XF_PTNAME(DEPTH_SRC) * src_buf1,
               XF_PTNAME(DEPTH_SRC) * src_buf2,
               XF_PTNAME(DEPTH_SRC) * src_buf3,
               XF_PTNAME(DEPTH_SRC) * src_buf4,
               XF_PTNAME(DEPTH_SRC) * src_buf5) {
// clang-format off
#pragma HLS INLINE off
    // clang-format on
    XF_PTNAME(DEPTH_DST) g_x = 0, out_val = 0;

    int STEP, STEP_OUT, p = 0;
    if ((DEPTH_DST == XF_48SP) || (DEPTH_DST == XF_16SP)) {
        STEP_OUT = 16;
        STEP = 8;
    } else {
        STEP_OUT = 8;
        STEP = 8;
    }

    for (int i = 0, k = 0; i < PLANES; i++, k += STEP) {
        short int M00 =
            (short int)(((short int)src_buf1[1].range(k + STEP - 1, k) + (short int)src_buf5[1].range(k + STEP - 1, k))
                        << 1);
        short int M01 =
            (short int)((short int)src_buf1[4].range(k + STEP - 1, k) + (short int)src_buf5[4].range(k + STEP - 1, k)) -
            ((short int)src_buf1[0].range(k + STEP - 1, k) + (short int)src_buf5[0].range(k + STEP - 1, k));
        short int A00 =
            (short int)(((short int)src_buf1[3].range(k + STEP - 1, k) + (short int)src_buf5[3].range(k + STEP - 1, k))
                        << 1);
        short int M02 =
            (short int)(((short int)src_buf2[0].range(k + STEP - 1, k) + (short int)src_buf4[0].range(k + STEP - 1, k))
                        << 2);
        short int M03 =
            (short int)((short int)src_buf2[1].range(k + STEP - 1, k) + (short int)src_buf4[1].range(k + STEP - 1, k))
            << 3;
        short int A01 =
            (short int)((short int)src_buf2[3].range(k + STEP - 1, k) + (short int)src_buf4[3].range(k + STEP - 1, k))
            << 3;
        short int A02 =
            (short int)((short int)src_buf2[4].range(k + STEP - 1, k) + (short int)src_buf4[4].range(k + STEP - 1, k))
            << 2;
        short int M04 = (short int)src_buf3[0].range(k + STEP - 1, k) * 6;
        short int M05 = (short int)src_buf3[1].range(k + STEP - 1, k) * 12;
        short int A03 = (short int)src_buf3[3].range(k + STEP - 1, k) * 12;
        short int A04 = (short int)src_buf3[4].range(k + STEP - 1, k) * 6;
        short int S00 = M00 + M02;
        short int S01 = M03 + M04 + M05;
        short int A0 = A00 + A01;
        short int A1 = A02 + A03;
        short int A2 = A04 + M01;
        short int FA = A0 + A1 + A2;
        short int FS = S00 + S01;
        short int out_x = FA - FS;

        g_x = (XF_PTNAME(DEPTH_DST))out_x;

        if ((DEPTH_DST == XF_8UP) || (DEPTH_DST == XF_24UP)) {
            if (out_x < 0)
                g_x = 0;
            else if (out_x > 255)
                g_x = 255;
        }
        out_val.range(p + (STEP_OUT - 1), p) = g_x;
        p += STEP_OUT;
    }
    return out_val;
}
/****************************************************************
 * Sobel Filter Y-Gradient used is 5x5
 *
 *       --- ---- ---- ---- ---
 *      | -1 |  -4 |  -6 |  -4 | -1 |
 *       --- ---- ---- ---- ---
 *      | -2 |  -8 | -12 |  -8 | -2 |
 *       --- ---- ---- ---- ---
 *      |  0 |   0 |   0 |   0 |  0 |
 *       --- ---- ---- ---- --- ---
 *      |  2 |   8 |  12 |   8 |  2 |
 *       --- ---- ---- ---- --- ---
 *      |  1 |   4 |   6 |   4 |  1 |
 *       --- ---- ---- ---- --- ---
 ******************************************************************/

template <int PLANES, int DEPTH_SRC, int DEPTH_DST>
XF_PTNAME(DEPTH_DST)
xFGradientY5x5(XF_PTNAME(DEPTH_SRC) * src_buf1,
               XF_PTNAME(DEPTH_SRC) * src_buf2,
               XF_PTNAME(DEPTH_SRC) * src_buf3,
               XF_PTNAME(DEPTH_SRC) * src_buf4,
               XF_PTNAME(DEPTH_SRC) * src_buf5) {
// clang-format off
#pragma HLS INLINE off
    // clang-format on
    XF_PTNAME(DEPTH_DST) g_y = 0, out_val = 0;
    int STEP, STEP_OUT, p = 0;
    if ((DEPTH_DST == XF_48SP) || (DEPTH_DST == XF_16SP)) {
        STEP_OUT = 16;
        STEP = 8;
    } else {
        STEP_OUT = 8;
        STEP = 8;
    }

    for (int i = 0, k = 0; i < PLANES; i++, k += STEP) {
        short int M00 =
            ((short int)src_buf5[0].range(k + STEP - 1, k) + (short int)src_buf5[4].range(k + STEP - 1, k)) -
            ((short int)src_buf1[0].range(k + STEP - 1, k) + (short int)src_buf1[4].range(k + STEP - 1, k));
        short int M01 =
            (short int)(((short int)src_buf1[1].range(k + STEP - 1, k) + (short int)src_buf1[3].range(k + STEP - 1, k))
                        << 2);
        short int A00 =
            (short int)(((short int)src_buf5[1].range(k + STEP - 1, k) + (short int)src_buf5[3].range(k + STEP - 1, k))
                        << 2);
        short int M02 =
            (short int)(((short int)src_buf2[0].range(k + STEP - 1, k) + (short int)src_buf2[4].range(k + STEP - 1, k))
                        << 1);
        short int A01 =
            (short int)(((short int)src_buf4[0].range(k + STEP - 1, k) + (short int)src_buf4[4].range(k + STEP - 1, k))
                        << 1);
        short int M03 =
            (short int)(((short int)src_buf2[1].range(k + STEP - 1, k) + (short int)src_buf2[3].range(k + STEP - 1, k))
                        << 3);
        short int A02 =
            (short int)(((short int)src_buf4[1].range(k + STEP - 1, k) + (short int)src_buf4[3].range(k + STEP - 1, k))
                        << 3);
        short int M04 = (short int)(src_buf1[2].range(k + STEP - 1, k) * 6);
        short int M05 = (short int)(src_buf2[2].range(k + STEP - 1, k) * 12);
        short int A03 = (short int)(src_buf4[2].range(k + STEP - 1, k) * 12);
        short int A04 = (short int)(src_buf5[2].range(k + STEP - 1, k) * 6);
        short int S00 = M01 + M02 + M03;
        short int S01 = M04 + M05;
        short int A0 = A00 + A01;
        short int A1 = A02 + A03;
        short int A2 = A04 + M00;
        short int FA = A0 + A1 + A2;
        short int FS = S00 + S01;
        short int out_y = FA - FS;

        g_y = (XF_PTNAME(DEPTH_DST))out_y;

        if ((DEPTH_DST == XF_8UP) || (DEPTH_DST == XF_24UP)) {
            if (out_y < 0)
                g_y = 0;
            else if (out_y > 255)
                g_y = 255;
        }
        out_val.range(p + (STEP_OUT - 1), p) = g_y;
        p += STEP_OUT;
    }
    return out_val;
}
/**
 * xFSobel5x5 : Applies the mask and Computes the gradient values
 *
 */
template <int NPC, int PLANES, int DEPTH_SRC, int DEPTH_DST>
void xFSobel5x5(XF_PTNAME(DEPTH_DST) * GradientvaluesX,
                XF_PTNAME(DEPTH_DST) * GradientvaluesY,
                XF_PTNAME(DEPTH_SRC) * src_buf1,
                XF_PTNAME(DEPTH_SRC) * src_buf2,
                XF_PTNAME(DEPTH_SRC) * src_buf3,
                XF_PTNAME(DEPTH_SRC) * src_buf4,
                XF_PTNAME(DEPTH_SRC) * src_buf5) {
// clang-format off
#pragma HLS INLINE off
// clang-format on

Compute_Grad_Loop:
    for (ap_uint<5> j = 0; j < XF_NPIXPERCYCLE(NPC); j++) {
// clang-format off
#pragma HLS LOOP_TRIPCOUNT min=8 max=8
#pragma HLS UNROLL
        // clang-format on
        GradientvaluesX[j] = xFGradientX5x5<PLANES, DEPTH_SRC, DEPTH_DST>(&src_buf1[j], &src_buf2[j], &src_buf3[j],
                                                                          &src_buf4[j], &src_buf5[j]);
        GradientvaluesY[j] = xFGradientY5x5<PLANES, DEPTH_SRC, DEPTH_DST>(&src_buf1[j], &src_buf2[j], &src_buf3[j],
                                                                          &src_buf4[j], &src_buf5[j]);
    }
}

/**************************************************************************************
 * ProcessSobel5x5 : Computes gradients for the column input data
 **************************************************************************************/
template <int SRC_T,
          int DST_T,
          int ROWS,
          int COLS,
          int PLANES,
          int DEPTH_SRC,
          int DEPTH_DST,
          int NPC,
          int WORDWIDTH_SRC,
          int WORDWIDTH_DST,
          int TC>
void ProcessSobel5x5(xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _src_mat,
                     xf::cv::Mat<DST_T, ROWS, COLS, NPC>& _dst_matx,
                     xf::cv::Mat<DST_T, ROWS, COLS, NPC>& _dst_maty,

                     XF_SNAME(WORDWIDTH_SRC) buf[5][(COLS >> XF_BITSHIFT(NPC))],
                     XF_PTNAME(DEPTH_SRC) src_buf1[XF_NPIXPERCYCLE(NPC) + 4],
                     XF_PTNAME(DEPTH_SRC) src_buf2[XF_NPIXPERCYCLE(NPC) + 4],
                     XF_PTNAME(DEPTH_SRC) src_buf3[XF_NPIXPERCYCLE(NPC) + 4],
                     XF_PTNAME(DEPTH_SRC) src_buf4[XF_NPIXPERCYCLE(NPC) + 4],
                     XF_PTNAME(DEPTH_SRC) src_buf5[XF_NPIXPERCYCLE(NPC) + 4],
                     XF_PTNAME(DEPTH_DST) GradientValuesX[XF_NPIXPERCYCLE(NPC)],
                     XF_PTNAME(DEPTH_DST) GradientValuesY[XF_NPIXPERCYCLE(NPC)],
                     XF_SNAME(WORDWIDTH_DST) & inter_valx,
                     XF_SNAME(WORDWIDTH_DST) & inter_valy,
                     uint16_t img_width,
                     uint16_t img_height,
                     ap_uint<13> row_ind,
                     uint16_t& shift_x,
                     uint16_t& shift_y,
                     ap_uint<4> tp1,
                     ap_uint<4> tp2,
                     ap_uint<4> mid,
                     ap_uint<4> bottom1,
                     ap_uint<4> bottom2,
                     ap_uint<13> row,
                     int& read_index,
                     int& write_index) {
// clang-format off
#pragma HLS INLINE
    // clang-format on
    XF_SNAME(WORDWIDTH_SRC) buf0, buf1, buf2, buf3, buf4;
    ap_uint<8> buf_size = XF_NPIXPERCYCLE(NPC) + 4;
    uint16_t npc = XF_NPIXPERCYCLE(NPC);
    ap_uint<8> max_loop = XF_WORDDEPTH(WORDWIDTH_DST);
    ap_uint<8> step = XF_PIXELDEPTH(DEPTH_DST);

Col_Loop:
    for (ap_uint<13> col = 0; col < img_width; col++) {
// clang-format off
#pragma HLS LOOP_TRIPCOUNT min=TC max=TC
#pragma HLS pipeline
        // clang-format on
        if (row < img_height)
            buf[row_ind][col] = _src_mat.read(read_index++);
        else
            buf[bottom2][col] = 0;

        buf0 = buf[tp1][col];
        buf1 = buf[tp2][col];
        buf2 = buf[mid][col];
        buf3 = buf[bottom1][col];
        buf4 = buf[bottom2][col];

        if (NPC == XF_NPPC8) {
            xfExtractPixels<NPC, WORDWIDTH_SRC, DEPTH_SRC>(&src_buf1[4], buf0, 0);
            xfExtractPixels<NPC, WORDWIDTH_SRC, DEPTH_SRC>(&src_buf2[4], buf1, 0);
            xfExtractPixels<NPC, WORDWIDTH_SRC, DEPTH_SRC>(&src_buf3[4], buf2, 0);
            xfExtractPixels<NPC, WORDWIDTH_SRC, DEPTH_SRC>(&src_buf4[4], buf3, 0);
            xfExtractPixels<NPC, WORDWIDTH_SRC, DEPTH_SRC>(&src_buf5[4], buf4, 0);
        } else {
            src_buf1[4] = buf0;
            src_buf2[4] = buf1;
            src_buf3[4] = buf2;
            src_buf4[4] = buf3;
            src_buf5[4] = buf4;
        }
        xFSobel5x5<NPC, PLANES, DEPTH_SRC, DEPTH_DST>(GradientValuesX, GradientValuesY, src_buf1, src_buf2, src_buf3,
                                                      src_buf4, src_buf5);

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
            shift_x = 0, shift_y = 0;
            inter_valx = 0;
            inter_valy = 0;

            xfPackPixels<NPC, WORDWIDTH_DST, DEPTH_DST>(&GradientValuesX[0], inter_valx, 2, (npc - 2), shift_x);
            xfPackPixels<NPC, WORDWIDTH_DST, DEPTH_DST>(&GradientValuesY[0], inter_valy, 2, (npc - 2), shift_y);

        } else {
            if ((NPC == XF_NPPC8)) {
                xfPackPixels<NPC, WORDWIDTH_DST, DEPTH_DST>(&GradientValuesX[0], inter_valx, 0, 2, shift_x);
                xfPackPixels<NPC, WORDWIDTH_DST, DEPTH_DST>(&GradientValuesY[0], inter_valy, 0, 2, shift_y);

                _dst_matx.write(write_index, inter_valx);
                _dst_maty.write(write_index++, inter_valy);

                shift_x = 0;
                shift_y = 0;
                inter_valx = 0;
                inter_valy = 0;
                xfPackPixels<NPC, WORDWIDTH_DST, DEPTH_DST>(&GradientValuesX[0], inter_valx, 2, (npc - 2), shift_x);
                xfPackPixels<NPC, WORDWIDTH_DST, DEPTH_DST>(&GradientValuesY[0], inter_valy, 2, (npc - 2), shift_y);
            } else {
                if (col >= 2) {
                    inter_valx((max_loop - 1), (max_loop - step)) = GradientValuesX[0];
                    inter_valy((max_loop - 1), (max_loop - step)) = GradientValuesY[0];
                    _dst_matx.write(write_index, inter_valx);
                    _dst_maty.write(write_index++, inter_valy);
                }
            }
        }
    } // Col_Loop
}

/**
 * xFSobelFilter5x5 : Computes Sobel gradient of the input image
 *                    for filtersize 5X5
 * _src_mat		: Input image
 * _gradx_mat	: GradientX output
 * _grady_mat	: GradientY output
 */
template <int SRC_T,
          int DST_T,
          int ROWS,
          int COLS,
          int PLANES,
          int DEPTH_SRC,
          int DEPTH_DST,
          int NPC,
          int WORDWIDTH_SRC,
          int WORDWIDTH_DST,
          int TC,
          bool USE_URAM>
void xFSobelFilter5x5(xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _src_mat,
                      xf::cv::Mat<DST_T, ROWS, COLS, NPC>& _dst_matx,
                      xf::cv::Mat<DST_T, ROWS, COLS, NPC>& _dst_maty,
                      uint16_t img_height,
                      uint16_t img_width) {
    ap_uint<13> row_ind;
    ap_uint<13> row, col;
    ap_uint<4> tp1, tp2, mid, bottom1, bottom2;
    ap_uint<5> i;

    ap_uint<8> buf_size = XF_NPIXPERCYCLE(NPC) + 4;
    ap_uint<9> step = XF_PIXELDEPTH(DEPTH_DST);
    ap_uint<9> max_loop = XF_WORDDEPTH(WORDWIDTH_DST);
    uint16_t shift_x = 0, shift_y = 0;
    int read_index = 0, write_index = 0;

    XF_PTNAME(DEPTH_DST) GradientValuesX[XF_NPIXPERCYCLE(NPC)];
    XF_PTNAME(DEPTH_DST) GradientValuesY[XF_NPIXPERCYCLE(NPC)];
// clang-format off
#pragma HLS ARRAY_PARTITION variable=GradientValuesX complete dim=1
#pragma HLS ARRAY_PARTITION variable=GradientValuesY complete dim=1
    // clang-format on

    XF_SNAME(WORDWIDTH_SRC) buf0, buf1, buf2, buf3, buf4;
    // Temporary buffers to hold image data from five rows
    XF_PTNAME(DEPTH_SRC)
    src_buf1[XF_NPIXPERCYCLE(NPC) + 4], src_buf2[XF_NPIXPERCYCLE(NPC) + 4], src_buf3[XF_NPIXPERCYCLE(NPC) + 4],
        src_buf4[XF_NPIXPERCYCLE(NPC) + 4], src_buf5[XF_NPIXPERCYCLE(NPC) + 4];
// clang-format off
#pragma HLS ARRAY_PARTITION variable=src_buf1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=src_buf2 complete dim=1
#pragma HLS ARRAY_PARTITION variable=src_buf3 complete dim=1
#pragma HLS ARRAY_PARTITION variable=src_buf4 complete dim=1
#pragma HLS ARRAY_PARTITION variable=src_buf5 complete dim=1
    // clang-format on

    XF_SNAME(WORDWIDTH_SRC) tmp_in;
    XF_SNAME(WORDWIDTH_DST) inter_valx = 0, inter_valy = 0;
    // Temporary buffer to hold image data from five rows
    XF_SNAME(WORDWIDTH_SRC) buf[5][(COLS >> XF_BITSHIFT(NPC))];
    if (USE_URAM) {
// clang-format off
#pragma HLS RESOURCE variable=buf core=RAM_S2P_URAM
#pragma HLS array reshape variable=buf dim=1 factor=5 cyclic
        // clang-format on
    } else {
// clang-format off
#pragma HLS RESOURCE variable=buf core=RAM_S2P_BRAM
#pragma HLS ARRAY_PARTITION variable=buf complete dim=1
        // clang-format on
    }

    row_ind = 2;

Clear_Row_Loop:
    for (col = 0; col < img_width; col++) {
// clang-format off
#pragma HLS LOOP_TRIPCOUNT min=TC max=TC
#pragma HLS pipeline
        // clang-format on
        buf[0][col] = 0;
        buf[1][col] = 0;
        buf[row_ind][col] = _src_mat.read(read_index++);
    }

    row_ind++;

Read_Row2_Loop:
    for (col = 0; col < img_width; col++) {
// clang-format off
#pragma HLS LOOP_TRIPCOUNT min=TC max=TC
#pragma HLS pipeline
        // clang-format on

        buf[row_ind][col] = _src_mat.read(read_index++);
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

        inter_valx = inter_valy = 0;

        ProcessSobel5x5<SRC_T, DST_T, ROWS, COLS, PLANES, DEPTH_SRC, DEPTH_DST, NPC, WORDWIDTH_SRC, WORDWIDTH_DST, TC>(
            _src_mat, _dst_matx, _dst_maty, buf, src_buf1, src_buf2, src_buf3, src_buf4, src_buf5, GradientValuesX,
            GradientValuesY, inter_valx, inter_valy, img_width, img_height, row_ind, shift_x, shift_y, tp1, tp2, mid,
            bottom1, bottom2, row, read_index, write_index);

        if ((NPC == XF_NPPC8) || (NPC == XF_NPPC16)) {
            for (ap_uint<6> i = 4; i < (XF_NPIXPERCYCLE(NPC) + 4); i++) {
                src_buf1[i] = 0;
                src_buf2[i] = 0;
                src_buf3[i] = 0;
                src_buf4[i] = 0;
                src_buf5[i] = 0;
            }

            GradientValuesX[0] = xFGradientX5x5<PLANES, DEPTH_SRC, DEPTH_DST>(&src_buf1[0], &src_buf2[0], &src_buf3[0],
                                                                              &src_buf4[0], &src_buf5[0]);
            GradientValuesX[1] = xFGradientX5x5<PLANES, DEPTH_SRC, DEPTH_DST>(&src_buf1[1], &src_buf2[1], &src_buf3[1],
                                                                              &src_buf4[1], &src_buf5[1]);
            GradientValuesY[0] = xFGradientY5x5<PLANES, DEPTH_SRC, DEPTH_DST>(&src_buf1[0], &src_buf2[0], &src_buf3[0],
                                                                              &src_buf4[0], &src_buf5[0]);
            GradientValuesY[1] = xFGradientY5x5<PLANES, DEPTH_SRC, DEPTH_DST>(&src_buf1[1], &src_buf2[1], &src_buf3[1],
                                                                              &src_buf4[1], &src_buf5[1]);

            xfPackPixels<NPC, WORDWIDTH_DST, DEPTH_DST>(&GradientValuesX[0], inter_valx, 0, 2, shift_x);
            xfPackPixels<NPC, WORDWIDTH_DST, DEPTH_DST>(&GradientValuesY[0], inter_valy, 0, 2, shift_y);

            _dst_matx.write(write_index, inter_valx);
            _dst_maty.write(write_index++, inter_valy);
        } else {
// clang-format off
#pragma HLS ALLOCATION function instances=xFGradientX5x5<PLANES, DEPTH_SRC, DEPTH_DST> limit=1 
#pragma HLS ALLOCATION function instances=xFGradientY5x5<PLANES, DEPTH_SRC, DEPTH_DST> limit=1
            // clang-format on

            src_buf1[buf_size - 1] = 0;
            src_buf2[buf_size - 1] = 0;
            src_buf3[buf_size - 1] = 0;
            src_buf4[buf_size - 1] = 0;
            src_buf5[buf_size - 1] = 0;

            GradientValuesX[0] = xFGradientX5x5<PLANES, DEPTH_SRC, DEPTH_DST>(&src_buf1[0], &src_buf2[0], &src_buf3[0],
                                                                              &src_buf4[0], &src_buf5[0]);
            GradientValuesY[0] = xFGradientY5x5<PLANES, DEPTH_SRC, DEPTH_DST>(&src_buf1[0], &src_buf2[0], &src_buf3[0],
                                                                              &src_buf4[0], &src_buf5[0]);
            inter_valx((max_loop - 1), (max_loop - step)) = GradientValuesX[0];
            inter_valy((max_loop - 1), (max_loop - step)) = GradientValuesY[0];
            _dst_matx.write(write_index, inter_valx);
            _dst_maty.write(write_index++, inter_valy);

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
            src_buf1[buf_size - 1] = 0;
            src_buf2[buf_size - 1] = 0;
            src_buf3[buf_size - 1] = 0;
            src_buf4[buf_size - 1] = 0;
            src_buf5[buf_size - 1] = 0;

            GradientValuesX[0] = xFGradientX5x5<PLANES, DEPTH_SRC, DEPTH_DST>(&src_buf1[0], &src_buf2[0], &src_buf3[0],
                                                                              &src_buf4[0], &src_buf5[0]);
            GradientValuesY[0] = xFGradientY5x5<PLANES, DEPTH_SRC, DEPTH_DST>(&src_buf1[0], &src_buf2[0], &src_buf3[0],
                                                                              &src_buf4[0], &src_buf5[0]);

            inter_valx((max_loop - 1), (max_loop - step)) = GradientValuesX[0];
            inter_valy((max_loop - 1), (max_loop - step)) = GradientValuesY[0];
            _dst_matx.write(write_index, inter_valx);
            _dst_maty.write(write_index++, inter_valy);
        }

        row_ind++;

        if (row_ind == 5) {
            row_ind = 0;
        }
    } // Row_Loop
}
// xFSobelFilter5x5

/*******************************************************************************
 * 			SobelFilter7x7
 *******************************************************************************
 *      SobelFilter X-Gradient used is 7X7
 *
 *       --- ---- ---- ---- ---  ---- ---  ----
 *      |  -1 |  -4 |   -5 | 0 |   5 |  4 |  1 |
 *       --- ---- ---- ---- ---  ---- ---  ----
 *      |  -6 | -24 |  -30 | 0 |  30 | 24 |  6 |
 *       --- ---- ---- ---- ---  ---- ---  ----
 *      | -15 | -60 |  -75 | 0 |  75 | 60 | 15 |
 *       --- ---- ---- ---- ---  ---- ---  ----
 *      | -20 | -80 | -100 | 0 | 100 | 80 | 20 |
 *       --- ---- ---- ---- ---  ---- ---  ----
 *      | -15 | -60 |  -75 | 0 |  75 | 60 | 15 |
 *       --- ---- ---- ---- ---  ---- ---  ----
 *      |  -6 | -24 |  -30 | 0 |  30 | 24 |  6 |
 *       --- ---- ---- ---- ---  ---- ---  ----
 *      |  -1 |  -4 |   -5 | 0 |   5 |  4 |  1 |
 *       --- ---- ---- ---- ---  ---- ---  ----
 ******************************************************************************/
template <int PLANES, int DEPTH_SRC, int DEPTH_DST>
XF_PTNAME(DEPTH_DST)
xFGradientX7x7(XF_PTNAME(DEPTH_SRC) * src_buf1,
               XF_PTNAME(DEPTH_SRC) * src_buf2,
               XF_PTNAME(DEPTH_SRC) * src_buf3,
               XF_PTNAME(DEPTH_SRC) * src_buf4,
               XF_PTNAME(DEPTH_SRC) * src_buf5,
               XF_PTNAME(DEPTH_SRC) * src_buf6,
               XF_PTNAME(DEPTH_SRC) * src_buf7) {
// clang-format off
#pragma HLS INLINE off
#pragma HLS PIPELINE II=1
    // clang-format on

    XF_PTNAME(DEPTH_DST) g_x = 0;
    XF_PTNAME(DEPTH_DST) val = 0;
    int STEP, STEP_OUT, p = 0;
    if ((DEPTH_DST == XF_48SP) || (DEPTH_DST == XF_16SP)) {
        STEP_OUT = 16;
        STEP = 8;
    } else if ((DEPTH_DST == XF_32SP)) {
        STEP_OUT = 32;
        STEP = 8;
    } else {
        STEP = 8;
        STEP_OUT = 8;
    }

    for (int i = 0, k = 0; i < PLANES; i++, k += STEP) {
        int Res = 0;
        ap_int<20> M00 = (ap_int<20>)(((ap_int<20>)src_buf1[6].range(k + STEP - 1, k) +
                                       (ap_int<20>)src_buf7[6].range(k + STEP - 1, k)) -
                                      ((ap_int<20>)src_buf1[0].range(k + STEP - 1, k) +
                                       (ap_int<20>)src_buf7[0].range(k + STEP - 1, k)));
        ap_int<20> M01 = (ap_int<20>)(((ap_int<20>)src_buf1[1].range(k + STEP - 1, k) +
                                       (ap_int<20>)src_buf7[1].range(k + STEP - 1, k))
                                      << 2);
        ap_int<20> A00 = (ap_int<20>)(((ap_int<20>)src_buf1[5].range(k + STEP - 1, k) +
                                       (ap_int<20>)src_buf7[5].range(k + STEP - 1, k))
                                      << 2);
        ap_int<20> M02 =
            (ap_int<20>)(((ap_int<20>)src_buf1[2].range(k + STEP - 1, k) +
                          (ap_int<20>)src_buf7[2].range(k + STEP - 1, k))
                         << 2) +
            (ap_int<20>)((ap_int<20>)src_buf1[2].range(k + STEP - 1, k) +
                         (ap_int<20>)src_buf7[2].range(k + STEP - 1, k)); //(src_buf1[2] + src_buf7[2]) * 5;
        ap_int<20> A01 = (ap_int<20>)(((ap_int<20>)src_buf1[4].range(k + STEP - 1, k) +
                                       (ap_int<20>)src_buf7[4].range(k + STEP - 1, k))
                                      << 2) +
                         (ap_int<20>)src_buf1[4].range(k + STEP - 1, k) +
                         (ap_int<20>)src_buf7[4].range(k + STEP - 1,
                                                       k); //(src_buf1[4] + src_buf7[4]) * 5;
        ap_int<20> M03 = (ap_int<20>)(((ap_int<20>)src_buf2[0].range(k + STEP - 1, k) +
                                       (ap_int<20>)src_buf6[0].range(k + STEP - 1, k))
                                      << 2) +
                         (ap_int<20>)(((ap_int<20>)src_buf2[0].range(k + STEP - 1, k) +
                                       (ap_int<20>)src_buf6[0].range(k + STEP - 1, k))
                                      << 1); //(src_buf2[0] + src_buf6[0]) * 6;
        ap_int<20> A02 = (ap_int<20>)(((ap_int<20>)src_buf2[6].range(k + STEP - 1, k) +
                                       (ap_int<20>)src_buf6[6].range(k + STEP - 1, k))
                                      << 2) +
                         (ap_int<20>)(((ap_int<20>)src_buf2[6].range(k + STEP - 1, k) +
                                       (ap_int<20>)src_buf6[6].range(k + STEP - 1, k))
                                      << 1); //(src_buf2[6] + src_buf6[6]) * 6;
        ap_int<20> M04 = (ap_int<20>)(((ap_int<20>)src_buf2[1].range(k + STEP - 1, k) +
                                       (ap_int<20>)src_buf6[1].range(k + STEP - 1, k))
                                      << 4) +
                         (ap_int<20>)(((ap_int<20>)src_buf2[1].range(k + STEP - 1, k) +
                                       (ap_int<20>)src_buf6[1].range(k + STEP - 1, k))
                                      << 3); //(src_buf2[1] + src_buf6[1]) * 24;
        ap_int<20> A03 = (ap_int<20>)(((ap_int<20>)src_buf2[5].range(k + STEP - 1, k) +
                                       (ap_int<20>)src_buf6[5].range(k + STEP - 1, k))
                                      << 4) +
                         (ap_int<20>)(((ap_int<20>)src_buf2[5].range(k + STEP - 1, k) +
                                       (ap_int<20>)src_buf6[5].range(k + STEP - 1, k))
                                      << 3); //(src_buf2[5] + src_buf6[5]) * 24;
        ap_int<20> M05 = (ap_int<20>)(((ap_int<20>)src_buf2[2].range(k + STEP - 1, k) +
                                       (ap_int<20>)src_buf6[2].range(k + STEP - 1, k))
                                      << 5) -
                         (ap_int<20>)(((ap_int<20>)src_buf2[2].range(k + STEP - 1, k) +
                                       (ap_int<20>)src_buf6[2].range(k + STEP - 1, k))
                                      << 1); //(src_buf2[2] + src_buf6[2]) * 30;
        ap_int<20> A04 = (ap_int<20>)(((ap_int<20>)src_buf2[4].range(k + STEP - 1, k) +
                                       (ap_int<20>)src_buf6[4].range(k + STEP - 1, k))
                                      << 5) -
                         (ap_int<20>)(((ap_int<20>)src_buf2[4].range(k + STEP - 1, k) +
                                       (ap_int<20>)src_buf6[4].range(k + STEP - 1, k))
                                      << 1); //(src_buf2[4] + src_buf6[4]) * 30;
        ap_int<20> M06 =
            (ap_int<20>)(((ap_int<20>)src_buf3[0].range(k + STEP - 1, k) +
                          (ap_int<20>)src_buf5[0].range(k + STEP - 1, k))
                         << 4) -
            (ap_int<20>)((ap_int<20>)src_buf3[0].range(k + STEP - 1, k) +
                         (ap_int<20>)src_buf5[0].range(k + STEP - 1, k)); //(src_buf3[0] + src_buf5[0]) * 15;
        ap_int<20> A05 =
            (ap_int<20>)(((ap_int<20>)src_buf3[6].range(k + STEP - 1, k) +
                          (ap_int<20>)src_buf5[6].range(k + STEP - 1, k))
                         << 4) -
            (ap_int<20>)((ap_int<20>)src_buf3[6].range(k + STEP - 1, k) +
                         (ap_int<20>)src_buf5[6].range(k + STEP - 1, k)); //(src_buf3[6] + src_buf5[6]) * 15;
        ap_int<20> M07 = (ap_int<20>)(((ap_int<20>)src_buf3[1].range(k + STEP - 1, k) +
                                       (ap_int<20>)src_buf5[1].range(k + STEP - 1, k))
                                      << 6) -
                         (ap_int<20>)(((ap_int<20>)src_buf3[1].range(k + STEP - 1, k) +
                                       (ap_int<20>)src_buf5[1].range(k + STEP - 1, k))
                                      << 2); //(src_buf3[1] + src_buf5[1]) * 60;
        ap_int<20> A06 = (ap_int<20>)(((ap_int<20>)src_buf3[5].range(k + STEP - 1, k) +
                                       (ap_int<20>)src_buf5[5].range(k + STEP - 1, k))
                                      << 6) -
                         (ap_int<20>)(((ap_int<20>)src_buf3[5].range(k + STEP - 1, k) +
                                       (ap_int<20>)src_buf5[5].range(k + STEP - 1, k))
                                      << 2); //(src_buf3[5] + src_buf5[5]) * 60;
        ap_int<20> M08 =
            (ap_int<20>)(((ap_int<20>)src_buf3[2].range(k + STEP - 1, k) +
                          (ap_int<20>)src_buf5[2].range(k + STEP - 1, k))
                         << 6) +
            (ap_int<20>)(((ap_int<20>)src_buf3[2].range(k + STEP - 1, k) +
                          (ap_int<20>)src_buf5[2].range(k + STEP - 1, k))
                         << 3) +
            (ap_int<20>)((ap_int<20>)src_buf3[2].range(k + STEP - 1, k) + (ap_int<20>)src_buf5[2].range(k + STEP - 1, k)
                         << 1) +
            (ap_int<20>)src_buf3[2].range(k + STEP - 1, k) +
            (ap_int<20>)src_buf5[2].range(k + STEP - 1,
                                          k); //(src_buf3[2] + src_buf5[2]) * 75;
        ap_int<20> A07 =
            (ap_int<20>)(((ap_int<20>)src_buf3[4].range(k + STEP - 1, k) +
                          (ap_int<20>)src_buf5[4].range(k + STEP - 1, k))
                         << 6) +
            (ap_int<20>)(((ap_int<20>)src_buf3[4].range(k + STEP - 1, k) +
                          (ap_int<20>)src_buf5[4].range(k + STEP - 1, k))
                         << 3) +
            (ap_int<20>)((ap_int<20>)src_buf3[4].range(k + STEP - 1, k) + (ap_int<20>)src_buf5[4].range(k + STEP - 1, k)
                         << 1) +
            (ap_int<20>)src_buf3[4].range(k + STEP - 1, k) +
            (ap_int<20>)src_buf5[4].range(k + STEP - 1,
                                          k); //(src_buf3[4] + src_buf5[4]) * 75;
        ap_int<20> M09 = (ap_int<20>)(((ap_int<20>)src_buf4[6].range(k + STEP - 1, k) -
                                       (ap_int<20>)src_buf4[0].range(k + STEP - 1, k))
                                      << 4) +
                         (ap_int<20>)(((ap_int<20>)src_buf4[6].range(k + STEP - 1, k) -
                                       (ap_int<20>)src_buf4[0].range(k + STEP - 1, k))
                                      << 2); //(src_buf4[6] - src_buf4[0]) * 20;
        ap_int<20> M10 = (ap_int<20>)(((ap_int<20>)src_buf4[5].range(k + STEP - 1, k) -
                                       (ap_int<20>)src_buf4[1].range(k + STEP - 1, k))
                                      << 6) +
                         (ap_int<20>)(((ap_int<20>)src_buf4[5].range(k + STEP - 1, k) -
                                       (ap_int<20>)src_buf4[1].range(k + STEP - 1, k))
                                      << 4); //(src_buf4[5] - src_buf4[1]) * 80;
        ap_int<20> M11 =
            (ap_int<20>)(((ap_int<20>)src_buf4[4].range(k + STEP - 1, k) -
                          (ap_int<20>)src_buf4[2].range(k + STEP - 1, k))
                         << 6) +
            (ap_int<20>)(((ap_int<20>)src_buf4[4].range(k + STEP - 1, k) -
                          (ap_int<20>)src_buf4[2].range(k + STEP - 1, k))
                         << 5) +
            (ap_int<20>)((ap_int<20>)src_buf4[4].range(k + STEP - 1, k) - (ap_int<20>)src_buf4[2].range(k + STEP - 1, k)
                         << 2); //(src_buf4[4] - src_buf4[2]) * 100;
        ap_int<20> FS00 = M01 + M02 + M03;
        ap_int<20> FS01 = M04 + M05;
        ap_int<20> FS02 = M06 + M07 + M08;
        ap_int<20> FA00 = A00 + A01;
        ap_int<20> FA01 = A02 + A03;
        ap_int<20> FA02 = A04 + A05;
        ap_int<20> FA03 = A06 + A07;
        ap_int<20> FA04 = M09 + M10 + M11;
        ap_int<20> FS0 = FS00 + FS01 + FS02;
        ap_int<20> FA0 = M00 + FA00 + FA01;
        ap_int<20> FA1 = FA02 + FA03 + FA04;

        Res = (FA0 + FA1) - (FS0);
        g_x = (XF_PTNAME(DEPTH_DST))Res;

        if ((DEPTH_DST == XF_8UP) || (DEPTH_DST == XF_24UP)) {
            if (Res < 0)
                g_x = 0;
            else if (Res > 255)
                g_x = 255;
        }

        if ((DEPTH_DST == XF_16SP) || (DEPTH_DST == XF_48SP)) {
            if (Res > 32767)
                g_x = 32767;
            else if (Res < -32768)
                g_x = -32768;
        }

        val.range(p + (STEP_OUT - 1), p) = g_x;
        p += STEP_OUT;
    }

    return val;
}

/********************************************************************
 *     SobelFilter Y-Gradient used is 7X7
 *
 *       --- ---- ---- ---- ---  ---- ---  ----
 *      | -1 |  -6 | -15 | -20 | -15 | -6 | -1 |
 *       --- ---- ---- ---- ---  ---- ---  ----
 *      | -4 | -24 | -60 | -80 | -60 |-24 | -4 |
 *       --- ---- ---- ---- ---  ---- ---  ----
 *      | -5 | -30 | -75 |-100 | -75 |-30 | -5 |
 *       --- ---- ---- ---- ---  ---- ---  ----
 *      |  0 |   0 |   0 |   0 |   0 |  0 |  0 |
 *       --- ---- ---- ---- ---  ---- ---  ----
 *      |  5 |  30 |  75 | 100 |  75 | 30 |  5 |
 *       --- ---- ---- ---- ---  ---- ---  ----
 *      |  4 |  24 |  60 |  80 |  60 | 24 |  4 |
 *       --- ---- ---- ---- ---  ---- ---  ----
 *      |  1 |   6 |  15 |  20 |  15 |  6 |  1 |
 *       --- ---- ---- ---- ---  ---- ---  ----
 ******************************************************************/
template <int PLANES, int DEPTH_SRC, int DEPTH_DST>
XF_PTNAME(DEPTH_DST)
xFGradientY7x7(XF_PTNAME(DEPTH_SRC) * src_buf1,
               XF_PTNAME(DEPTH_SRC) * src_buf2,
               XF_PTNAME(DEPTH_SRC) * src_buf3,
               XF_PTNAME(DEPTH_SRC) * src_buf4,
               XF_PTNAME(DEPTH_SRC) * src_buf5,
               XF_PTNAME(DEPTH_SRC) * src_buf6,
               XF_PTNAME(DEPTH_SRC) * src_buf7) {
// clang-format off
#pragma HLS INLINE off
#pragma HLS PIPELINE II=1
    // clang-format on
    XF_PTNAME(DEPTH_DST) g_y = 0, val = 0;
    int STEP, STEP_OUT, p = 0;
    if ((DEPTH_DST == XF_48SP) || (DEPTH_DST == XF_16SP)) {
        STEP_OUT = 16;
        STEP = 8;
    } else if ((DEPTH_DST == XF_32SP)) {
        STEP_OUT = 32;
        STEP = 8;
    } else {
        STEP = 8;
        STEP_OUT = 8;
    }

    for (int i = 0, k = 0; i < PLANES; i++, k += STEP) {
        int Res = 0;
        ap_int<20> M00 = (src_buf7[0].range(k + STEP - 1, k) + src_buf7[6].range(k + STEP - 1, k)) -
                         (src_buf1[0].range(k + STEP - 1, k) + src_buf1[6].range(k + STEP - 1, k));
        ap_int<20> M01 = ((ap_int<20>)(src_buf1[1].range(k + STEP - 1, k) + src_buf1[5].range(k + STEP - 1, k)) << 2) +
                         ((ap_int<20>)(src_buf1[1].range(k + STEP - 1, k) + src_buf1[5].range(k + STEP - 1, k))
                          << 1); //(src_buf1[1] + src_buf1[5]) * 6;
        ap_int<20> A00 = ((ap_int<20>)(src_buf7[1].range(k + STEP - 1, k) + src_buf7[5].range(k + STEP - 1, k)) << 2) +
                         ((ap_int<20>)(src_buf7[1].range(k + STEP - 1, k) + src_buf7[5].range(k + STEP - 1, k))
                          << 1); //(src_buf7[1] + src_buf7[5]) * 6;
        ap_int<20> M02 = ((ap_int<20>)(src_buf1[2].range(k + STEP - 1, k) + src_buf1[4].range(k + STEP - 1, k)) << 4) -
                         (src_buf1[2].range(k + STEP - 1, k) +
                          src_buf1[4].range(k + STEP - 1,
                                            k)); // (src_buf1[2] + src_buf1[4]) * 15;
        ap_int<20> A01 = ((ap_int<20>)(src_buf7[2].range(k + STEP - 1, k) + src_buf7[4].range(k + STEP - 1, k)) << 4) -
                         (src_buf7[2].range(k + STEP - 1, k) +
                          src_buf7[4].range(k + STEP - 1,
                                            k)); //(src_buf7[2] + src_buf7[4]) * 15;
        ap_int<20> M03 = (ap_int<20>)(src_buf2[0].range(k + STEP - 1, k) + src_buf2[6].range(k + STEP - 1, k)) << 2;
        ap_int<20> A02 = (ap_int<20>)(src_buf6[0].range(k + STEP - 1, k) + src_buf6[6].range(k + STEP - 1, k)) << 2;
        ap_int<20> M04 = ((ap_int<20>)(src_buf2[1].range(k + STEP - 1, k) + src_buf2[5].range(k + STEP - 1, k)) << 4) +
                         ((ap_int<20>)(src_buf2[1].range(k + STEP - 1, k) + src_buf2[5].range(k + STEP - 1, k))
                          << 3); //(src_buf2[1] + src_buf2[5]) * 24;
        ap_int<20> A03 = ((ap_int<20>)(src_buf6[1].range(k + STEP - 1, k) + src_buf6[5].range(k + STEP - 1, k)) << 4) +
                         ((ap_int<20>)(src_buf6[1].range(k + STEP - 1, k) + src_buf6[5].range(k + STEP - 1, k))
                          << 3); //(src_buf6[1] + src_buf6[5]) * 24;
        ap_int<20> M05 = ((ap_int<20>)(src_buf2[2].range(k + STEP - 1, k) + src_buf2[4].range(k + STEP - 1, k)) << 6) -
                         ((ap_int<20>)(src_buf2[2].range(k + STEP - 1, k) + src_buf2[4].range(k + STEP - 1, k))
                          << 2); //(src_buf2[2] + src_buf2[4]) * 60;
        ap_int<20> A04 = ((ap_int<20>)(src_buf6[2].range(k + STEP - 1, k) + src_buf6[4].range(k + STEP - 1, k)) << 6) -
                         ((ap_int<20>)(src_buf6[2].range(k + STEP - 1, k) + src_buf6[4].range(k + STEP - 1, k))
                          << 2); //(src_buf6[2] + src_buf6[4]) * 60;
        ap_int<20> M06 = ((ap_int<20>)(src_buf3[0].range(k + STEP - 1, k) + src_buf3[6].range(k + STEP - 1, k)) << 2) +
                         (src_buf3[0].range(k + STEP - 1, k) +
                          src_buf3[6].range(k + STEP - 1, k)); //(src_buf3[0] + src_buf3[6]) * 5;
        ap_int<20> A05 = ((ap_int<20>)(src_buf5[0].range(k + STEP - 1, k) + src_buf5[6].range(k + STEP - 1, k)) << 2) +
                         (src_buf5[0].range(k + STEP - 1, k) +
                          src_buf5[6].range(k + STEP - 1, k)); //(src_buf5[0] + src_buf5[6]) * 5;
        ap_int<20> M07 = ((ap_int<20>)(src_buf3[1].range(k + STEP - 1, k) + src_buf3[5].range(k + STEP - 1, k)) << 5) -
                         ((ap_int<20>)(src_buf3[1].range(k + STEP - 1, k) + src_buf3[5].range(k + STEP - 1, k))
                          << 1); //(src_buf3[1] + src_buf3[5]) * 30;
        ap_int<20> A06 = ((ap_int<20>)(src_buf5[1].range(k + STEP - 1, k) + src_buf5[5].range(k + STEP - 1, k)) << 5) -
                         ((ap_int<20>)(src_buf5[1].range(k + STEP - 1, k) + src_buf5[5].range(k + STEP - 1, k))
                          << 1); //(src_buf5[1] + src_buf5[5]) * 30;

        ap_int<20> M08 = ((ap_int<20>)(src_buf3[2].range(k + STEP - 1, k) + src_buf3[4].range(k + STEP - 1, k)) << 6) +
                         ((ap_int<20>)(src_buf3[2].range(k + STEP - 1, k) + src_buf3[4].range(k + STEP - 1, k)) << 3) +
                         ((ap_int<20>)(src_buf3[2].range(k + STEP - 1, k) + src_buf3[4].range(k + STEP - 1, k)) << 1) +
                         (src_buf3[2].range(k + STEP - 1, k) +
                          src_buf3[4].range(k + STEP - 1,
                                            k)); //(src_buf3[2] + src_buf3[4]) * 75;
        ap_int<20> A07 = ((ap_int<20>)(src_buf5[2].range(k + STEP - 1, k) + src_buf5[4].range(k + STEP - 1, k)) << 6) +
                         ((ap_int<20>)(src_buf5[2].range(k + STEP - 1, k) + src_buf5[4].range(k + STEP - 1, k)) << 3) +
                         ((ap_int<20>)(src_buf5[2].range(k + STEP - 1, k) + src_buf5[4].range(k + STEP - 1, k)) << 1) +
                         (src_buf5[2].range(k + STEP - 1, k) +
                          src_buf5[4].range(k + STEP - 1,
                                            k)); //(src_buf5[2] + src_buf5[4]) * 75;
        ap_int<20> M09 = ((ap_int<20>)(src_buf7[3].range(k + STEP - 1, k) - src_buf1[3].range(k + STEP - 1, k)) << 4) +
                         ((ap_int<20>)(src_buf7[3].range(k + STEP - 1, k) - src_buf1[3].range(k + STEP - 1, k))
                          << 2); //(src_buf7[3] - src_buf1[3]) * 20;
        ap_int<20> M10 = ((ap_int<20>)(src_buf6[3].range(k + STEP - 1, k) - src_buf2[3].range(k + STEP - 1, k)) << 6) +
                         ((ap_int<20>)(src_buf6[3].range(k + STEP - 1, k) - src_buf2[3].range(k + STEP - 1, k))
                          << 4); //(src_buf6[3] - src_buf2[3]) * 80;
        ap_int<20> M11 = ((ap_int<20>)(src_buf5[3].range(k + STEP - 1, k) - src_buf3[3].range(k + STEP - 1, k)) << 6) +
                         ((ap_int<20>)(src_buf5[3].range(k + STEP - 1, k) - src_buf3[3].range(k + STEP - 1, k)) << 5) +
                         ((ap_int<20>)(src_buf5[3].range(k + STEP - 1, k) - src_buf3[3].range(k + STEP - 1, k))
                          << 2); //(src_buf5[3] - src_buf3[3]) * 100;
        ap_int<20> FS00 = M01 + M02 + M03;
        ap_int<20> FS01 = M04 + M05;
        ap_int<20> FS02 = M06 + M07 + M08;
        ap_int<20> FA00 = A00 + A01;
        ap_int<20> FA01 = A02 + A03;
        ap_int<20> FA02 = A04 + A05;
        ap_int<20> FA03 = A06 + A07;
        ap_int<20> FA04 = M09 + M10 + M11;
        ap_int<20> FS0 = FS00 + FS01 + FS02;
        ap_int<20> FA0 = M00 + FA00 + FA01;
        ap_int<20> FA1 = FA02 + FA03 + FA04;
        Res = (FA0 + FA1) - (FS0);
        g_y = (XF_PTNAME(DEPTH_DST))Res;

        if ((DEPTH_DST == XF_8UP) || (DEPTH_DST == XF_24UP)) {
            if (Res < 0)
                g_y = 0;
            else if (Res > 255)
                g_y = 255;
        }

        if ((DEPTH_DST == XF_16SP) || (DEPTH_DST == XF_48SP)) {
            if (Res > 32767)
                g_y = 32767;
            else if (Res < -32768)
                g_y = -32768;
        }

        // g_y = (XF_PTNAME(DEPTH_DST))Res;

        val.range(p + (STEP_OUT - 1), p) = (XF_PTNAME(DEPTH_DST))g_y;
        p += STEP_OUT;
    }

    return val;
}

/**
 * xFSobel7x7 : Applies the mask and Computes the gradient values
 *              for filtersize 7x7
 */
template <int NPC, int PLANES, int DEPTH_SRC, int DEPTH_DST>
void xFSobel7x7(XF_PTNAME(DEPTH_DST) * GradientvaluesX,
                XF_PTNAME(DEPTH_DST) * GradientvaluesY,
                XF_PTNAME(DEPTH_SRC) src_buf1[XF_NPIXPERCYCLE(NPC) + 6],
                XF_PTNAME(DEPTH_SRC) src_buf2[XF_NPIXPERCYCLE(NPC) + 6],
                XF_PTNAME(DEPTH_SRC) src_buf3[XF_NPIXPERCYCLE(NPC) + 6],
                XF_PTNAME(DEPTH_SRC) src_buf4[XF_NPIXPERCYCLE(NPC) + 6],
                XF_PTNAME(DEPTH_SRC) src_buf5[XF_NPIXPERCYCLE(NPC) + 6],
                XF_PTNAME(DEPTH_SRC) src_buf6[XF_NPIXPERCYCLE(NPC) + 6],
                XF_PTNAME(DEPTH_SRC) src_buf7[XF_NPIXPERCYCLE(NPC) + 6]) {
// clang-format off
#pragma HLS INLINE
    // clang-format on
    for (ap_uint<9> j = 0; j < XF_NPIXPERCYCLE(NPC); j++) {
// clang-format off
#pragma HLS LOOP_TRIPCOUNT min=8 max=8
#pragma HLS UNROLL
        // clang-format on
        GradientvaluesX[j] = xFGradientX7x7<PLANES, DEPTH_SRC, DEPTH_DST>(
            &src_buf1[j], &src_buf2[j], &src_buf3[j], &src_buf4[j], &src_buf5[j], &src_buf6[j], &src_buf7[j]);

        GradientvaluesY[j] = xFGradientY7x7<PLANES, DEPTH_SRC, DEPTH_DST>(
            &src_buf1[j], &src_buf2[j], &src_buf3[j], &src_buf4[j], &src_buf5[j], &src_buf6[j], &src_buf7[j]);
    }
}

/**************************************************************************************
 * ProcessSobel7x7 : Computes gradients for the column input data
 **************************************************************************************/

template <int SRC_T,
          int DST_T,
          int ROWS,
          int COLS,
          int PLANES,
          int DEPTH_SRC,
          int DEPTH_DST,
          int NPC,
          int WORDWIDTH_SRC,
          int WORDWIDTH_DST,
          int TC>
void ProcessSobel7x7(xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _src_mat,
                     xf::cv::Mat<DST_T, ROWS, COLS, NPC>& _gradx_mat,
                     xf::cv::Mat<DST_T, ROWS, COLS, NPC>& _grady_mat,
                     XF_SNAME(WORDWIDTH_SRC) buf[7][(COLS >> XF_BITSHIFT(NPC))],
                     XF_PTNAME(DEPTH_SRC) src_buf1[XF_NPIXPERCYCLE(NPC) + 6],
                     XF_PTNAME(DEPTH_SRC) src_buf2[XF_NPIXPERCYCLE(NPC) + 6],
                     XF_PTNAME(DEPTH_SRC) src_buf3[XF_NPIXPERCYCLE(NPC) + 6],
                     XF_PTNAME(DEPTH_SRC) src_buf4[XF_NPIXPERCYCLE(NPC) + 6],
                     XF_PTNAME(DEPTH_SRC) src_buf5[XF_NPIXPERCYCLE(NPC) + 6],
                     XF_PTNAME(DEPTH_SRC) src_buf6[XF_NPIXPERCYCLE(NPC) + 6],
                     XF_PTNAME(DEPTH_SRC) src_buf7[XF_NPIXPERCYCLE(NPC) + 6],
                     XF_PTNAME(DEPTH_DST) GradientValuesX[XF_NPIXPERCYCLE(NPC)],
                     XF_PTNAME(DEPTH_DST) GradientValuesY[XF_NPIXPERCYCLE(NPC)],
                     XF_SNAME(WORDWIDTH_DST) & inter_valx,
                     XF_SNAME(WORDWIDTH_DST) & inter_valy,
                     uint16_t img_width,
                     uint16_t img_height,
                     ap_uint<13> row_ind,
                     uint16_t& shiftx,
                     uint16_t& shifty,
                     ap_uint<4> tp1,
                     ap_uint<4> tp2,
                     ap_uint<4> tp3,
                     ap_uint<4> mid,
                     ap_uint<4> bottom1,
                     ap_uint<4> bottom2,
                     ap_uint<4> bottom3,
                     ap_uint<13> row,
                     int& read_index,
                     int& write_index) {
// clang-format off
#pragma HLS INLINE
    // clang-format on
    XF_SNAME(WORDWIDTH_SRC) buf0, buf1, buf2, buf3, buf4, buf5, buf6;
    uint16_t npc = XF_NPIXPERCYCLE(NPC);
    ap_uint<10> max_loop = XF_WORDDEPTH(WORDWIDTH_DST);

Col_Loop:
    for (ap_uint<13> col = 0; col < img_width; col++) {
// clang-format off
#pragma HLS LOOP_TRIPCOUNT min=TC max=TC
#pragma HLS pipeline
        // clang-format on
        if (row < img_height)
            buf[row_ind][col] = _src_mat.read(read_index++);
        else
            buf[bottom3][col] = 0;
        buf0 = buf[tp1][col];
        buf1 = buf[tp2][col];
        buf2 = buf[tp3][col];
        buf3 = buf[mid][col];
        buf4 = buf[bottom1][col];
        buf5 = buf[bottom2][col];
        buf6 = buf[bottom3][col];

        if (row == 26 && col == 15) printf("hello");

        if (NPC == XF_NPPC8) {
            xfExtractData<NPC, WORDWIDTH_SRC, DEPTH_SRC>(src_buf1, src_buf2, src_buf3, src_buf4, src_buf5, src_buf6,
                                                         src_buf7, buf0, buf1, buf2, buf3, buf4, buf5, buf6);
        } else {
            src_buf1[6] = buf0;
            src_buf2[6] = buf1;
            src_buf3[6] = buf2;
            src_buf4[6] = buf3;
            src_buf5[6] = buf4;
            src_buf6[6] = buf5;
            src_buf7[6] = buf6;
        }
        xFSobel7x7<NPC, PLANES, DEPTH_SRC, DEPTH_DST>(GradientValuesX, GradientValuesY, src_buf1, src_buf2, src_buf3,
                                                      src_buf4, src_buf5, src_buf6, src_buf7);

        xfCopyData<NPC, DEPTH_SRC>(src_buf1, src_buf2, src_buf3, src_buf4, src_buf5, src_buf6, src_buf7);

        if (col == 0) {
            shiftx = 0;
            shifty = 0;
            inter_valx = 0;
            inter_valy = 0;

            xfPackPixels<NPC, WORDWIDTH_DST, DEPTH_DST>(&GradientValuesX[0], inter_valx, 3, (npc - 3), shiftx);
            xfPackPixels<NPC, WORDWIDTH_DST, DEPTH_DST>(&GradientValuesY[0], inter_valy, 3, (npc - 3), shifty);

        } else {
            if ((NPC == XF_NPPC8)) {
                xfPackPixels<NPC, WORDWIDTH_DST, DEPTH_DST>(&GradientValuesX[0], inter_valx, 0, 3, shiftx);
                xfPackPixels<NPC, WORDWIDTH_DST, DEPTH_DST>(&GradientValuesY[0], inter_valy, 0, 3, shifty);

                _gradx_mat.write(write_index, inter_valx);
                _grady_mat.write(write_index++, inter_valy);
                shiftx = 0;
                shifty = 0;
                inter_valx = 0;
                inter_valy = 0;

                xfPackPixels<NPC, WORDWIDTH_DST, DEPTH_DST>(&GradientValuesX[0], inter_valx, 3, (npc - 3), shiftx);
                xfPackPixels<NPC, WORDWIDTH_DST, DEPTH_DST>(&GradientValuesY[0], inter_valy, 3, (npc - 3), shifty);

            } else {
                if (col >= 3) {
                    inter_valx((max_loop - 1), (max_loop - XF_PIXELDEPTH(DEPTH_DST))) = GradientValuesX[0];
                    inter_valy((max_loop - 1), (max_loop - XF_PIXELDEPTH(DEPTH_DST))) = GradientValuesY[0];
                    _gradx_mat.write(write_index, inter_valx);
                    _grady_mat.write(write_index++, inter_valy);
                }
            }
        }
    } // Col_Loop
}

template <int SRC_T,
          int DST_T,
          int ROWS,
          int COLS,
          int PLANES,
          int DEPTH_SRC,
          int DEPTH_DST,
          int NPC,
          int WORDWIDTH_SRC,
          int WORDWIDTH_DST,
          int TC>
void RightBorder7x7(xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _src_mat,
                    xf::cv::Mat<DST_T, ROWS, COLS, NPC>& _gradx_mat,
                    xf::cv::Mat<DST_T, ROWS, COLS, NPC>& _grady_mat,
                    XF_PTNAME(DEPTH_SRC) src_buf1[XF_NPIXPERCYCLE(NPC) + 6],
                    XF_PTNAME(DEPTH_SRC) src_buf2[XF_NPIXPERCYCLE(NPC) + 6],
                    XF_PTNAME(DEPTH_SRC) src_buf3[XF_NPIXPERCYCLE(NPC) + 6],
                    XF_PTNAME(DEPTH_SRC) src_buf4[XF_NPIXPERCYCLE(NPC) + 6],
                    XF_PTNAME(DEPTH_SRC) src_buf5[XF_NPIXPERCYCLE(NPC) + 6],
                    XF_PTNAME(DEPTH_SRC) src_buf6[XF_NPIXPERCYCLE(NPC) + 6],
                    XF_PTNAME(DEPTH_SRC) src_buf7[XF_NPIXPERCYCLE(NPC) + 6],
                    XF_PTNAME(DEPTH_DST) GradientValuesX[XF_NPIXPERCYCLE(NPC)],
                    XF_PTNAME(DEPTH_DST) GradientValuesY[XF_NPIXPERCYCLE(NPC)],
                    XF_SNAME(WORDWIDTH_DST) & inter_valx,
                    XF_SNAME(WORDWIDTH_DST) & inter_valy,
                    uint16_t& shiftx,
                    uint16_t& shifty,
                    int& read_index,
                    int& write_index) {
    //#pragma HLS INLINE off
    ap_uint<4> i = 0;
    ap_uint<5> buf_size = (XF_NPIXPERCYCLE(NPC) + 6);
    ap_uint<10> max_loop = XF_WORDDEPTH(WORDWIDTH_DST);

    if ((NPC == XF_NPPC8)) {
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

            GradientValuesX[i] = xFGradientX7x7<PLANES, DEPTH_SRC, DEPTH_DST>(
                &src_buf1[i], &src_buf2[i], &src_buf3[i], &src_buf4[i], &src_buf5[i], &src_buf6[i], &src_buf7[i]);

            GradientValuesY[i] = xFGradientY7x7<PLANES, DEPTH_SRC, DEPTH_DST>(
                &src_buf1[i], &src_buf2[i], &src_buf3[i], &src_buf4[i], &src_buf5[i], &src_buf6[i], &src_buf7[i]);
        }
        xfPackPixels<NPC, WORDWIDTH_DST, DEPTH_DST>(&GradientValuesX[0], inter_valx, 0, 3, shiftx);
        xfPackPixels<NPC, WORDWIDTH_DST, DEPTH_DST>(&GradientValuesY[0], inter_valy, 0, 3, shifty);

        _gradx_mat.write(write_index, inter_valx);
        _grady_mat.write(write_index++, inter_valy);
        shiftx = 0;
        shifty = 0;
        inter_valx = 0;
        inter_valy = 0;
    } else {
        src_buf1[6] = 0;
        src_buf2[6] = 0;
        src_buf3[6] = 0;
        src_buf4[6] = 0;
        src_buf5[6] = 0;
        src_buf6[6] = 0;
        src_buf7[6] = 0;

        for (ap_uint<5> k = 0; k < 3; k++) {
// clang-format off
#pragma HLS LOOP_TRIPCOUNT min=3 max=3
#pragma HLS ALLOCATION function instances=xFGradientX7x7<PLANES, DEPTH_SRC, DEPTH_DST> limit=1 
#pragma HLS ALLOCATION function instances=xFGradientY7x7<PLANES, DEPTH_SRC, DEPTH_DST> limit=1
            // clang-format on

            XF_PTNAME(DEPTH_DST)
            x1 = xFGradientX7x7<PLANES, DEPTH_SRC, DEPTH_DST>(&src_buf1[0], &src_buf2[0], &src_buf3[0], &src_buf4[0],
                                                              &src_buf5[0], &src_buf6[0], &src_buf7[0]);

            XF_PTNAME(DEPTH_DST)
            y1 = xFGradientY7x7<PLANES, DEPTH_SRC, DEPTH_DST>(&src_buf1[0], &src_buf2[0], &src_buf3[0], &src_buf4[0],
                                                              &src_buf5[0], &src_buf6[0], &src_buf7[0]);

            xfCopyData<NPC, DEPTH_SRC>(src_buf1, src_buf2, src_buf3, src_buf4, src_buf5, src_buf6, src_buf7);
            inter_valx((max_loop - 1), (max_loop - XF_PIXELDEPTH(DEPTH_DST))) = x1; // GradientValuesX[0];
            inter_valy((max_loop - 1), (max_loop - XF_PIXELDEPTH(DEPTH_DST))) = y1; // GradientValuesY[0];
            _gradx_mat.write(write_index, inter_valx);
            _grady_mat.write(write_index++, inter_valy);
        }
    }
}
/**
 * xFSobelFilter7x7 : Computes Sobel gradient of the input image
 *                    for filter size 7x7
 * _src_mat		: Input image
 * _gradx_mat	: GradientX output
 * _grady_mat	: GradientY output
 */
template <int SRC_T,
          int DST_T,
          int ROWS,
          int COLS,
          int PLANES,
          int DEPTH_SRC,
          int DEPTH_DST,
          int NPC,
          int WORDWIDTH_SRC,
          int WORDWIDTH_DST,
          int TC,
          bool USE_URAM>
void xFSobelFilter7x7(xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _src_mat,
                      xf::cv::Mat<DST_T, ROWS, COLS, NPC>& _gradx_mat,
                      xf::cv::Mat<DST_T, ROWS, COLS, NPC>& _grady_mat,
                      uint16_t img_height,
                      uint16_t img_width) {
    ap_uint<13> row_ind, row, col;
    ap_uint<4> tp1, tp2, tp3, mid, bottom1, bottom2, bottom3;
    ap_uint<5> i;
    int read_index = 0, write_index = 0;

    // Gradient output values stored in these buffer
    XF_PTNAME(DEPTH_DST) GradientValuesX[XF_NPIXPERCYCLE(NPC)];
    XF_PTNAME(DEPTH_DST) GradientValuesY[XF_NPIXPERCYCLE(NPC)];

    if (NPC > 1) {
// clang-format off
#pragma HLS ARRAY_PARTITION variable=GradientValuesX complete dim=1
#pragma HLS ARRAY_PARTITION variable=GradientValuesY complete dim=1
        // clang-format on
    }

    // Temporary buffers to hold image data from three rows.
    XF_PTNAME(DEPTH_SRC)
    src_buf1[XF_NPIXPERCYCLE(NPC) + 6], src_buf2[XF_NPIXPERCYCLE(NPC) + 6], src_buf3[XF_NPIXPERCYCLE(NPC) + 6],
        src_buf4[XF_NPIXPERCYCLE(NPC) + 6], src_buf5[XF_NPIXPERCYCLE(NPC) + 6];
    XF_PTNAME(DEPTH_SRC)
    src_buf6[XF_NPIXPERCYCLE(NPC) + 6], src_buf7[XF_NPIXPERCYCLE(NPC) + 6];
// clang-format off
#pragma HLS ARRAY_PARTITION variable=src_buf1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=src_buf2 complete dim=1
#pragma HLS ARRAY_PARTITION variable=src_buf3 complete dim=1
#pragma HLS ARRAY_PARTITION variable=src_buf4 complete dim=1
#pragma HLS ARRAY_PARTITION variable=src_buf5 complete dim=1
#pragma HLS ARRAY_PARTITION variable=src_buf6 complete dim=1
#pragma HLS ARRAY_PARTITION variable=src_buf7 complete dim=1
    // clang-format on

    XF_SNAME(WORDWIDTH_DST) inter_valx = 0, inter_valy = 0;
    uint16_t shiftx = 0, shifty = 0;

    XF_SNAME(WORDWIDTH_SRC) buf[7][(COLS >> XF_BITSHIFT(NPC))];
    if (USE_URAM) {
// clang-format off
#pragma HLS RESOURCE variable=buf core=RAM_S2P_URAM
#pragma HLS array reshape variable=buf dim=1 factor=7 cyclic
        // clang-format on
    } else {
// clang-format off
#pragma HLS RESOURCE variable=buf core=RAM_S2P_BRAM
#pragma HLS ARRAY_PARTITION variable=buf complete dim=1
        // clang-format on
    }
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
        buf[row_ind][col] = _src_mat.read(read_index++);
    }
    row_ind++;

Read_Row1_Loop:
    for (col = 0; col < img_width; col++) {
// clang-format off
#pragma HLS LOOP_TRIPCOUNT min=TC max=TC
#pragma HLS pipeline
        // clang-format on

        buf[row_ind][col] = _src_mat.read(read_index++);
    }
    row_ind++;

Read_Row2_Loop:
    for (col = 0; col < img_width; col++) {
// clang-format off
#pragma HLS LOOP_TRIPCOUNT min=TC max=TC
#pragma HLS pipeline
        // clang-format on

        buf[row_ind][col] = _src_mat.read(read_index++);
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
        inter_valx = inter_valy = 0;
        /***********		Process complete row
         * **********/
        ProcessSobel7x7<SRC_T, DST_T, ROWS, COLS, PLANES, DEPTH_SRC, DEPTH_DST, NPC, WORDWIDTH_SRC, WORDWIDTH_DST, TC>(
            _src_mat, _gradx_mat, _grady_mat, buf, src_buf1, src_buf2, src_buf3, src_buf4, src_buf5, src_buf6, src_buf7,
            GradientValuesX, GradientValuesY, inter_valx, inter_valy, img_width, img_height, row_ind, shiftx, shifty,
            tp1, tp2, tp3, mid, bottom1, bottom2, bottom3, row, read_index, write_index);

        RightBorder7x7<SRC_T, DST_T, ROWS, COLS, PLANES, DEPTH_SRC, DEPTH_DST, NPC, WORDWIDTH_SRC, WORDWIDTH_DST, TC>(
            _src_mat, _gradx_mat, _grady_mat, src_buf1, src_buf2, src_buf3, src_buf4, src_buf5, src_buf6, src_buf7,
            GradientValuesX, GradientValuesY, inter_valx, inter_valy, shiftx, shifty, read_index, write_index);

        row_ind++;
        if (row_ind == 7) {
            row_ind = 0;
        }
    } // Row_Loop ends here
}
// xFSobelFilter7x7

template <int BORDER_TYPE,
          int FILTER_TYPE,
          int SRC_T,
          int DST_T,
          int ROWS,
          int COLS,
          int NPC = 1,
          bool USE_URAM = false>
void Sobel(xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _src_mat,
           xf::cv::Mat<DST_T, ROWS, COLS, NPC>& _dst_matx,
           xf::cv::Mat<DST_T, ROWS, COLS, NPC>& _dst_maty) {
// clang-format off
#pragma HLS INLINE OFF
    // clang-format on

    uint16_t width = _src_mat.cols >> XF_BITSHIFT(NPC);
    uint16_t height = _src_mat.rows;

#ifndef __SYNTHESIS__
    assert(((FILTER_TYPE == XF_FILTER_3X3) || (FILTER_TYPE == XF_FILTER_5X5) || (FILTER_TYPE == XF_FILTER_7X7)) &&
           " Filter width must be XF_FILTER_3X3, XF_FILTER_5X5 or XF_FILTER_7X7 ");

    assert(((NPC == XF_NPPC1) || (NPC == XF_NPPC8)) && "NPC must be XF_NPPC1 or XF_NPPC8");

    assert((BORDER_TYPE == XF_BORDER_CONSTANT) && "Border type must be XF_BORDER_CONSTANT ");

    assert(((_src_mat.rows <= ROWS) && (_src_mat.cols <= COLS)) && "ROWS and COLS should be greater than input image");
#endif
    if (FILTER_TYPE == XF_FILTER_3X3) {
        xFSobelFilter3x3<SRC_T, DST_T, ROWS, COLS, XF_CHANNELS(SRC_T, NPC), XF_DEPTH(SRC_T, NPC), XF_DEPTH(DST_T, NPC),
                         NPC, XF_WORDWIDTH(SRC_T, NPC), XF_WORDWIDTH(DST_T, NPC), (COLS >> XF_BITSHIFT(NPC)), USE_URAM>(
            _src_mat, _dst_matx, _dst_maty, height, width);
    }

    else if (FILTER_TYPE == XF_FILTER_5X5) {
        xFSobelFilter5x5<SRC_T, DST_T, ROWS, COLS, XF_CHANNELS(SRC_T, NPC), XF_DEPTH(SRC_T, NPC), XF_DEPTH(DST_T, NPC),
                         NPC, XF_WORDWIDTH(SRC_T, NPC), XF_WORDWIDTH(DST_T, NPC), (COLS >> XF_BITSHIFT(NPC)), USE_URAM>(
            _src_mat, _dst_matx, _dst_maty, height, width);
    }

    else if (FILTER_TYPE == XF_FILTER_7X7) {
        xFSobelFilter7x7<SRC_T, DST_T, ROWS, COLS, XF_CHANNELS(SRC_T, NPC), XF_DEPTH(SRC_T, NPC), XF_DEPTH(DST_T, NPC),
                         NPC, XF_WORDWIDTH(SRC_T, NPC), XF_WORDWIDTH(DST_T, NPC), (COLS >> XF_BITSHIFT(NPC)), USE_URAM>(
            _src_mat, _dst_matx, _dst_maty, height, width);
    }
}
} // namespace cv
} // namespace xf
// xFSobelFilter
#endif // _XF_SOBEL_HPP_
