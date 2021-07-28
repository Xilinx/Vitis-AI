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

#ifndef _XF_FAST_HPP_
#define _XF_FAST_HPP_

#include "hls_stream.h"
#include "common/xf_common.hpp"
#include "common/xf_utility.hpp"

#define __MIN(a, b) ((a < b) ? a : b)
#define __MAX(a, b) ((a > b) ? a : b)

#define PSize 16
#define NUM 25

namespace xf {
namespace cv {

// coreScore computes the score for corner pixels
// For a given pixel identified as corner in process_function, the theshold is
// increaded by a small value in each iteration till the pixel becomes
// a non-corner. That value of threshold becomes the score for that corner pixel.
static void xFCoreScore(short int* flag_d, int _threshold, uchar_t* core) {
// clang-format off
    #pragma HLS INLINE
    // clang-format on
    short int flag_d_min2[NUM - 1];
    short int flag_d_max2[NUM - 1];
    short int flag_d_min4[NUM - 3];
    short int flag_d_max4[NUM - 3];
    short int flag_d_min8[NUM - 7];
    short int flag_d_max8[NUM - 7];

    for (ap_uint<5> i = 0; i < NUM - 1; i++) {
        flag_d_min2[i] = __MIN(flag_d[i], flag_d[i + 1]);
        flag_d_max2[i] = __MAX(flag_d[i], flag_d[i + 1]);
    }

    for (ap_uint<5> i = 0; i < NUM - 3; i++) {
        flag_d_min4[i] = __MIN(flag_d_min2[i], flag_d_min2[i + 2]);
        flag_d_max4[i] = __MAX(flag_d_max2[i], flag_d_max2[i + 2]);
    }

    for (ap_uint<5> i = 0; i < NUM - 7; i++) {
        flag_d_min8[i] = __MIN(flag_d_min4[i], flag_d_min4[i + 4]);
        flag_d_max8[i] = __MAX(flag_d_max4[i], flag_d_max4[i + 4]);
    }

    uchar_t a0 = _threshold;

    for (ap_uint<5> i = 0; i < PSize; i += 2) {
        short int a = 255;
        if (PSize == 16) {
            a = flag_d_min8[i + 1];
        }
        //		else {
        //			for(ap_uint<5> j=1;j<PSize/2+1;j++)
        //			{
        //				a=__MIN(a,flag_d[i+j]);
        //			}
        //		}
        a0 = __MAX(a0, __MIN(a, flag_d[i])); // a0 >= _threshold
        a0 = __MAX(a0, __MIN(a, flag_d[i + PSize / 2 + 1]));
    }
    short int b0 = -_threshold;
    for (ap_uint<5> i = 0; i < PSize; i += 2) {
        short int b = -255;
        if (PSize == 16) {
            b = flag_d_max8[i + 1];
        }
        //		} else {
        //			for(ap_uint<5> j=1;j<PSize/2+1;j++)
        //			{
        //				b=__MAX(b,flag_d[i+j]);
        //			}
        //		}
        b0 = __MIN(b0, __MAX(b, flag_d[i])); // b0 <= -_threshold
        b0 = __MIN(b0, __MAX(b, flag_d[i + PSize / 2 + 1]));
    }
    *core = __MAX(a0, -b0) - 1;
} // Core window score computation complete

template <int NPC, int WORDWIDTH, int DEPTH, int WIN_SZ, int WIN_SZ_SQ>
void xFfastProc(XF_PTNAME(DEPTH) OutputValues[XF_NPIXPERCYCLE(NPC)],
                XF_PTNAME(DEPTH) src_buf[WIN_SZ][XF_NPIXPERCYCLE(NPC) + (WIN_SZ - 1)],
                ap_uint<8> win_size,
                uchar_t _threshold,
                XF_PTNAME(DEPTH) & pack_corners) {
// clang-format off
    #pragma HLS INLINE
    // clang-format on

    uchar_t kx = 0, ix = 0;

    // XF_SNAME(WORDWIDTH) tbuf_temp;
    XF_PTNAME(DEPTH) tbuf_temp = 0;

    ////////////////////////////////////////////////
    // Main code goes here
    // Bresenham's circle score computation
    short int flag_d[(1 << XF_BITSHIFT(NPC))][NUM] = {0}, flag_val[(1 << XF_BITSHIFT(NPC))][NUM] = {0};

// clang-format off
    #pragma HLS ARRAY_PARTITION variable=flag_val dim=1
    #pragma HLS ARRAY_PARTITION variable=flag_d dim=1
    // clang-format on

    for (ap_uint<4> i = 0; i < 1; i++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT MAX=1
        #pragma HLS LOOP_FLATTEN off
        #pragma HLS PIPELINE II=1
        // clang-format on
        // Compute the intensity difference between the candidate pixel and pixels on the Bresenham's circle
        flag_d[i][0] = src_buf[3][3 + i] - src_buf[0][3 + i];  // tbuf4[3+i] - tbuf1[3+i];
        flag_d[i][1] = src_buf[3][3 + i] - src_buf[0][4 + i];  // tbuf4[3+i] - tbuf1[4+i];
        flag_d[i][2] = src_buf[3][3 + i] - src_buf[1][5 + i];  // tbuf4[3+i] - tbuf2[5+i];
        flag_d[i][3] = src_buf[3][3 + i] - src_buf[2][6 + i];  // tbuf4[3+i] - tbuf3[6+i];
        flag_d[i][4] = src_buf[3][3 + i] - src_buf[3][6 + i];  // tbuf4[3+i] - tbuf4[6+i];
        flag_d[i][5] = src_buf[3][3 + i] - src_buf[4][6 + i];  // tbuf4[3+i] - tbuf5[6+i];
        flag_d[i][6] = src_buf[3][3 + i] - src_buf[5][5 + i];  // tbuf4[3+i] - tbuf6[5+i];
        flag_d[i][7] = src_buf[3][3 + i] - src_buf[6][4 + i];  // tbuf4[3+i] - tbuf7[4+i];
        flag_d[i][8] = src_buf[3][3 + i] - src_buf[6][3 + i];  // tbuf4[3+i] - tbuf7[3+i];
        flag_d[i][9] = src_buf[3][3 + i] - src_buf[6][2 + i];  // tbuf4[3+i] - tbuf7[2+i];
        flag_d[i][10] = src_buf[3][3 + i] - src_buf[5][1 + i]; // tbuf4[3+i] - tbuf6[1+i];
        flag_d[i][11] = src_buf[3][3 + i] - src_buf[4][0 + i]; // tbuf4[3+i] - tbuf5[0+i];
        flag_d[i][12] = src_buf[3][3 + i] - src_buf[3][0 + i]; // tbuf4[3+i] - tbuf4[0+i];
        flag_d[i][13] = src_buf[3][3 + i] - src_buf[2][0 + i]; // tbuf4[3+i] - tbuf3[0+i];
        flag_d[i][14] = src_buf[3][3 + i] - src_buf[1][1 + i]; // tbuf4[3+i] - tbuf2[1+i];
        flag_d[i][15] = src_buf[3][3 + i] - src_buf[0][2 + i]; // tbuf4[3+i] - tbuf1[2+i];
        // Repeating the first 9 values
        flag_d[i][16] = flag_d[i][0];
        flag_d[i][17] = flag_d[i][1];
        flag_d[i][18] = flag_d[i][2];
        flag_d[i][19] = flag_d[i][3];
        flag_d[i][20] = flag_d[i][4];
        flag_d[i][21] = flag_d[i][5];
        flag_d[i][22] = flag_d[i][6];
        flag_d[i][23] = flag_d[i][7];
        flag_d[i][24] = flag_d[i][8];

        // Classification of pixels on the Bresenham's circle into brighter, darker or similar w.r.t.
        // the candidate pixel
        for (ap_uint<4> j = 0; j < 8; j++) {
// clang-format off
            #pragma HLS unroll
            // clang-format on
            if (flag_d[i][j] > _threshold)
                flag_val[i][j] = 1;
            else if (flag_d[i][j] < -_threshold)
                flag_val[i][j] = 2;
            else
                flag_val[i][j] = 0;

            if (flag_d[i][j + 8] > _threshold)
                flag_val[i][j + 8] = 1;
            else if (flag_d[i][j + 8] < -_threshold)
                flag_val[i][j + 8] = 2;
            else
                flag_val[i][j + 8] = 0;
            // Repeating the first 9 values
            flag_val[i][j + PSize] = flag_val[i][j];
        }
        flag_val[i][PSize / 2 + PSize] = flag_val[i][PSize / 2];
        flag_d[i][PSize / 2 + PSize] = flag_d[i][PSize / 2];

        // Bresenham's circle score computation complete

        // Decision making for corners
        uchar_t core = 0;
        uchar_t iscorner = 0;
        uchar_t count = 1;
        for (ap_uint<5> c = 1; c < PSize + PSize / 2 + 1; c++) {
// clang-format off
            #pragma HLS LOOP_TRIPCOUNT MAX=25
            #pragma HLS UNROLL
            // clang-format on
            if ((flag_val[i][c - 1] == flag_val[i][c]) && flag_val[i][c] > 0) {
                count++;
                if (count > PSize / 2) {
                    iscorner = 1; // Candidate pixel is a corner
                }
            } else {
                count = 1;
            }
        } // Corner position computation complete
        // NMS Score Computation
        if (iscorner) {
            xFCoreScore(flag_d[i], _threshold, &core);
            pack_corners.range(ix + 7, ix) = 255;
        } else
            pack_corners.range(ix + 7, ix) = 0;
        ix += 8;
        // Pack the 8-bit score values into 64-bit words
        tbuf_temp.range(kx + 7, kx) = core; // Set bits in a range of positions.
        kx += 8;
    }
    // return tbuf_temp;

    OutputValues[0] = tbuf_temp; // array[(WIN_SZ_SQ)>>1];
    return;
}

template <int SRC_T, int ROWS, int COLS, int DEPTH, int NPC, int WORDWIDTH, int TC, int WIN_SZ, int WIN_SZ_SQ>
void ProcessFast(xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _src_mat,
                 xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _out_mat,
                 XF_SNAME(WORDWIDTH) buf[WIN_SZ][(COLS >> XF_BITSHIFT(NPC))],
                 XF_PTNAME(DEPTH) src_buf[WIN_SZ][XF_NPIXPERCYCLE(NPC) + (WIN_SZ - 1)],
                 XF_PTNAME(DEPTH) OutputValues[XF_NPIXPERCYCLE(NPC)],
                 XF_SNAME(WORDWIDTH) & P0,
                 uint16_t img_width,
                 uint16_t img_height,
                 uint16_t& shift_x,
                 ap_uint<13> row_ind[WIN_SZ],
                 ap_uint<13> row,
                 ap_uint<8> win_size,
                 uchar_t _threshold,
                 XF_PTNAME(DEPTH) & pack_corners,
                 int& read_index,
                 int& write_index) {
// clang-format off
    #pragma HLS INLINE
    // clang-format on

    XF_SNAME(WORDWIDTH) buf_cop[WIN_SZ];
// clang-format off
    #pragma HLS ARRAY_PARTITION variable=buf_cop complete dim=1
    // clang-format on

    uint16_t npc = XF_NPIXPERCYCLE(NPC);
    uint16_t col_loop_var = 0;
    if (npc == 1) {
        col_loop_var = (WIN_SZ >> 1);
    } else {
        col_loop_var = 1;
    }
    for (int extract_px = 0; extract_px < WIN_SZ; extract_px++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=WIN_SZ max=WIN_SZ
        #pragma HLS unroll
        // clang-format on
        for (int ext_copy = 0; ext_copy < npc + WIN_SZ - 1; ext_copy++) {
// clang-format off
            #pragma HLS unroll
            // clang-format on
            src_buf[extract_px][ext_copy] = 0;
        }
    }

Col_Loop:
    for (ap_uint<13> col = 0; col < ((img_width) >> XF_BITSHIFT(NPC)) + col_loop_var; col++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=TC max=TC
        #pragma HLS pipeline
        #pragma HLS LOOP_FLATTEN OFF
        // clang-format on
        if (row < img_height && col < (img_width >> XF_BITSHIFT(NPC)))
            buf[row_ind[win_size - 1]][col] = _src_mat.read(read_index++); // Read data

        if (NPC == XF_NPPC8) {
            for (int copy_buf_var = 0; copy_buf_var < WIN_SZ; copy_buf_var++) {
// clang-format off
                #pragma HLS LOOP_TRIPCOUNT min=WIN_SZ max=WIN_SZ
                #pragma HLS UNROLL
                // clang-format on
                if ((row > (img_height - 1)) && (copy_buf_var > (win_size - 1 - (row - (img_height - 1))))) {
                    buf_cop[copy_buf_var] = buf[(row_ind[win_size - 1 - (row - (img_height - 1))])][col];
                } else {
                    if (col < (img_width >> XF_BITSHIFT(NPC)))
                        buf_cop[copy_buf_var] = buf[(row_ind[copy_buf_var])][col];
                }
            }

            XF_PTNAME(DEPTH) src_buf_temp_copy[WIN_SZ][XF_NPIXPERCYCLE(NPC)];
            XF_PTNAME(DEPTH) src_buf_temp_copy_extract[XF_NPIXPERCYCLE(NPC)];

            for (int extract_px = 0; extract_px < WIN_SZ; extract_px++) {
// clang-format off
                #pragma HLS LOOP_TRIPCOUNT min=WIN_SZ max=WIN_SZ
                #pragma HLS unroll
                // clang-format on
                XF_SNAME(WORDWIDTH) toextract = buf_cop[extract_px];
                xfExtractPixels<NPC, WORDWIDTH, DEPTH>(src_buf_temp_copy_extract, toextract, 0);
                for (int ext_copy = 0; ext_copy < npc; ext_copy++) {
// clang-format off
                    #pragma HLS unroll
                    // clang-format on
                    src_buf_temp_copy[extract_px][ext_copy] = src_buf_temp_copy_extract[ext_copy];
                }
            }
            for (int extract_px = 0; extract_px < WIN_SZ; extract_px++) {
// clang-format off
                #pragma HLS LOOP_TRIPCOUNT min=WIN_SZ max=WIN_SZ
                // clang-format on
                for (int col_warp = 0; col_warp < (WIN_SZ >> 1); col_warp++) {
// clang-format off
                    #pragma HLS UNROLL
                    #pragma HLS LOOP_TRIPCOUNT min=WIN_SZ max=WIN_SZ
                    // clang-format on
                    if (col == img_width >> XF_BITSHIFT(NPC)) {
                        src_buf[extract_px][col_warp + npc + (WIN_SZ >> 1)] =
                            src_buf[extract_px][npc + (WIN_SZ >> 1) - 1];
                    } else {
                        src_buf[extract_px][col_warp + npc + (WIN_SZ >> 1)] = src_buf_temp_copy[extract_px][col_warp];
                    }
                }
            }

            if (col == 0) {
                for (int extract_px = 0; extract_px < WIN_SZ; extract_px++) {
// clang-format off
                    #pragma HLS LOOP_TRIPCOUNT min=WIN_SZ max=WIN_SZ
                    // clang-format on
                    for (int col_warp = 0; col_warp < npc + (WIN_SZ >> 1); col_warp++) {
// clang-format off
                        #pragma HLS UNROLL
                        #pragma HLS LOOP_TRIPCOUNT min=WIN_SZ max=WIN_SZ
                        // clang-format on
                        src_buf[extract_px][col_warp] = src_buf_temp_copy[extract_px][0];
                    }
                }
            }

            XF_PTNAME(DEPTH) src_buf_temp_med_apply[WIN_SZ][XF_NPIXPERCYCLE(NPC) + (WIN_SZ - 1)];

            for (int applyfast = 0; applyfast < npc; applyfast++) {
// clang-format off
                #pragma HLS UNROLL
                // clang-format on
                for (int copyi = 0; copyi < WIN_SZ; copyi++) {
                    for (int copyj = 0; copyj < WIN_SZ; copyj++) {
                        src_buf_temp_med_apply[copyi][copyj] = src_buf[copyi][copyj + applyfast];
                    }
                }

                XF_PTNAME(DEPTH) OutputValues_percycle[1];

                OutputValues_percycle[0] = 0;

                if (row < (img_height) && row >= 6 && (!(col <= 1 && applyfast < 3)) &&
                    (!(col == (((img_width) >> XF_BITSHIFT(NPC))) && applyfast > 4))) // && (!(col==1 && applyfast<=6)))
                {
                    xFfastProc<NPC, WORDWIDTH, DEPTH, WIN_SZ, WIN_SZ_SQ>(OutputValues_percycle, src_buf_temp_med_apply,
                                                                         WIN_SZ, _threshold, pack_corners);
                }

                if (row >= img_height) {
                    OutputValues_percycle[0] = 0;
                }

                OutputValues[applyfast] = OutputValues_percycle[0];
            }

            if (col >= 1) {
                shift_x = 0;
                P0 = 0;
                xfPackPixels<NPC, WORDWIDTH, DEPTH>(OutputValues, P0, 0, npc, shift_x);
                _out_mat.write(write_index++, P0);
            }

            for (int extract_px = 0; extract_px < WIN_SZ; extract_px++) {
// clang-format off
                #pragma HLS LOOP_TRIPCOUNT min=WIN_SZ max=WIN_SZ
                // clang-format on
                for (int col_warp = 0; col_warp < (WIN_SZ >> 1); col_warp++) {
// clang-format off
                    #pragma HLS UNROLL
                    #pragma HLS LOOP_TRIPCOUNT min=WIN_SZ max=WIN_SZ
                    // clang-format on
                    src_buf[extract_px][col_warp] = src_buf[extract_px][col_warp + npc];
                }
            }

            for (int extract_px = 0; extract_px < WIN_SZ; extract_px++) {
// clang-format off
                #pragma HLS LOOP_TRIPCOUNT min=WIN_SZ max=WIN_SZ
                // clang-format on
                for (int col_warp = 0; col_warp < npc; col_warp++) {
// clang-format off
                    #pragma HLS UNROLL
                    #pragma HLS LOOP_TRIPCOUNT min=WIN_SZ max=WIN_SZ
                    // clang-format on
                    src_buf[extract_px][col_warp + (WIN_SZ >> 1)] = src_buf_temp_copy[extract_px][col_warp];
                }
            }

        } else {
            for (int copy_buf_var = 0; copy_buf_var < WIN_SZ; copy_buf_var++) {
// clang-format off
                #pragma HLS LOOP_TRIPCOUNT min=WIN_SZ max=WIN_SZ
                #pragma HLS UNROLL
                // clang-format on
                if ((row > (img_height - 1)) && (copy_buf_var > (win_size - 1 - (row - (img_height - 1))))) {
                    buf_cop[copy_buf_var] = buf[(row_ind[win_size - 1 - (row - (img_height - 1))])][col];
                } else {
                    buf_cop[copy_buf_var] = buf[(row_ind[copy_buf_var])][col];
                }
            }
            for (int extract_px = 0; extract_px < WIN_SZ; extract_px++) {
// clang-format off
                #pragma HLS LOOP_TRIPCOUNT min=WIN_SZ max=WIN_SZ
                #pragma HLS UNROLL
                // clang-format on
                if (col < img_width) {
                    src_buf[extract_px][win_size - 1] = buf_cop[extract_px];
                } else {
                    src_buf[extract_px][win_size - 1] = src_buf[extract_px][win_size - 2];
                }
            }

            if ((col < (img_width) && row < (img_height)) && col >= 6 && row >= 6) {
                xFfastProc<NPC, WORDWIDTH, DEPTH, WIN_SZ, WIN_SZ_SQ>(OutputValues, src_buf, win_size, _threshold,
                                                                     pack_corners);
            }

            if (row >= img_height || col >= img_width) {
                OutputValues[0] = 0;
            }

            if (col >= (WIN_SZ >> 1)) {
                _out_mat.write(write_index++, OutputValues[0]);
            }
            for (int wrap_buf = 0; wrap_buf < WIN_SZ; wrap_buf++) {
// clang-format off
                #pragma HLS UNROLL
                #pragma HLS LOOP_TRIPCOUNT min=WIN_SZ max=WIN_SZ
                // clang-format on
                for (int col_warp = 0; col_warp < WIN_SZ - 1; col_warp++) {
// clang-format off
                    #pragma HLS UNROLL
                    #pragma HLS LOOP_TRIPCOUNT min=WIN_SZ max=WIN_SZ
                    // clang-format on
                    if (col == 0) {
                        src_buf[wrap_buf][col_warp] = src_buf[wrap_buf][win_size - 1];
                    } else {
                        src_buf[wrap_buf][col_warp] = src_buf[wrap_buf][col_warp + 1];
                    }
                }
            }
        }
    } // Col_Loop
}

template <int SRC_T, int ROWS, int COLS, int DEPTH, int NPC, int WORDWIDTH, int TC, int WIN_SZ, int WIN_SZ_SQ>
void xFfast7x7(xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _src_mat,
               xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _out_mat,
               ap_uint<8> win_size,
               uint16_t img_height,
               uint16_t img_width,
               uchar_t _threshold) {
    ap_uint<13> row_ind[WIN_SZ];
// clang-format off
    #pragma HLS ARRAY_PARTITION variable=row_ind complete dim=1
    // clang-format on

    XF_PTNAME(DEPTH) pack_corners;

    uint16_t shift_x = 0;
    ap_uint<13> row, col;
    XF_PTNAME(DEPTH) OutputValues[XF_NPIXPERCYCLE(NPC)];
// clang-format off
    #pragma HLS ARRAY_PARTITION variable=OutputValues complete dim=1
    // clang-format on

    XF_PTNAME(DEPTH) src_buf[WIN_SZ][XF_NPIXPERCYCLE(NPC) + (WIN_SZ - 1)];
// clang-format off
    #pragma HLS ARRAY_PARTITION variable=src_buf complete dim=1
    #pragma HLS ARRAY_PARTITION variable=src_buf complete dim=2
    // clang-format on
    // src_buf1 et al merged
    XF_SNAME(WORDWIDTH) P0;

    XF_SNAME(WORDWIDTH) buf[WIN_SZ][(COLS >> XF_BITSHIFT(NPC))];
// clang-format off
    #pragma HLS ARRAY_PARTITION variable=buf complete dim=1
    #pragma HLS RESOURCE variable=buf core=RAM_S2P_BRAM
    // clang-format on

    // initializing row index

    for (int init_row_ind = 0; init_row_ind < win_size; init_row_ind++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=WIN_SZ max=WIN_SZ
        // clang-format on
        row_ind[init_row_ind] = init_row_ind;
    }

    int read_index = 0;
    int write_index = 0;

read_lines:
    for (int init_buf = row_ind[win_size >> 1]; init_buf < row_ind[win_size - 1]; init_buf++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=WIN_SZ max=WIN_SZ
        // clang-format on
        for (col = 0; col<img_width>> XF_BITSHIFT(NPC); col++) {
// clang-format off
            #pragma HLS LOOP_TRIPCOUNT min=TC max=TC
            #pragma HLS pipeline
            #pragma HLS LOOP_FLATTEN OFF
            // clang-format on
            buf[init_buf][col] = _src_mat.read(read_index++);
        }
    }

    // takes care of top borders
    for (col = 0; col<img_width>> XF_BITSHIFT(NPC); col++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=TC max=TC
        // clang-format on
        for (int init_buf = 0; init_buf<WIN_SZ>> 1; init_buf++) {
// clang-format off
            #pragma HLS LOOP_TRIPCOUNT min=WIN_SZ max=WIN_SZ
            #pragma HLS UNROLL
            // clang-format on
            buf[init_buf][col] = 0; // buf[row_ind[win_size>>1]][col];
        }
    }

Row_Loop:
    for (row = (win_size >> 1); row < img_height + (win_size >> 1); row++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=ROWS max=ROWS
        // clang-format on

        P0 = 0;
        ProcessFast<SRC_T, ROWS, COLS, DEPTH, NPC, WORDWIDTH, TC, WIN_SZ, WIN_SZ_SQ>(
            _src_mat, _out_mat, buf, src_buf, OutputValues, P0, img_width, img_height, shift_x, row_ind, row, win_size,
            _threshold, pack_corners, read_index, write_index);

        // update indices
        ap_uint<13> zero_ind = row_ind[0];
        for (int init_row_ind = 0; init_row_ind < WIN_SZ - 1; init_row_ind++) {
// clang-format off
            #pragma HLS LOOP_TRIPCOUNT min=WIN_SZ max=WIN_SZ
            #pragma HLS UNROLL
            // clang-format on
            row_ind[init_row_ind] = row_ind[init_row_ind + 1];
        }
        row_ind[win_size - 1] = zero_ind;
    } // Row_Loop
}

template <int NPC, int DEPTH, int WIN_SZ, int WIN_SZ_SQ>
void xFnmsProc(XF_PTNAME(DEPTH) OutputValues[XF_NPIXPERCYCLE(NPC)],
               XF_PTNAME(DEPTH) src_buf[WIN_SZ][XF_NPIXPERCYCLE(NPC) + (WIN_SZ - 1)],
               ap_uint<8> win_size) {
// clang-format off
    #pragma HLS INLINE
    // clang-format on

    XF_PTNAME(DEPTH) pix;
    // Comparing scores of the candidate pixel with neighbors in a 3x3 window
    if (src_buf[1][1] != 0) { // if score of candidate pixel != 0

        if ((src_buf[1][1] > src_buf[1][0]) && (src_buf[1][1] > src_buf[1][2]) && (src_buf[1][1] > src_buf[0][0]) &&
            (src_buf[1][1] > src_buf[0][1]) && (src_buf[1][1] > src_buf[0][2]) && (src_buf[1][1] > src_buf[2][0]) &&
            (src_buf[1][1] > src_buf[2][1]) && (src_buf[1][1] > src_buf[2][2])) {
            pix = 255;
        } else {
            pix = 0;
        }
    } else {
        pix = 0;
    }

    OutputValues[0] = pix; // array[(WIN_SZ_SQ)>>1];
    return;
}

template <int SRC_T, int ROWS, int COLS, int DEPTH, int NPC, int WORDWIDTH, int TC, int WIN_SZ, int WIN_SZ_SQ>
void Processfastnms(xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _src_mat,
                    xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _out_mat,
                    XF_SNAME(WORDWIDTH) buf[WIN_SZ][(COLS >> XF_BITSHIFT(NPC))],
                    XF_PTNAME(DEPTH) src_buf[WIN_SZ][XF_NPIXPERCYCLE(NPC) + (WIN_SZ - 1)],
                    XF_PTNAME(DEPTH) OutputValues[XF_NPIXPERCYCLE(NPC)],
                    XF_SNAME(WORDWIDTH) & P0,
                    uint16_t img_width,
                    uint16_t img_height,
                    uint16_t& shift_x,
                    ap_uint<13> row_ind[WIN_SZ],
                    ap_uint<13> row,
                    ap_uint<8> win_size,
                    int& read_index,
                    int& write_index) {
// clang-format off
    #pragma HLS INLINE
    // clang-format on

    XF_SNAME(WORDWIDTH) buf_cop[WIN_SZ];
// clang-format off
    #pragma HLS ARRAY_PARTITION variable=buf_cop complete dim=1
    // clang-format on

    uint16_t npc = XF_NPIXPERCYCLE(NPC);
    uint16_t col_loop_var = 0;
    if (npc == 1) {
        col_loop_var = (WIN_SZ >> 1);
    } else {
        col_loop_var = 1;
    }
    for (int extract_px = 0; extract_px < WIN_SZ; extract_px++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=WIN_SZ max=WIN_SZ
        #pragma HLS unroll
        // clang-format on
        for (int ext_copy = 0; ext_copy < npc + WIN_SZ - 1; ext_copy++) {
// clang-format off
            #pragma HLS unroll
            // clang-format on
            src_buf[extract_px][ext_copy] = 0;
        }
    }

Col_Loop:
    for (ap_uint<13> col = 0; col < ((img_width) >> XF_BITSHIFT(NPC)) + col_loop_var; col++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=TC max=TC
        #pragma HLS pipeline
        #pragma HLS LOOP_FLATTEN OFF
        // clang-format on
        if (row < img_height && col < (img_width >> XF_BITSHIFT(NPC)))
            buf[row_ind[win_size - 1]][col] = _src_mat.read(read_index++); // Read data

        if (NPC == XF_NPPC8) {
            for (int copy_buf_var = 0; copy_buf_var < WIN_SZ; copy_buf_var++) {
// clang-format off
                #pragma HLS LOOP_TRIPCOUNT min=WIN_SZ max=WIN_SZ
                #pragma HLS UNROLL
                // clang-format on
                if ((row > (img_height - 1)) && (copy_buf_var > (win_size - 1 - (row - (img_height - 1))))) {
                    buf_cop[copy_buf_var] = buf[(row_ind[win_size - 1 - (row - (img_height - 1))])][col];
                } else {
                    if (col < (img_width >> XF_BITSHIFT(NPC)))
                        buf_cop[copy_buf_var] = buf[(row_ind[copy_buf_var])][col];
                    //					else
                    //						buf_cop[copy_buf_var] = buf_cop[copy_buf_var];
                }
            }

            XF_PTNAME(DEPTH) src_buf_temp_copy[WIN_SZ][XF_NPIXPERCYCLE(NPC)];
            XF_PTNAME(DEPTH) src_buf_temp_copy_extract[XF_NPIXPERCYCLE(NPC)];

            for (int extract_px = 0; extract_px < WIN_SZ; extract_px++) {
// clang-format off
                #pragma HLS LOOP_TRIPCOUNT min=WIN_SZ max=WIN_SZ
                #pragma HLS unroll
                // clang-format on
                XF_SNAME(WORDWIDTH) toextract = buf_cop[extract_px];
                xfExtractPixels<NPC, WORDWIDTH, DEPTH>(src_buf_temp_copy_extract, toextract, 0);
                for (int ext_copy = 0; ext_copy < npc; ext_copy++) {
// clang-format off
                    #pragma HLS unroll
                    // clang-format on
                    src_buf_temp_copy[extract_px][ext_copy] = src_buf_temp_copy_extract[ext_copy];
                }
            }
            for (int extract_px = 0; extract_px < WIN_SZ; extract_px++) {
// clang-format off
                #pragma HLS LOOP_TRIPCOUNT min=WIN_SZ max=WIN_SZ
                // clang-format on
                for (int col_warp = 0; col_warp < (WIN_SZ >> 1); col_warp++) {
// clang-format off
                    #pragma HLS UNROLL
                    #pragma HLS LOOP_TRIPCOUNT min=WIN_SZ max=WIN_SZ
                    // clang-format on
                    if (col == img_width >> XF_BITSHIFT(NPC)) {
                        src_buf[extract_px][col_warp + npc + (WIN_SZ >> 1)] =
                            src_buf[extract_px][npc + (WIN_SZ >> 1) - 1];
                    } else {
                        src_buf[extract_px][col_warp + npc + (WIN_SZ >> 1)] = src_buf_temp_copy[extract_px][col_warp];
                    }
                }
            }

            if (col == 0) {
                for (int extract_px = 0; extract_px < WIN_SZ; extract_px++) {
// clang-format off
                    #pragma HLS LOOP_TRIPCOUNT min=WIN_SZ max=WIN_SZ
                    // clang-format on
                    for (int col_warp = 0; col_warp < npc + (WIN_SZ >> 1); col_warp++) {
// clang-format off
                        #pragma HLS UNROLL
                        #pragma HLS LOOP_TRIPCOUNT min=WIN_SZ max=WIN_SZ
                        // clang-format on
                        src_buf[extract_px][col_warp] = src_buf_temp_copy[extract_px][0];
                    }
                }
            }

            XF_PTNAME(DEPTH) src_buf_temp_med_apply[WIN_SZ][XF_NPIXPERCYCLE(NPC) + (WIN_SZ - 1)];
            for (int applyfast = 0; applyfast < npc; applyfast++) {
// clang-format off
                #pragma HLS UNROLL
                // clang-format on
                for (int copyi = 0; copyi < WIN_SZ; copyi++) {
                    for (int copyj = 0; copyj < WIN_SZ; copyj++) {
                        src_buf_temp_med_apply[copyi][copyj] = src_buf[copyi][copyj + applyfast];
                    }
                }
                XF_PTNAME(DEPTH) OutputValues_percycle[XF_NPIXPERCYCLE(NPC)];
                xFnmsProc<NPC, DEPTH, WIN_SZ, WIN_SZ_SQ>(OutputValues_percycle, src_buf_temp_med_apply, WIN_SZ);
                OutputValues[applyfast] = OutputValues_percycle[0];
            }
            if (col >= 1) {
                shift_x = 0;
                P0 = 0;
                xfPackPixels<NPC, WORDWIDTH, DEPTH>(OutputValues, P0, 0, npc, shift_x);
                _out_mat.write(write_index++, P0);
            }

            for (int extract_px = 0; extract_px < WIN_SZ; extract_px++) {
// clang-format off
                #pragma HLS LOOP_TRIPCOUNT min=WIN_SZ max=WIN_SZ
                // clang-format on
                for (int col_warp = 0; col_warp < (WIN_SZ >> 1); col_warp++) {
// clang-format off
                    #pragma HLS UNROLL
                    #pragma HLS LOOP_TRIPCOUNT min=WIN_SZ max=WIN_SZ
                    // clang-format on
                    src_buf[extract_px][col_warp] = src_buf[extract_px][col_warp + npc];
                }
            }

            for (int extract_px = 0; extract_px < WIN_SZ; extract_px++) {
// clang-format off
                #pragma HLS LOOP_TRIPCOUNT min=WIN_SZ max=WIN_SZ
                // clang-format on
                for (int col_warp = 0; col_warp < npc; col_warp++) {
// clang-format off
                    #pragma HLS UNROLL
                    #pragma HLS LOOP_TRIPCOUNT min=WIN_SZ max=WIN_SZ
                    // clang-format on
                    src_buf[extract_px][col_warp + (WIN_SZ >> 1)] = src_buf_temp_copy[extract_px][col_warp];
                }
            }

        } else {
            for (int copy_buf_var = 0; copy_buf_var < WIN_SZ; copy_buf_var++) {
// clang-format off
                #pragma HLS LOOP_TRIPCOUNT min=WIN_SZ max=WIN_SZ
                #pragma HLS UNROLL
                // clang-format on
                if ((row > (img_height - 1)) && (copy_buf_var > (win_size - 1 - (row - (img_height - 1))))) {
                    buf_cop[copy_buf_var] = buf[(row_ind[win_size - 1 - (row - (img_height - 1))])][col];
                } else {
                    buf_cop[copy_buf_var] = buf[(row_ind[copy_buf_var])][col];
                }
            }
            for (int extract_px = 0; extract_px < WIN_SZ; extract_px++) {
// clang-format off
                #pragma HLS LOOP_TRIPCOUNT min=WIN_SZ max=WIN_SZ
                #pragma HLS UNROLL
                // clang-format on
                if (col < img_width) {
                    src_buf[extract_px][win_size - 1] = buf_cop[extract_px];
                } else {
                    src_buf[extract_px][win_size - 1] = src_buf[extract_px][win_size - 2];
                }
            }

            xFnmsProc<NPC, DEPTH, WIN_SZ, WIN_SZ_SQ>(OutputValues, src_buf, win_size);

            if (col >= (WIN_SZ >> 1)) {
                _out_mat.write(write_index++, OutputValues[0]);
            }
            for (int wrap_buf = 0; wrap_buf < WIN_SZ; wrap_buf++) {
// clang-format off
                #pragma HLS UNROLL
                #pragma HLS LOOP_TRIPCOUNT min=WIN_SZ max=WIN_SZ
                // clang-format on
                for (int col_warp = 0; col_warp < WIN_SZ - 1; col_warp++) {
// clang-format off
                    #pragma HLS UNROLL
                    #pragma HLS LOOP_TRIPCOUNT min=WIN_SZ max=WIN_SZ
                    // clang-format on
                    if (col == 0) {
                        src_buf[wrap_buf][col_warp] = src_buf[wrap_buf][win_size - 1];
                    } else {
                        src_buf[wrap_buf][col_warp] = src_buf[wrap_buf][col_warp + 1];
                    }
                }
            }
        }
    } // Col_Loop
}

template <int SRC_T, int ROWS, int COLS, int DEPTH, int NPC, int WORDWIDTH, int TC, int WIN_SZ, int WIN_SZ_SQ>
void xFfastnms(xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _src_mat,
               xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _out_mat,
               ap_uint<8> win_size,
               uint16_t img_height,
               uint16_t img_width) {
    ap_uint<13> row_ind[WIN_SZ];
// clang-format off
    #pragma HLS ARRAY_PARTITION variable=row_ind complete dim=1
    // clang-format on

    uint16_t shift_x = 0;
    ap_uint<13> row, col;
    XF_PTNAME(DEPTH) OutputValues[XF_NPIXPERCYCLE(NPC)];
// clang-format off
    #pragma HLS ARRAY_PARTITION variable=OutputValues complete dim=1
    // clang-format on

    XF_PTNAME(DEPTH) src_buf[WIN_SZ][XF_NPIXPERCYCLE(NPC) + (WIN_SZ - 1)];
// clang-format off
    #pragma HLS ARRAY_PARTITION variable=src_buf complete dim=1
    #pragma HLS ARRAY_PARTITION variable=src_buf complete dim=2
    // clang-format on
    // src_buf1 et al merged
    XF_SNAME(WORDWIDTH) P0;

    XF_SNAME(WORDWIDTH) buf[WIN_SZ][(COLS >> XF_BITSHIFT(NPC))];
// clang-format off
    #pragma HLS ARRAY_PARTITION variable=buf complete dim=1
    #pragma HLS RESOURCE variable=buf core=RAM_S2P_BRAM
    // clang-format on

    // initializing row index

    for (int init_row_ind = 0; init_row_ind < win_size; init_row_ind++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=WIN_SZ max=WIN_SZ
        // clang-format on
        row_ind[init_row_ind] = init_row_ind;
    }

    int readind_val = 0, writeind_val = 0;

read_lines:
    for (int init_buf = row_ind[win_size >> 1]; init_buf < row_ind[win_size - 1]; init_buf++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=WIN_SZ max=WIN_SZ
        // clang-format on
        for (col = 0; col<img_width>> XF_BITSHIFT(NPC); col++) {
// clang-format off
            #pragma HLS LOOP_TRIPCOUNT min=TC max=TC
            #pragma HLS pipeline
            #pragma HLS LOOP_FLATTEN OFF
            // clang-format on
            buf[init_buf][col] = _src_mat.read(readind_val++);
        }
    }

    // takes care of top borders
    for (col = 0; col<img_width>> XF_BITSHIFT(NPC); col++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=TC max=TC
        // clang-format on
        for (int init_buf = 0; init_buf<WIN_SZ>> 1; init_buf++) {
// clang-format off
            #pragma HLS LOOP_TRIPCOUNT min=WIN_SZ max=WIN_SZ
            #pragma HLS UNROLL
            // clang-format on
            buf[init_buf][col] = buf[row_ind[win_size >> 1]][col];
        }
    }

Row_Loop:
    for (row = (win_size >> 1); row < img_height + (win_size >> 1); row++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=ROWS max=ROWS
        // clang-format on

        P0 = 0;
        Processfastnms<SRC_T, ROWS, COLS, DEPTH, NPC, WORDWIDTH, TC, WIN_SZ, WIN_SZ_SQ>(
            _src_mat, _out_mat, buf, src_buf, OutputValues, P0, img_width, img_height, shift_x, row_ind, row, win_size,
            readind_val, writeind_val);

        // update indices
        ap_uint<13> zero_ind = row_ind[0];
        for (int init_row_ind = 0; init_row_ind < WIN_SZ - 1; init_row_ind++) {
// clang-format off
            #pragma HLS LOOP_TRIPCOUNT min=WIN_SZ max=WIN_SZ
            #pragma HLS UNROLL
            // clang-format on
            row_ind[init_row_ind] = row_ind[init_row_ind + 1];
        }
        row_ind[win_size - 1] = zero_ind;
    } // Row_Loop
}

template <int SRC_T, int ROWS, int COLS, int DEPTH, int NPC, int WORDWIDTH_SRC, int WORDWIDTH_DST, int NMSVAL>
void xFFastCornerDetection(xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _src_mat,
                           xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _dst_mat,
                           unsigned short _image_height,
                           unsigned short _image_width,
                           uchar_t _threshold) {
#ifndef __SYNTHESIS__
    assert(((DEPTH == XF_8UP)) &&
           "Invalid Depth. The function xFFast "
           "is valid only for the Depths AU_8U");

    assert(((NMSVAL == 0) || (NMSVAL == 1)) && "Invalid Value. The NMS value should be either 0 or 1");

    assert(((_image_height <= ROWS) && (_image_width <= COLS)) && "ROWS and COLS should be greater than input image");
#endif

    xf::cv::Mat<SRC_T, ROWS, COLS, NPC> _dst(_image_height, _image_width);
// clang-format off
        #pragma HLS DATAFLOW
// clang-format on
#pragma HLS stream variable = _dst.data dim = 1 depth = 2

    if (NMSVAL == 1) {
        xFfast7x7<SRC_T, ROWS, COLS, DEPTH, NPC, WORDWIDTH_SRC, (COLS >> XF_BITSHIFT(NPC)) + (7 >> 1), 7, 7 * 7>(
            _src_mat, _dst, 7, _image_height, _image_width, _threshold);
        xFfastnms<SRC_T, ROWS, COLS, DEPTH, NPC, WORDWIDTH_SRC, (COLS >> XF_BITSHIFT(NPC)) + (3 >> 1), 3, 3 * 3>(
            _dst, _dst_mat, 3, _image_height, _image_width);
    } else if (NMSVAL == 0) {
        xFfast7x7<SRC_T, ROWS, COLS, DEPTH, NPC, WORDWIDTH_SRC, (COLS >> XF_BITSHIFT(NPC)) + (7 >> 1), 7, 7 * 7>(
            _src_mat, _dst_mat, 7, _image_height, _image_width, _threshold);
    }
}

template <int NMS, int SRC_T, int ROWS, int COLS, int NPC = 1>
void fast(xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _src_mat,
          xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _dst_mat,
          unsigned char _threshold) {
// clang-format off
    #pragma HLS inline off
    // clang-format on

    // clang-format off
    // clang-format on

    xFFastCornerDetection<SRC_T, ROWS, COLS, XF_DEPTH(SRC_T, NPC), NPC, XF_WORDWIDTH(SRC_T, NPC), XF_32UW, NMS>(
        _src_mat, _dst_mat, _src_mat.rows, _src_mat.cols, _threshold);
}
} // namespace cv
} // namespace xf
#endif //_XF_FAST_HPP_
