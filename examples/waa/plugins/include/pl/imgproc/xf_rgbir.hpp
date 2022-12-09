/*
 * Copyright 2021 Xilinx, Inc.
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

#ifndef _XF_RGBIR_HPP_
#define _XF_RGBIR_HPP_

#include <limits>
#include "common/xf_video_mem.hpp"
#include "common/xf_utility.hpp"
#include "imgproc/xf_rgbir_bilinear.hpp"
#include "imgproc/xf_duplicateimage.hpp"

namespace xf {
namespace cv {

// Internal constants
#define _NPPC (XF_NPIXPERCYCLE(NPPC))       // Number of pixel per clock to be processed
#define _NPPC_SHIFT_VAL (XF_BITSHIFT(NPPC)) // Gives log base 2 on NPPC; Used for shifting purpose in case of division
#define _ECPR ((((K_COLS >> 1) + (_NPPC - 1)) / _NPPC)) // Extra clocks required for processing a row
#define _NP_IN_PREV \
    (_NPPC - ((K_COLS >> 1) - (((K_COLS >> 1) / _NPPC) * _NPPC))) // No.of valid destination pixels in previous clock
#define _DST_PIX_WIDTH (XF_PIXELDEPTH(XF_DEPTH(DST_T, NPPC)))     // destination pixel width

// ======================================================================================
// A generic structure for filter operation
// --------------------------------------------------------------------------------------
// Template Args:-
//        SRC_T : Data type of soruce image element
//         ROWS : Image height
//         COLS : Image width
//         NPPC : No.of pixels per clock
//     BORDER_T : Type of boder to be used for edge pixel(s) computation
// ......................................................................................

template <typename F,
          int SRC_T,
          int DST_T,
          int XFCV_DEPTH,
          int ROWS,
          int COLS,
          int K_ROWS,
          int K_COLS,
          int NPPC = 1,
          int BORDER_T = XF_BORDER_CONSTANT,
          int USE_URAM = 0,
          int USE_MAT = 1,
          int BFORMAT = 0>
class GenericFilter {
   public:
    // Internal regsiters/buffers
    xf::cv::Window<K_ROWS, XF_NPIXPERCYCLE(NPPC) + (K_COLS - 1), XF_DTUNAME(SRC_T, NPPC)>
        src_blk; // Kernel sized image block with pixel parallelism
    xf::cv::LineBuffer<K_ROWS - 1,
                       (COLS >> _NPPC_SHIFT_VAL),
                       XF_TNAME(SRC_T, NPPC),
                       (USE_URAM ? RAM_S2P_URAM : RAM_S2P_BRAM),
                       (USE_URAM ? K_ROWS - 1 : 1)>
        buff; // Line Buffer for K_ROWS from the image

    // Internal Registers
    unsigned short num_clks_per_row; // No.of clocks required for processing one row
    unsigned int rd_ptr;             // Read pointer
    unsigned int wr_ptr;             // Write pointer-rggb
    unsigned int wr_ptr_ir;          // Write pointer-ir

    // Default Constructor
    GenericFilter() {
        num_clks_per_row = 0;
        rd_ptr = 0;
        wr_ptr = 0;
        wr_ptr_ir = 0;
    }

    // Internal functions
    void initialize(xf::cv::Mat<SRC_T, ROWS, COLS, NPPC>& _src);

    void process_row(unsigned short readRow,
                     unsigned short row,
                     xf::cv::Mat<SRC_T, ROWS, COLS, NPPC>& _src,
                     ap_int<4> R_IR_C1_wgts[5][5],
                     ap_int<4> R_IR_C2_wgts[5][5],
                     ap_int<4> B_at_R_wgts[5][5],
                     ap_int<4> IR_at_R_wgts[3][3],
                     ap_int<4> IR_at_B_wgts[3][3],
                     xf::cv::Mat<DST_T, ROWS, COLS, NPPC, XFCV_DEPTH>& _dst_rggb,
                     xf::cv::Mat<DST_T, ROWS, COLS, NPPC>& _dst_half_ir);
    void process_image(xf::cv::Mat<SRC_T, ROWS, COLS, NPPC>& _src,
                       ap_int<4> R_IR_C1_wgts[5][5],
                       ap_int<4> R_IR_C2_wgts[5][5],
                       ap_int<4> B_at_R_wgts[5][5],
                       ap_int<4> IR_at_R_wgts[3][3],
                       ap_int<4> IR_at_B_wgts[3][3],
                       xf::cv::Mat<DST_T, ROWS, COLS, NPPC, XFCV_DEPTH>& _dst_rggb,
                       xf::cv::Mat<DST_T, ROWS, COLS, NPPC>& _dst_half_ir);
};

// -----------------------------------------------------------------------------------
// Function to initialize internal regsiters and buffers
// -----------------------------------------------------------------------------------
template <typename RGBIR,
          int SRC_T,
          int DST_T,
          int XFCV_DEPTH,
          int ROWS,
          int COLS,
          int K_ROWS,
          int K_COLS,
          int NPPC,
          int BORDER_T,
          int USE_URAM,
          int USE_MAT,
          int BFORMAT>
void GenericFilter<RGBIR,
                   SRC_T,
                   DST_T,
                   XFCV_DEPTH,
                   ROWS,
                   COLS,
                   K_ROWS,
                   K_COLS,
                   NPPC,
                   BORDER_T,
                   USE_URAM,
                   USE_MAT,
                   BFORMAT>::initialize(xf::cv::Mat<SRC_T, ROWS, COLS, NPPC>& _src) {
#pragma HLS INLINE

    // Computing no.of clocks required for processing a row of given image dimensions
    num_clks_per_row = (_src.cols + _NPPC - 1) >> _NPPC_SHIFT_VAL;

    // Read/Write pointer set to start location of input image
    rd_ptr = 0;
    wr_ptr = 0;
    wr_ptr_ir = 0;

    return;
} // End of initialize()

// -----------------------------------------------------------------------------------
// Function to process a row
// -----------------------------------------------------------------------------------
template <typename RGBIR,
          int SRC_T,
          int DST_T,
          int XFCV_DEPTH,
          int ROWS,
          int COLS,
          int K_ROWS,
          int K_COLS,
          int NPPC,
          int BORDER_T,
          int USE_URAM,
          int USE_MAT,
          int BFORMAT>
void GenericFilter<RGBIR,
                   SRC_T,
                   DST_T,
                   XFCV_DEPTH,
                   ROWS,
                   COLS,
                   K_ROWS,
                   K_COLS,
                   NPPC,
                   BORDER_T,
                   USE_URAM,
                   USE_MAT,
                   BFORMAT>::process_row(unsigned short readRow,
                                         unsigned short row,
                                         xf::cv::Mat<SRC_T, ROWS, COLS, NPPC>& _src,
                                         ap_int<4> R_IR_C1_wgts[5][5],
                                         ap_int<4> R_IR_C2_wgts[5][5],
                                         ap_int<4> B_at_R_wgts[5][5],
                                         ap_int<4> IR_at_R_wgts[3][3],
                                         ap_int<4> IR_at_B_wgts[3][3],
                                         xf::cv::Mat<DST_T, ROWS, COLS, NPPC, XFCV_DEPTH>& _dst_rggb,
                                         xf::cv::Mat<DST_T, ROWS, COLS, NPPC>& _dst_half_ir) {
#pragma HLS INLINE OFF

    // --------------------------------------
    // Constants
    // --------------------------------------
    const uint32_t _TC = (COLS >> _NPPC_SHIFT_VAL) + (K_COLS >> 1); // MAX Trip Count per row

    // --------------------------------------
    // Internal variables
    // --------------------------------------
    // Loop count variable
    unsigned short col_loop_cnt = num_clks_per_row + _ECPR;
    unsigned short col_wo_npc = 0, original_col = 0;
    bool toggle_weights = 0, toggle_weights_ir = 0;
    bool flag = 0;
    ap_uint<10> count = 0;
    unsigned short candidateRow = row;
    unsigned short candidateCol = col_wo_npc;
    RGBIR oper;

    // To store out pixels in packed format
    XF_TNAME(DST_T, NPPC) out_pixels, out_pixels_ir;
    XF_TNAME(SRC_T, NPPC) in_pixel;

// --------------------------------------
// Initialize source block buffer to all zeros
// --------------------------------------
SRC_INIT_LOOP:
    for (unsigned short kr = 0; kr < K_ROWS; kr++) {
#pragma HLS UNROLL

        for (unsigned short kc = 0; kc < (_NPPC + K_COLS - 1); kc++) {
#pragma HLS UNROLL
            src_blk.val[kr][kc] = 0;
        }
    }

// --------------------------------------
// Process columns of the row
// --------------------------------------
COL_LOOP:
    for (unsigned short c = 0; c < col_loop_cnt; c++) {
// clang-format off
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=COLS/NPPC
#pragma HLS DEPENDENCE variable=buff.val inter false
        // clang-format on

        // Fetch next pixel of current row
        // .........................................................
        in_pixel = ((readRow < _src.rows) && (c < num_clks_per_row)) ? _src.read(rd_ptr++) : (XF_TNAME(SRC_T, NPPC))0;

    // Fetch data from RAMs and store in 'src_blk' for processing
    // .........................................................
    BUFF_RD_LOOP:
        for (unsigned short kr = 0; kr < K_ROWS; kr++) {
#pragma HLS UNROLL
            //#pragma HLS DEPENDENCE variable=buff.val inter false

            XF_TNAME(SRC_T, NPPC) tmp_rd_buff;

            // Read packed data
            tmp_rd_buff =
                (kr == (K_ROWS - 1)) ? in_pixel : (c < num_clks_per_row) ? buff.val[kr][c] : (XF_TNAME(SRC_T, NPPC))0;

            // Extract pixels from packed data and store in 'src_blk'
            xfExtractPixels<NPPC, XF_WORDWIDTH(SRC_T, NPPC), XF_DEPTH(SRC_T, NPPC)>(src_blk.val[kr], tmp_rd_buff,
                                                                                    (K_COLS - 1));
        }

    // Process the kernel block
    // ........................
    PROCESS_BLK_LOOP:
        for (int pix_idx = 0; pix_idx < _NPPC; pix_idx++) {
#pragma HLS UNROLL

            XF_DTUNAME(DST_T, NPPC) out_pix;
            XF_DTUNAME(DST_T, NPPC) out_pix_ir;

            XF_DTUNAME(SRC_T, NPPC) NxM_src_blk[K_ROWS][K_COLS];
            XF_DTUNAME(SRC_T, NPPC) IR_3x3_src_blk[3][3]; // 3x3 block for IR filter
                                                          // clang-format off
#pragma HLS ARRAY_PARTITION variable=NxM_src_blk complete dim=1
#pragma HLS ARRAY_PARTITION variable=NxM_src_blk complete dim=2
        // clang-format on

        // Extract _NPPC, NxM-blocks from 'src_blk'
        REARRANGE_LOOP:
            for (unsigned short kr = 0; kr < K_ROWS; kr++) {
                //#pragma HLS UNROLL
                for (unsigned short kc = 0; kc < K_COLS; kc++) {
#pragma HLS UNROLL
                    NxM_src_blk[kr][kc] = src_blk.val[kr][pix_idx + kc];
                    if ((kr > 0 && kr < 4) &&
                        (kc > 0 && kc < 4)) { // Filling 3x3 block for IR calculation from 5x5 block
                        IR_3x3_src_blk[kr - 1][kc - 1] = NxM_src_blk[kr][kc];
                    }
                }
            }

            original_col = c * _NPPC + pix_idx;
            col_wo_npc = c * _NPPC - 2 + pix_idx;
            candidateCol = col_wo_npc;

            if (BFORMAT == XF_BAYER_GR) {
                candidateRow = row + 1;
                candidateCol = c * _NPPC + pix_idx;
                //				candidateCol = col_wo_npc + 2;
            }

            if (c >= _ECPR) {
                out_pix = NxM_src_blk[K_ROWS / 2][K_COLS / 2];
                out_pix_ir = NxM_src_blk[K_ROWS / 2][K_COLS / 2];

                if (((((candidateRow - 2) % 4) == 0) && ((candidateCol % 4) == 0)) ||
                    (((candidateRow % 4) == 0) && (((candidateCol - 2) % 4) == 0))) {
                    oper.apply_filter(NxM_src_blk, B_at_R_wgts, out_pix); // B at R
                } else if ((candidateRow & 0x0001) ==
                           1) { // BG Mode - This is odd row, IR location. Compute R here with 5x5 filter

                    if (((candidateCol - 1) % 4) == 0) {
                        oper.apply_filter(NxM_src_blk, R_IR_C1_wgts,
                                          out_pix); // B at IR - Constellation-1 (Red on the top left)
                    } else if (((candidateCol + 1) % 4) == 0) {
                        oper.apply_filter(NxM_src_blk, R_IR_C2_wgts,
                                          out_pix); // B at IR - Constellation-2 (Blue on the top left)
                    }
                }
                if ((((candidateRow % 4) == 0) &&
                     ((candidateCol % 4) == 0)) || // BG Mode - B location, apply 3x3 IR filter
                    ((((candidateRow - 2) % 4) == 0) && (((candidateCol - 2) % 4) == 0))) {
                    oper.template apply_filter<3, 3>(IR_3x3_src_blk, IR_at_B_wgts, out_pix_ir); // IR at B location
                } else if (((((candidateRow - 2) % 4) == 0) && ((candidateCol % 4) == 0)) ||
                           (((candidateRow % 4) == 0) &&
                            (((candidateCol - 2) % 4) == 0))) { // BG Mode - R location, apply 3x3 IR filter

                    oper.template apply_filter<3, 3>(IR_3x3_src_blk, IR_at_R_wgts, out_pix_ir); // IR at R location
                }

                // Start packing the out pixel value every clock of NPPC
                out_pixels.range(((pix_idx + 1) * _DST_PIX_WIDTH) - 1, (pix_idx * _DST_PIX_WIDTH)) = out_pix;
                out_pixels_ir.range(((pix_idx + 1) * _DST_PIX_WIDTH) - 1, (pix_idx * _DST_PIX_WIDTH)) = out_pix_ir;
            }
        }

        // Write the data out to DDR
        // .........................
        if (c >= _ECPR) {
            _dst_rggb.write(wr_ptr++, out_pixels);

            _dst_half_ir.write(wr_ptr_ir++, out_pixels_ir);
        }

        // Move the data in Line Buffers
        // ...........................................
        if (c < num_clks_per_row) {
        BUFF_WR_LOOP:
            for (unsigned short kr = 0; kr < K_ROWS - 1; kr++) {
#pragma HLS UNROLL
                buff.val[kr][c] = src_blk.val[kr + 1][K_COLS - 1];
            }
        }

    // Now get ready for next cycle of computation. So copy the last K_COLS-1 data to start location of 'src_blk'
    // ...........................................
    SHIFT_LOOP:
        for (unsigned short kr = 0; kr < K_ROWS; kr++) {
#pragma HLS UNROLL
            for (unsigned short kc = 0; kc < K_COLS - 1; kc++) {
#pragma HLS UNROLL
                src_blk.val[kr][kc] = src_blk.val[kr][_NPPC + kc];
            }
        }
    }

    return;
} // End of process_row_gauss()

// -----------------------------------------------------------------------------------
// Main function that runs the filter over the image
// -----------------------------------------------------------------------------------
template <typename RGBIR,
          int SRC_T,
          int DST_T,
          int XFCV_DEPTH,
          int ROWS,
          int COLS,
          int K_ROWS,
          int K_COLS,
          int NPPC,
          int BORDER_T,
          int USE_URAM,
          int USE_MAT,
          int BFORMAT>
void GenericFilter<RGBIR,
                   SRC_T,
                   DST_T,
                   XFCV_DEPTH,
                   ROWS,
                   COLS,
                   K_ROWS,
                   K_COLS,
                   NPPC,
                   BORDER_T,
                   USE_URAM,
                   USE_MAT,
                   BFORMAT>::process_image(xf::cv::Mat<SRC_T, ROWS, COLS, NPPC>& _src,
                                           ap_int<4> R_IR_C1_wgts[5][5],
                                           ap_int<4> R_IR_C2_wgts[5][5],
                                           ap_int<4> B_at_R_wgts[5][5],
                                           ap_int<4> IR_at_R_wgts[3][3],
                                           ap_int<4> IR_at_B_wgts[3][3],
                                           xf::cv::Mat<DST_T, ROWS, COLS, NPPC, XFCV_DEPTH>& _dst_rggb,
                                           xf::cv::Mat<DST_T, ROWS, COLS, NPPC>& _dst_half_ir) {
#pragma HLS INLINE OFF
    // Constant declaration
    const uint32_t _TC =
        ((COLS >> _NPPC_SHIFT_VAL) + (K_COLS >> 1)) / NPPC; // MAX Trip Count per row considering N-Pixel parallelsim

    // ----------------------------------
    // Start process with initialization
    // ----------------------------------
    initialize(_src);

// ----------------------------------
// Initialize Line Buffer
// ----------------------------------
// Part1: Initialize the buffer with 1st (kernel height)/2 rows of image
//        Start filling rows from (kernel height)/2 and rest depending on border type
READ_LINES_INIT:
    for (unsigned short r = (K_ROWS >> 1); r < (K_ROWS - 1); r++) { // Note: Ignoring last row
#pragma HLS UNROLL
        for (unsigned short c = 0; c < num_clks_per_row; c++) {
// clang-format off
#pragma HLS PIPELINE
#pragma HLS LOOP_TRIPCOUNT min=1 max=K_ROWS
            // clang-format on
            buff.val[r][c] = _src.read(rd_ptr++); // Reading the rows of image
        }
    }
// Part2: Take care of borders depending on border type.
//        In border replicate mode, fill with 1st row of the image.
BORDER_INIT:
    for (unsigned short r = 0; r < (K_ROWS >> 1); r++) {
#pragma HLS UNROLL
        for (unsigned short c = 0; c < num_clks_per_row; c++) {
#pragma HLS PIPELINE
// clang-format off
#pragma HLS LOOP_TRIPCOUNT min=1 max=K_ROWS/2
            // clang-format on
            buff.val[r][c] = (BORDER_T == XF_BORDER_REPLICATE) ? buff.val[K_ROWS >> 1][c] : (XF_TNAME(SRC_T, NPPC))0;
        }
    }

    short int row = 0;
    short int rLoopEnd = _src.rows + (K_ROWS >> 1);
// ----------------------------------
// Processing each row of the image
// ----------------------------------
ROW_LOOP:
    for (unsigned short r = (K_ROWS >> 1); r < rLoopEnd; r++) {
// clang-format off
#pragma HLS LOOP_TRIPCOUNT min=1 max=ROWS
        // clang-format on
        process_row(r, row, _src, R_IR_C1_wgts, R_IR_C2_wgts, B_at_R_wgts, IR_at_R_wgts, IR_at_B_wgts, _dst_rggb,
                    _dst_half_ir);
        row++;
    }

    return;
} // End of process_image()

// ======================================================================================

// ======================================================================================
// Class for applying the specific filters
// ======================================================================================

template <int SRC_T, int K_ROWS, int K_COLS, int NPPC>
class RGBIR {
   public:
    const int NUM_CH = XF_CHANNELS(SRC_T, NPPC);
    static constexpr int KR_MOD = ((K_ROWS + 1) / 2);
    static constexpr int BIT_DEPTH = XF_DTPIXELDEPTH(SRC_T, NPPC);

    // -------------------------------------------------------------------------
    // Creating apply function (applying filter)
    // -------------------------------------------------------------------------
    // void apply_filter(XF_DTUNAME(SRC_T, NPPC) patch[F_ROWS][F_COLS], XF_DTUNAME(SRC_T, NPPC) wgt[F_ROWS][F_COLS],
    // XF_DTUNAME(SRC_T, NPPC) &pix) {
    template <int F_ROWS = K_ROWS, int F_COLS = K_COLS>
    void apply_filter(XF_DTUNAME(SRC_T, NPPC) patch[F_ROWS][F_COLS],
                      ap_int<4> wgt[F_ROWS][F_COLS],
                      XF_DTUNAME(SRC_T, NPPC) & pix) {
#pragma HLS INLINE OFF

        // Weights used for applying the filter are the values by which the pixel values are to be shifted
        int partial_sum[F_ROWS] = {0};
        int sum = 0;

        XF_DTUNAME(SRC_T, NPPC) tempVal = 0, NtempVal = 0;

    apply_row:
        for (int fr = 0; fr < F_ROWS; fr++) {
// clang-format off
        #pragma HLS PIPELINE II=1
        // clang-format on
        apply_col:
            for (int fc = 0; fc < F_COLS; fc++) {
                //				#pragma HLS UNROLL
                if (wgt[fr][fc] > 0) {
                    if (wgt[fr][fc] == 7) { // wgt is -1
                        partial_sum[fr] -= patch[fr][fc];
                    } else if (wgt[fr][fc] == 6) { // wgt is 0
                        partial_sum[fr] += 0;
                    } else {
                        partial_sum[fr] += patch[fr][fc] >> (__ABS((char)wgt[fr][fc]));
                    }
                } else if (wgt[fr][fc] < 0) {
                    partial_sum[fr] -= patch[fr][fc] >> (__ABS((char)wgt[fr][fc]));
                }
            }
        }

    apply_sum:
        for (int fsum = 0; fsum < F_ROWS; fsum++) {
#pragma HLS UNROLL
            sum += partial_sum[fsum];
        }

        pix = xf::cv::xf_satcast<BIT_DEPTH>(sum);

        return;
    }
};

// ======================================================================================
// Top level RGB-IR Demosaic API
// --------------------------------------------------------------------------------------
// Template Args:-
//         TYPE : Data type of source image
//         ROWS : Image height
//         COLS : Image width
//         NPPC : No.of pixels per clock
//     BORDER_T : Type of border to be used for edge pixel(s) computation
//                (XF_BORDER_REPLICATE, XF_BORDER_CONSTANT, XF_BORDER_REFLECT_101, XF_BORDER_REFLECT)
// ......................................................................................
#define _RGB_IR_ RGB_IR<TYPE, K_SIZE, K_SIZE, NPPC>

// --------------------------------------------------------------------------------------
// Function to perform demosaicing on RGB-IR image
// --------------------------------------------------------------------------------------
template <int FSIZE1 = 5,
          int FSIZE2 = 3,
          int BFORMAT = 0,
          int TYPE,
          int ROWS,
          int COLS,
          int NPPC = 1,
          int XFCV_DEPTH,
          int BORDER_T = XF_BORDER_CONSTANT,
          int USE_URAM = 0>
void RGBIR_Demosaic(xf::cv::Mat<TYPE, ROWS, COLS, NPPC>& _src,
                    char R_IR_C1_wgts[FSIZE1 * FSIZE1],
                    char R_IR_C2_wgts[FSIZE1 * FSIZE1],
                    char B_at_R_wgts[FSIZE1 * FSIZE1],
                    char IR_at_R_wgts[FSIZE2 * FSIZE2],
                    char IR_at_B_wgts[FSIZE2 * FSIZE2],
                    xf::cv::Mat<TYPE, ROWS, COLS, NPPC, XFCV_DEPTH>& _dst_rggb,
                    xf::cv::Mat<TYPE, ROWS, COLS, NPPC>& _dst_half_ir) {
#pragma HLS INLINE OFF

    ap_int<4> R_IR_C1_wgts_loc[FSIZE1][FSIZE1], R_IR_C2_wgts_loc[FSIZE1][FSIZE1], B_at_R_wgts_loc[FSIZE1][FSIZE1];
    ap_int<4> IR_at_R_wgts_loc[FSIZE2][FSIZE2], IR_at_B_wgts_loc[FSIZE2][FSIZE2];
// clang-format off
#pragma HLS ARRAY_PARTITION variable=R_IR_C1_wgts_loc complete dim=0
#pragma HLS ARRAY_PARTITION variable=R_IR_C2_wgts_loc complete dim=0
#pragma HLS ARRAY_PARTITION variable=B_at_R_wgts_loc complete dim=0
#pragma HLS ARRAY_PARTITION variable=IR_at_R_wgts_loc complete dim=0
#pragma HLS ARRAY_PARTITION variable=IR_at_B_wgts_loc complete dim=0
// clang-format on

FILTER1_ROW:
    for (ap_int<4> i = 0; i < FSIZE1; i++) {
    FILTER1_COL:
        for (ap_int<4> j = 0; j < FSIZE1; j++) {
            R_IR_C1_wgts_loc[i][j] = (ap_int<4>)R_IR_C1_wgts[i * FSIZE1 + j];
            R_IR_C2_wgts_loc[i][j] = (ap_int<4>)R_IR_C2_wgts[i * FSIZE1 + j];
            B_at_R_wgts_loc[i][j] = (ap_int<4>)B_at_R_wgts[i * FSIZE1 + j];
        }
    }

FILTER2_ROW:
    for (ap_int<4> k = 0; k < FSIZE2; k++) {
    FILTER2_COL:
        for (ap_int<4> l = 0; l < FSIZE2; l++) {
            IR_at_R_wgts_loc[k][l] = (ap_int<4>)IR_at_R_wgts[k * FSIZE2 + l];
            IR_at_B_wgts_loc[k][l] = (ap_int<4>)IR_at_B_wgts[k * FSIZE2 + l];
        }
    }

    xf::cv::GenericFilter<RGBIR<TYPE, FSIZE1, FSIZE1, NPPC>, TYPE, TYPE, XFCV_DEPTH, ROWS, COLS, FSIZE1, FSIZE1, NPPC,
                          BORDER_T, USE_URAM, 1, BFORMAT>
        rgbIr_filter;

    rgbIr_filter.process_image(_src, R_IR_C1_wgts_loc, R_IR_C2_wgts_loc, B_at_R_wgts_loc, IR_at_R_wgts_loc,
                               IR_at_B_wgts_loc, _dst_rggb, _dst_half_ir);

    return;
}

// ======================================================================================

template <int FSIZE1 = 5,
          int FSIZE2 = 3,
          int BFORMAT = 0,
          int TYPE,
          int ROWS,
          int COLS,
          int NPPC = 1,
          int XFCV_DEPTH,
          int BORDER_T = XF_BORDER_CONSTANT,
          int USE_URAM = 0>
void rgbir2bayer(xf::cv::Mat<TYPE, ROWS, COLS, NPPC>& _src,
                 char R_IR_C1_wgts[FSIZE1 * FSIZE1],
                 char R_IR_C2_wgts[FSIZE1 * FSIZE1],
                 char B_at_R_wgts[FSIZE1 * FSIZE1],
                 char IR_at_R_wgts[FSIZE2 * FSIZE2],
                 char IR_at_B_wgts[FSIZE2 * FSIZE2],
                 char sub_wgts[4],
                 xf::cv::Mat<TYPE, ROWS, COLS, NPPC>& _dst_rggb,
                 xf::cv::Mat<TYPE, ROWS, COLS, NPPC>& _dst_ir) {
#pragma HLS DATAFLOW

    xf::cv::Mat<TYPE, ROWS, COLS, NPPC, 3 * COLS> rggbOutput(_src.rows, _src.cols);
    xf::cv::Mat<TYPE, ROWS, COLS, NPPC> halfIrOutput(_src.rows, _src.cols);
    xf::cv::Mat<TYPE, ROWS, COLS, NPPC> fullIrOutput(_src.rows, _src.cols);
    xf::cv::Mat<TYPE, ROWS, COLS, NPPC> fullIrOutput_copy1(_src.rows, _src.cols);
    xf::cv::Mat<TYPE, ROWS, COLS, NPPC> fullIrOutput_copy2(_src.rows, _src.cols);

    xf::cv::RGBIR_Demosaic<FSIZE1, FSIZE2, BFORMAT, TYPE, ROWS, COLS, NPPC, 3 * COLS, XF_BORDER_CONSTANT, USE_URAM>(
        _src, R_IR_C1_wgts, R_IR_C2_wgts, B_at_R_wgts, IR_at_R_wgts, IR_at_B_wgts, rggbOutput, halfIrOutput);
    xf::cv::IR_bilinear<BFORMAT>(halfIrOutput, fullIrOutput);
    xf::cv::duplicateMat(fullIrOutput, fullIrOutput_copy1, _dst_ir);
    xf::cv::weightedSub<BFORMAT, TYPE, ROWS, COLS, NPPC, 3 * COLS>(sub_wgts, rggbOutput, fullIrOutput_copy1, _dst_rggb);
}

} // namespace cv
} // namespace xf

#endif //_XF_RGBIR_HPP_
