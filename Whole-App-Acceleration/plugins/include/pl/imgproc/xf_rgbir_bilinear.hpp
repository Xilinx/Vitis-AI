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

#ifndef _XF_RGBIR_BILINEAR_HPP_
#define _XF_RGBIR_BILINEAR_HPP_

#include "common/xf_common.hpp"

namespace xf {
namespace cv {

// Internal constants
#define _NPPC (XF_NPIXPERCYCLE(NPPC))       // Number of pixels per clock to be processed
#define _NPPC_SHIFT_VAL (XF_BITSHIFT(NPPC)) // Gives log base 2 on NPPC; Used for shifting purpose in case of division
#define _ECPR ((((K_COLS >> 1) + (_NPPC - 1)) / _NPPC)) // Extra clocks required for processing a row
#define _NP_IN_PREV \
    (_NPPC - ((K_COLS >> 1) - (((K_COLS >> 1) / _NPPC) * _NPPC))) // No.of valid destination pixels in previous clock
#define _DST_PIX_WIDTH (XF_PIXELDEPTH(XF_DEPTH(DST_T, NPPC)))     // destination pixel width

/* Linear interpolation */
template <int TYPE, int NPC = 1>
XF_DTUNAME(TYPE, NPC)
interp1(XF_DTUNAME(TYPE, NPC) val1, XF_DTUNAME(TYPE, NPC) val2, XF_DTUNAME(TYPE, NPC) val) {
#pragma HLS INLINE OFF
    int ret = val1 + val * (val2 - val1);
    return xf::cv::xf_satcast<XF_DTPIXELDEPTH(TYPE, NPC)>(ret);
}

template <typename F,
          int SRC_T,
          int DST_T,
          int ROWS,
          int COLS,
          int K_ROWS,
          int K_COLS,
          int NPPC = 1,
          int BORDER_T = XF_BORDER_CONSTANT,
          int USE_URAM = 0,
          int USE_MAT = 1,
          int BFORMAT = 0>
class GenericFilter1 {
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
    unsigned int wr_ptr;             // Write pointer

    // Default Constructor
    GenericFilter1() {
        num_clks_per_row = 0;
        rd_ptr = 0;
        wr_ptr = 0;
    }

    // Internal functions
    void initialize(xf::cv::Mat<SRC_T, ROWS, COLS, NPPC>& _src);

    void process_row(unsigned short readRow,
                     unsigned short row,
                     xf::cv::Mat<SRC_T, ROWS, COLS, NPPC>& _src,
                     xf::cv::Mat<DST_T, ROWS, COLS, NPPC>& _dst_full_ir);
    void process_image(xf::cv::Mat<SRC_T, ROWS, COLS, NPPC>& _src, xf::cv::Mat<DST_T, ROWS, COLS, NPPC>& _dst_full_ir);
};

// -----------------------------------------------------------------------------------
// Function to initialize internal regsiters and buffers
// -----------------------------------------------------------------------------------
template <typename RGBIR,
          int SRC_T,
          int DST_T,
          int ROWS,
          int COLS,
          int K_ROWS,
          int K_COLS,
          int NPPC,
          int BORDER_T,
          int USE_URAM,
          int USE_MAT,
          int BFORMAT>
void GenericFilter1<RGBIR, SRC_T, DST_T, ROWS, COLS, K_ROWS, K_COLS, NPPC, BORDER_T, USE_URAM, USE_MAT, BFORMAT>::
    initialize(xf::cv::Mat<SRC_T, ROWS, COLS, NPPC>& _src) {
#pragma HLS INLINE

    // Computing no.of clocks required for processing a row of given image dimensions
    num_clks_per_row = (_src.cols + _NPPC - 1) >> _NPPC_SHIFT_VAL;

    // Read/Write pointer set to start location of input image
    rd_ptr = 0;
    wr_ptr = 0;

    return;
} // End of initialize()

// -----------------------------------------------------------------------------------
// Function to process a row
// -----------------------------------------------------------------------------------
template <typename IR_BILINEAR,
          int SRC_T,
          int DST_T,
          int ROWS,
          int COLS,
          int K_ROWS,
          int K_COLS,
          int NPPC,
          int BORDER_T,
          int USE_URAM,
          int USE_MAT,
          int BFORMAT>
void GenericFilter1<IR_BILINEAR, SRC_T, DST_T, ROWS, COLS, K_ROWS, K_COLS, NPPC, BORDER_T, USE_URAM, USE_MAT, BFORMAT>::
    process_row(unsigned short readRow,
                unsigned short row,
                xf::cv::Mat<SRC_T, ROWS, COLS, NPPC>& _src,
                xf::cv::Mat<DST_T, ROWS, COLS, NPPC>& _dst_full_ir) {
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
    short col_wo_npc = 0;
    short original_col = 0;

    // To store out pixels in packed format
    XF_TNAME(DST_T, NPPC) out_pixels;
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
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 1 max = _TC

#pragma HLS DEPENDENCE variable = buff.val inter false

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

#pragma HLS ARRAY_PARTITION variable = NxM_src_blk complete dim = 1
#pragma HLS ARRAY_PARTITION variable = NxM_src_blk complete dim = 2

        // Extract _NPPC, NxM-blocks from 'src_blk'
        REARRANGE_LOOP:
            for (unsigned short kr = 0; kr < K_ROWS; kr++) {
#pragma HLS UNROLL
                for (unsigned short kc = 0; kc < K_COLS; kc++) {
#pragma HLS UNROLL
                    NxM_src_blk[kr][kc] = src_blk.val[kr][pix_idx + kc];
                }
            }

            IR_BILINEAR oper;
            bool flag = 0;
            original_col = c * _NPPC + pix_idx;
            col_wo_npc = c * _NPPC - 1 + pix_idx;
            if ((BFORMAT == XF_BAYER_BG) || (BFORMAT == XF_BAYER_RG)) {
                if ((((row & 0x0001) == 1) && (col_wo_npc & 0x0001) == 1)) // BG, RG Mode - Even row, odd column and
                {                                                          //				odd row, even column
                    oper.apply_filter(NxM_src_blk, out_pix);
                } else {
                    out_pix = NxM_src_blk[K_ROWS / 2][K_COLS / 2];
                }

            } else if ((BFORMAT == XF_BAYER_GR)) {
                if (((row & 0x0001) == 0) && ((col_wo_npc & 0x0001) == 1)) // GB, GR Mode - This is even row, odd column
                {
                    oper.apply_filter(NxM_src_blk, out_pix);
                } else {
                    out_pix = NxM_src_blk[K_ROWS / 2][K_COLS / 2];
                }
            } else {
                if ((((row & 0x0001) == 0) && ((col_wo_npc & 0x0001) == 0)) ||
                    ((((row & 0x0001) == 1) &&
                      ((col_wo_npc & 0x0001) == 1)))) // GB, GR Mode - This is even row, even column
                {                                     // and odd row, odd column
                    oper.apply_filter(NxM_src_blk, out_pix);
                } else {
                    out_pix = NxM_src_blk[K_ROWS / 2][K_COLS / 2];
                }
            }
            // Start packing the out pixel value every clock of NPPC
            out_pixels.range(((pix_idx + 1) * _DST_PIX_WIDTH) - 1, (pix_idx * _DST_PIX_WIDTH)) = out_pix;
        }

        // Write the data out to DDR
        // .........................
        if (c >= _ECPR) {
            _dst_full_ir.write(wr_ptr++, out_pixels);
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
template <typename IR_BILINEAR,
          int SRC_T,
          int DST_T,
          int ROWS,
          int COLS,
          int K_ROWS,
          int K_COLS,
          int NPPC,
          int BORDER_T,
          int USE_URAM,
          int USE_MAT,
          int BFORMAT>
void GenericFilter1<IR_BILINEAR, SRC_T, DST_T, ROWS, COLS, K_ROWS, K_COLS, NPPC, BORDER_T, USE_URAM, USE_MAT, BFORMAT>::
    process_image(xf::cv::Mat<SRC_T, ROWS, COLS, NPPC>& _src, xf::cv::Mat<DST_T, ROWS, COLS, NPPC>& _dst_full_ir) {
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
#pragma HLS PIPELINE
#pragma HLS LOOP_TRIPCOUNT min = 1 max = _TC
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
#pragma HLS LOOP_TRIPCOUNT min = 1 max = _TC
            buff.val[r][c] = (BORDER_T == XF_BORDER_REPLICATE) ? buff.val[K_ROWS >> 1][c] : (XF_TNAME(SRC_T, NPPC))0;
        }
    }

    short int row = 0;
// ----------------------------------
// Processing each row of the image
// ----------------------------------
ROW_LOOP:
    for (unsigned short r = (K_ROWS >> 1); r < _src.rows + (K_ROWS >> 1); r++) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = ROWS
        process_row(r, row, _src, _dst_full_ir);
        row++;
    }

    return;
} // End of process_image()

// ======================================================================================

template <int SRC_T, int NPPC>
class IR_BILINEAR {
   public:
    const int NUM_CH = XF_CHANNELS(SRC_T, NPPC);

    // -------------------------------------------------------------------------
    // Creating apply function (applying filter)
    // -------------------------------------------------------------------------
    void apply_filter(XF_DTUNAME(SRC_T, NPPC) patch[3][3], XF_DTUNAME(SRC_T, NPPC) & pix) {
#pragma HLS INLINE OFF

        XF_DTUNAME(SRC_T, NPPC) partial_sum_0, partial_sum_1;
        XF_DTUNAME(SRC_T, NPPC) res;

        res = (patch[0][1] + patch[1][0] + patch[1][2] + patch[2][1]) >> 2;

        pix = res;
        return;
    }
};

template <int BPATTERN = 0,
          int TYPE,
          int ROWS,
          int COLS,
          int NPPC = 1,
          int BORDER_T = XF_BORDER_CONSTANT,
          int USE_URAM = 0>
void IR_bilinear(xf::cv::Mat<TYPE, ROWS, COLS, NPPC>& _src, xf::cv::Mat<TYPE, ROWS, COLS, NPPC>& _dst) {
    short int rd_idx = 0, wr_idx = 0;

    xf::cv::GenericFilter1<IR_BILINEAR<TYPE, NPPC>, TYPE, TYPE, ROWS, COLS, 3, 3, NPPC, BORDER_T, USE_URAM, 1, BPATTERN>
        Ir_filter;

    Ir_filter.process_image(_src, _dst);
}

//===============================================================================

/* Function to retrieve original R pixel value and replace in final image */

//===============================================================================

template <int BFORMAT = 0, int INTYPE, int OUTTYPE, int ROWS, int COLS, int NPPC = 1, int XFCV_DEPTH>
void copyRpixel(xf::cv::Mat<INTYPE, ROWS, COLS, NPPC, XFCV_DEPTH>& _src,
                xf::cv::Mat<OUTTYPE, ROWS, COLS, NPPC>& _src2,
                xf::cv::Mat<OUTTYPE, ROWS, COLS, NPPC>& _dst) {
#pragma HLS INLINE OFF
    int rd_index = 0, wr_index = 0;
    unsigned short candidateRow = 0, candidateCol = 0;
ROW_LOOP_COPYR:
    for (int r = 0; r < _src.rows; r++) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = ROWS
#pragma HLS LOOP_FLATTEN OFF
    COL_LOOP_COPYR:
        for (int c = 0, count = 0; c < _src.cols; c++, count++) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = COLS
#pragma HLS PIPELINE II = 1
            XF_TNAME(OUTTYPE, NPPC) inVal = _src.read(rd_index);
            XF_TNAME(OUTTYPE, NPPC) dstVal = _src2.read(rd_index++);
            if (BFORMAT == XF_BAYER_GR) {
                candidateRow = r + 1;
                candidateCol = c + 2;
            }
            if (((((candidateRow - 2) % 4) == 0) && ((candidateCol % 4) == 0)) ||
                (((candidateRow % 4) == 0) &&
                 (((candidateCol - 2) % 4) == 0))) { // BG Mode - This is even row, R location.

                dstVal.range((XF_DTPIXELDEPTH(INTYPE, NPPC) * 3) - 1, XF_DTPIXELDEPTH(INTYPE, NPPC) * 2) = inVal;
                _dst.write(wr_index++, dstVal);

            } else {
                _dst.write(wr_index++, dstVal);
            }
        }
    }
}

template <int BFORMAT = 0, int INTYPE, int ROWS, int COLS, int NPPC = 1, int XFCV_DEPTH>
void weightedSub(const char weights[4],
                 xf::cv::Mat<INTYPE, ROWS, COLS, NPPC, XFCV_DEPTH>& _src1,
                 xf::cv::Mat<INTYPE, ROWS, COLS, NPPC>& _src2,
                 xf::cv::Mat<INTYPE, ROWS, COLS, NPPC>& _dst) {
    ap_uint<4> wgts[4] = {0};
    for (int i = 0; i < 4; i++) {
        wgts[i] = weights[i];
    }
#pragma HLS INLINE OFF
    int rd_index = 0, wr_index = 0;
ROW_LOOP_COPYR:
    for (unsigned short row = 0; row < _src1.rows; row++) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = ROWS
#pragma HLS LOOP_FLATTEN OFF
    COL_LOOP_COPYR:
        for (unsigned short col = 0, count = 0; col < _src1.cols; col++) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = COLS
#pragma HLS PIPELINE II = 1
            XF_TNAME(INTYPE, NPPC) inVal1 = _src1.read(rd_index);
            XF_TNAME(INTYPE, NPPC) inVal2 = _src2.read(rd_index++);
            unsigned short tmp1 = 0;
            ap_int<17> tmp2 = 0;
            if (BFORMAT == XF_BAYER_GR) {
                if ((((row & 0x0001) == 0) && ((col & 0x0001) == 0)) ||
                    (((row & 0x0001) == 1) && ((col & 0x0001) == 1))) {        // G Pixel
                    tmp1 = inVal2 >> wgts[0];                                  // G has medium level of reduced weight
                } else if ((((row & 0x0001) == 0) && ((col & 0x0001) == 1))) { // R Pixel
                    tmp1 = inVal2 >> wgts[1];                                  // R has lowest level of reduced weight
                } else if (((((row - 1) % 4) == 0) && ((col % 4) == 0)) ||
                           ((((row + 1) % 4) == 0) && (((col - 2) % 4) == 0))) { // B Pixel
                    tmp1 = inVal2 >> wgts[2];                                    // B has low level of reduced weight
                } else if ((((((row - 1) % 4)) == 0) && (((col - 2) % 4) == 0)) ||
                           (((((row + 1) % 4)) == 0) && (((col) % 4) == 0))) { // Calculated B Pixel
                    tmp1 = inVal2 >> wgts[3];                                  // B has highest level of reduced weight
                }
            }
            if (BFORMAT == XF_BAYER_BG) {
                if ((((row & 0x0001) == 0) && ((col & 0x0001) == 1)) ||
                    (((row & 0x0001) == 1) && ((col & 0x0001) == 0))) {        // G Pixel
                    tmp1 = inVal2 >> wgts[0];                                  // G has medium level of reduced weight
                } else if ((((row & 0x0001) == 1) && ((col & 0x0001) == 1))) { // R Pixel
                    tmp1 = inVal2 >> wgts[1];                                  // R has lowest level of reduced weight
                } else if (((((row) % 4) == 0) && (((col) % 4) == 0)) ||
                           ((((row - 2) % 4) == 0) && (((col - 2) % 4) == 0))) { // B Pixel
                    tmp1 = inVal2 >> wgts[2];                                    // B has low level of reduced weight
                } else if ((((((row) % 4)) == 0) && (((col - 2) % 4) == 0)) ||
                           (((((row - 2) % 4)) == 0) && (((col) % 4) == 0))) { // Calculated B Pixel
                    tmp1 = inVal2 >> wgts[3];                                  // B has highest level of reduced weight
                }
            }
            tmp2 = inVal1 - tmp1;

            if (tmp2 < 0) {
                tmp2 = 0;
            }

            _dst.write(wr_index++, (XF_CTUNAME(INTYPE, NPPC))tmp2);
        }
    }
}
}
}
#endif
