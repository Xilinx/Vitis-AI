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

#ifndef _XF_BPC_HPP_
#define _XF_BPC_HPP_

#include "ap_int.h"
#include "common/xf_utility.hpp"
#include "common/xf_video_mem.hpp"
//#include "xf_extra_utility.h"

// int glpbal=0;

namespace xf {

namespace cv {

// ======================================================================================
// A generic structure for BPC operation
// --------------------------------------------------------------------------------------
// Template Args:-
//        SRC_T : Data type of soruce image element
//         ROWS : Image height
//         COLS : Image width
//         NPPC : No.of pixels per clock
//     BORDER_T : Type of boder to be used for edge pixel(s) computation
// ......................................................................................

// Some macros related to template (for easiness of coding)
#define _GENERIC_BPC_TPLT_DEC                                                                  \
    template <typename F, int SRC_T, int ROWS, int COLS, int K_ROWS, int K_COLS, int NPPC = 1, \
              int BORDER_T = XF_BORDER_CONSTANT, int USE_URAM = 0>
#define _GENERIC_BPC_TPLT \
    template <typename F, int SRC_T, int ROWS, int COLS, int K_ROWS, int K_COLS, int NPPC, int BORDER_T, int USE_URAM>
#define _GENERIC_BPC GenericBPC<F, SRC_T, ROWS, COLS, K_ROWS, K_COLS, NPPC, BORDER_T, USE_URAM>

// Some global constants
#define CH_IDX_T uint8_t
#define K_ROW_IDX_T uint8_t
#define K_COL_IDX_T uint8_t
#define COL_IDX_T uint16_t // Support upto 65,535
#define ROW_IDX_T uint16_t // Support upto 65,535
#define SIZE_IDX_T uint32_t

// Some internal constants
#define _NPPC (XF_NPIXPERCYCLE(NPPC))                   // Number of pixel per clock to be processed
#define _NPPC_SHIFT_VAL (XF_BITSHIFT(NPPC))             // Gives log base 2 on NPPC; Used for shifting purpose in
                                                        // case of division
#define _ECPR ((((K_COLS >> 1) + (_NPPC - 1)) / _NPPC)) // Extra clocks required for processing a row
#define _NP_IN_PREV \
    (_NPPC - ((K_COLS >> 1) - (((K_COLS >> 1) / _NPPC) * _NPPC))) // No.of valid destination pixels in previous clock
#define _DST_PIX_WIDTH (XF_PIXELDEPTH(XF_DEPTH(SRC_T, NPPC)))     // destination pixel width

_GENERIC_BPC_TPLT_DEC class GenericBPC {
   public:
    // Internal regsiters/buffers
    xf::cv::Window<K_ROWS, XF_NPIXPERCYCLE(NPPC) + (K_COLS - 1), XF_DTUNAME(SRC_T, NPPC)>
        src_blk;                                 // Kernel sized image block with pixel parallelism
    xf::cv::Scalar<K_ROWS, K_ROW_IDX_T> row_idx; // To store row index for circular buffer access
    xf::cv::LineBuffer<K_ROWS,
                       (COLS >> _NPPC_SHIFT_VAL),
                       XF_TNAME(SRC_T, NPPC),
                       (USE_URAM ? RAM_S2P_URAM : RAM_S2P_BRAM),
                       (USE_URAM ? K_ROWS : 1)>
        buff; // Line Buffer for K_ROWS from the image

    // Internal Registers
    COL_IDX_T num_clks_per_row; // No.of clocks required for processing one row
    SIZE_IDX_T rd_ptr;          // Read pointer
    SIZE_IDX_T wr_ptr;          // Write pointer
    // uint8_t          threshold;        // Threshold value used to classify as
    // similar (0) / brighter(1) / darker(2)

    // Default Constructor
    GenericBPC() {
// clang-format off
#pragma HLS INLINE
        // clang-format on
        num_clks_per_row = 0;
        rd_ptr = 0;
        wr_ptr = 0;
    }

    // Internal functions
    void initialize(xf::cv::Mat<SRC_T, ROWS, COLS, NPPC>& _src);

    void update_row_idx();
    void process_row(ROW_IDX_T r,
                     xf::cv::Mat<SRC_T, ROWS, COLS, NPPC>& _src,
                     xf::cv::Mat<SRC_T, ROWS, COLS, NPPC>& _dst);
    void process_image(xf::cv::Mat<SRC_T, ROWS, COLS, NPPC>& _src, xf::cv::Mat<SRC_T, ROWS, COLS, NPPC>& _dst);
};

// -----------------------------------------------------------------------------------
// Function to initialize internal regsiters and buffers
// -----------------------------------------------------------------------------------
_GENERIC_BPC_TPLT void _GENERIC_BPC::initialize(xf::cv::Mat<SRC_T, ROWS, COLS, NPPC>& _src) {
#pragma HLS INLINE

    // Computing no.of clocks required for processing a row of given image
    // dimensions
    num_clks_per_row = (_src.cols + _NPPC - 1) >> _NPPC_SHIFT_VAL;

    // Read/Write pointer set to start location of input image
    rd_ptr = 0;
    wr_ptr = 0;

    // Initialize row-index values
    for (K_ROW_IDX_T kr = 0; kr < K_ROWS; kr++) {
#pragma HLS UNROLL
        row_idx.val[kr] = kr;
    }

    return;
} // End of initialize()

// -----------------------------------------------------------------------------------
// Function to process a row
// -----------------------------------------------------------------------------------
_GENERIC_BPC_TPLT void _GENERIC_BPC::process_row(ROW_IDX_T r,
                                                 xf::cv::Mat<SRC_T, ROWS, COLS, NPPC>& _src,
                                                 xf::cv::Mat<SRC_T, ROWS, COLS, NPPC>& _dst) {
#pragma HLS INLINE OFF

    // --------------------------------------
    // Constants
    // --------------------------------------
    const uint32_t _TC = (COLS >> _NPPC_SHIFT_VAL) + (K_COLS >> 1); // MAX Trip Count per row

    // --------------------------------------
    // Internal variables
    // --------------------------------------
    // Loop count variable
    COL_IDX_T col_loop_cnt = num_clks_per_row + _ECPR;
    ap_uint<32> pix_pos;
    short col = -(K_COLS >> 1);

    // To store out pixels in packed format
    XF_TNAME(SRC_T, NPPC) out_pixels, prev_out_pixels;

// --------------------------------------
// Initialize source block buffer to all zeros
// --------------------------------------
SRC_INIT_LOOP:
    for (K_ROW_IDX_T kr = 0; kr < K_ROWS; kr++) {
#pragma HLS UNROLL

        for (K_COL_IDX_T kc = 0; kc < (_NPPC + K_COLS - 1); kc++) {
#pragma HLS UNROLL
            src_blk.val[kr][kc] = 0;
        }
    }

// --------------------------------------
// Process columns of the row
// --------------------------------------
COL_LOOP:
    for (COL_IDX_T c = 0; c < col_loop_cnt; c++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 1 max = _TC
        //#pragma HLS LOOP_FLATTEN OFF

        // Fetch next row of source image and store in internal RAMs
        // .........................................................
        if ((r < _src.rows) && (c < num_clks_per_row)) {
            buff.val[row_idx.val[K_ROWS - 1]][c] = _src.read(rd_ptr++);
        }

    // Fetch data from RAMs and store in 'src_blk' for processing
    // .........................................................
    BUFF_RD_LOOP:
        for (K_ROW_IDX_T kr = 0; kr < K_ROWS; kr++) {
#pragma HLS UNROLL
            XF_TNAME(SRC_T, NPPC) tmp_rd_buff;

            // Read packed data
            tmp_rd_buff = buff.val[row_idx.val[kr]][c]; // tmp_rd_buff = (c < num_clks_per_row)
                                                        // ? buff.val[row_idx.val[kr]][c] :
                                                        // (XF_TNAME(SRC_T, NPPC))0;

            // Extract pixels from packed data and store in 'src_blk'
            xfExtractPixels<NPPC, XF_WORDWIDTH(SRC_T, NPPC), XF_DEPTH(SRC_T, NPPC)>(src_blk.val[kr], tmp_rd_buff,
                                                                                    (K_COLS - 1));
        }
    // if (c >= _ECPR) {
    //   xFSetBorder<K_ROWS, (_NPPC + K_COLS-1), XF_DTUNAME(SRC_T, NPPC),
    //   BORDER_T>(src_blk, r, (c<<_NPPC_SHIFT_VAL),
    //   _src.rows, _src.cols);
    //}

    // Process the kernel block
    // ........................
    PROCESS_BLK_LOOP:
        for (int pix_idx = 0; pix_idx < _NPPC; pix_idx++) {
#pragma HLS UNROLL
            XF_DTUNAME(SRC_T, NPPC) NxM_src_blk[K_ROWS][K_COLS];
// clang-format off
#pragma HLS ARRAY_PARTITION variable = NxM_src_blk complete
            // clang-format on
            XF_DTUNAME(SRC_T, NPPC) out_pix;

        // Extract _NPPC, NxM-blocks from 'src_blk'
        REARRANGE_LOOP:
            for (K_ROW_IDX_T kr = 0; kr < K_ROWS; kr++) {
#pragma HLS UNROLL
                for (K_COL_IDX_T kc = 0; kc < K_COLS; kc++) {
#pragma HLS UNROLL
                    NxM_src_blk[kr][kc] = src_blk.val[kr][pix_idx + kc];
                }
            }

            // Apply the filter on the NxM_src_blk
            F oper;
            oper.apply(NxM_src_blk, &out_pix);

            // Start packing the out pixel value every clock of NPPC
            out_pixels.range(((pix_idx + 1) * _DST_PIX_WIDTH) - 1, (pix_idx * _DST_PIX_WIDTH)) = out_pix;
        }

        col = col + _NPPC;

        // Write the data out to DDR
        // .........................
        if (c >= _ECPR) {
            if (_NP_IN_PREV == _NPPC) { // Case of (K_COLS / 2) is divisible by NPPC
                _dst.write(wr_ptr++, out_pixels);
            } else {
                // Taking '_NP_IN_PREV' pixels from 'prev_out_pixels' (MSB side) and
                // (_NPPC - _NP_IN_PREV) from
                // 'out_pixels' (LSB)
                prev_out_pixels.range((_NP_IN_PREV * _DST_PIX_WIDTH) - 1, 0) =
                    prev_out_pixels.range((_NPPC * _DST_PIX_WIDTH) - 1, ((_NPPC - _NP_IN_PREV) * _DST_PIX_WIDTH));
                prev_out_pixels.range((_NPPC * _DST_PIX_WIDTH) - 1, (_NP_IN_PREV * _DST_PIX_WIDTH)) =
                    out_pixels.range(((_NPPC - _NP_IN_PREV) * _DST_PIX_WIDTH) - 1, 0);

                _dst.write(wr_ptr++, prev_out_pixels);
            }
        }
        prev_out_pixels = out_pixels;

    // Now get ready for next cycle of coputation. So copy the last K_COLS-1 data
    // to start location of 'src_blk'
    // ...........................................
    SHIFT_LOOP:
        for (K_ROW_IDX_T kr = 0; kr < K_ROWS; kr++) {
#pragma HLS UNROLL
            for (K_COL_IDX_T kc = 0; kc < K_COLS - 1; kc++) {
#pragma HLS UNROLL
                src_blk.val[kr][kc] = src_blk.val[kr][_NPPC + kc];
            }
        }
    }

    return;
} // End of process_row()

// -----------------------------------------------------------------------------------
// Function to update row index (Cyclic shift)
// -----------------------------------------------------------------------------------
_GENERIC_BPC_TPLT void _GENERIC_BPC::update_row_idx() {
#pragma HLS INLINE OFF

    K_ROW_IDX_T tmp_idx = row_idx.val[0];

    for (K_ROW_IDX_T kr = 0; kr < K_ROWS - 1; kr++) {
#pragma HLS UNROLL
        row_idx.val[kr] = row_idx.val[kr + 1];
    }
    row_idx.val[K_ROWS - 1] = tmp_idx;

    return;
} // End of update_row_idx

// -----------------------------------------------------------------------------------
// Main function that runs the filter over the image
// -----------------------------------------------------------------------------------
_GENERIC_BPC_TPLT void _GENERIC_BPC::process_image(xf::cv::Mat<SRC_T, ROWS, COLS, NPPC>& _src,
                                                   xf::cv::Mat<SRC_T, ROWS, COLS, NPPC>& _dst) {
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
//        Start filling rows from (kernel height)/2 and rest depending on border
//        type
READ_LINES_INIT:
    for (K_ROW_IDX_T r = (K_ROWS >> 1); r < (K_ROWS - 1); r++) { // Note: Ignoring last row
#pragma HLS UNROLL
        for (COL_IDX_T c = 0; c < num_clks_per_row; c++) {
#pragma HLS PIPELINE
#pragma HLS LOOP_TRIPCOUNT min = 1 max = _TC
            buff.val[r][c] = _src.read(rd_ptr++); // Reading the rows of image
        }
    }
// Part2: Take care of borders depending on border type.
//        In border replicate mode, fill with 1st row of the image.
BORDER_INIT:
    for (K_ROW_IDX_T r = 0; r < (K_ROWS >> 1); r++) {
#pragma HLS UNROLL
        for (COL_IDX_T c = 0; c < num_clks_per_row; c++) {
#pragma HLS PIPELINE
#pragma HLS LOOP_TRIPCOUNT min = 1 max = _TC
            buff.val[r][c] = (BORDER_T == XF_BORDER_REPLICATE) ? buff.val[K_ROWS >> 1][c] : (XF_TNAME(SRC_T, NPPC))0;
        }
    }

// ----------------------------------
// Processing each row of the image
// ----------------------------------
ROW_LOOP:
    for (ROW_IDX_T r = (K_ROWS >> 1); r < _src.rows + (K_ROWS >> 1); r++) {
//#pragma HLS PIPELINE
#pragma HLS LOOP_TRIPCOUNT min = 1 max = ROWS
        process_row(r, _src, _dst);
        update_row_idx();
    }

    return;
} // End of process_image()

// ======================================================================================

// ======================================================================================
// Class for BPC computation
// ======================================================================================
#define _BPC_P_SIZE 5

template <int SRC_T, int NPPC>
class BPC {
   public:
    // -------------------------------------------------------------------------
    // Creating apply function
    // Inputs: patch of NxN size
    // Ouputs: out_pix
    // -------------------------------------------------------------------------
    void apply(XF_DTUNAME(SRC_T, NPPC) patch[_BPC_P_SIZE][_BPC_P_SIZE], XF_DTUNAME(SRC_T, NPPC) * out_pix) {
#pragma HLS INLINE

        XF_DTUNAME(SRC_T, NPPC) out_val;
        XF_DTUNAME(SRC_T, NPPC) array[9];
        XF_DTUNAME(SRC_T, NPPC) array_channel[8];
#pragma HLS ARRAY_PARTITION variable = array complete dim = 1

        int array_ptr = 0;
    Compute_Grad_Loop:
        for (int copy_arr = 0; copy_arr < _BPC_P_SIZE; copy_arr = copy_arr + 2) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 5
#pragma HLS UNROLL
            for (int copy_in = 0; copy_in < _BPC_P_SIZE; copy_in = copy_in + 2) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 5
#pragma HLS UNROLL
                array[array_ptr] = patch[copy_arr][copy_in];
                array_ptr++;
            }
        }
        // for(int channel=0,k=0;channel<PLANES;channel++,k+=8)
        //	{
        //#pragma HLS LOOP_TRIPCOUNT min=1 max=PLANES
        //#pragma HLS UNROLL

        for (int p = 0; p < 4; p++) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 9
#pragma HLS UNROLL
            array_channel[p] = array[p];
        }
        for (int l = 4; l < 8; l++) {
            array_channel[l] = array[l + 1];
        }
        XF_DTUNAME(SRC_T, NPPC) min = array_channel[0];
        XF_DTUNAME(SRC_T, NPPC) max = array_channel[0];
    xFApplyMaskLoop:
        for (int16_t j = 1; j < 8; j++) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 9
            //#pragma HLS LOOP_FLATTEN off

            if (array_channel[j] > max) {
                max = array_channel[j];
            }
            if (array_channel[j] < min) {
                min = array_channel[j];
            }
        }
        XF_DTUNAME(SRC_T, NPPC) finalout = 0;

        if (array[4] < min)
            finalout = min;
        else if (array[4] > max)
            finalout = max;
        else
            finalout = array[4];

        out_val = finalout;
        //	}

        *out_pix = out_val;

        return;
    }
};

// ======================================================================================

// ======================================================================================
// Top BPC API
// --------------------------------------------------------------------------------------
// Template Args:-
//         TYPE : Data type of soruce image element
//         ROWS : Image height
//         COLS : Image width
//         NPPC : No.of pixels per clock
//     BORDER_T : Type of boder to be used for edge pixel(s) computation
//                (XF_BORDER_REPLICATE, XF_BORDER_CONSTANT,
//                XF_BORDER_REFLECT_101, XF_BORDER_REFLECT)
// ......................................................................................
#define _BPC_ BPC<TYPE, NPPC>

// --------------------------------------------------------------------------------------
// Below function will generate list of corners
// --------------------------------------------------------------------------------------
template <int TYPE, int ROWS, int COLS, int NPPC = 1, int BORDER_T = XF_BORDER_CONSTANT, int USE_URAM = 0>
void badpixelcorrection(xf::cv::Mat<TYPE, ROWS, COLS, NPPC>& _src, xf::cv::Mat<TYPE, ROWS, COLS, NPPC>& _dst) {
// clang-format off
#pragma HLS INLINE OFF
    // clang-format on
    GenericBPC<_BPC_, TYPE, ROWS, COLS, _BPC_P_SIZE, _BPC_P_SIZE, NPPC, BORDER_T, USE_URAM> bpc;

    bpc.process_image(_src, _dst);

    return;
}

#undef _BPC_

// ======================================================================================

// Some clean up for macros used
#undef _BPC_P_SIZE
#undef _BPC_NMS_P_SIZE
#undef _GENERIC_BPC_TPLT_DEC
#undef _GENERIC_BPC_TPLT
#undef _GENERIC_BPC

#undef CH_IDX_T
#undef K_ROW_IDX_T
#undef K_COL_IDX_T
#undef COL_IDX_T
#undef ROW_IDX_T
#undef SIZE_IDX_T

#undef _NPPC
#undef _NPPC_SHIFT_VAL
#undef _ECPR
#undef _NP_IN_PREV
#undef _DST_PIX_WIDTH

} // namespace xf
}

#endif //_XF_BPC_HPP_
