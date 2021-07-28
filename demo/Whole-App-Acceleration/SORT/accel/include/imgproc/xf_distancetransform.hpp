/***************************************************************************
  Copyright (c) 2016, Xilinx, Inc.
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
 modification,
  are permitted provided that the following conditions are met:

  1. Redistributions of source code must retain the above copyright notice,
  this list of conditions and the following disclaimer.

  2. Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

  3. Neither the name of the copyright holder nor the names of its contributors
  may be used to endorse or promote products derived from this software
  without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 AND
  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
  THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 ARE DISCLAIMED.
  IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
 DIRECT, INDIRECT,
  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 LIMITED TO,
  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
 BUSINESS INTERRUPTION)
  HOWEVER CXFSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 LIABILITY,
  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 OF THIS SOFTWARE,
  EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

 ***************************************************************************/
#ifndef __XF_VITIS_DISTANCETRANSFORM_HPP__
#define __XF_VITIS_DISTANCETRANSFORM_HPP__

#include "ap_int.h"
#include "common/xf_common.hpp"
#include "common/xf_utility.hpp"
#include "common/xf_video_mem.hpp"
#include "hls_stream.h"

namespace xf {
namespace cv {

constexpr int _WINDOW_SIZE_ = 3;
constexpr int K_ROWS = _WINDOW_SIZE_;
constexpr int SET_MAX_VAL = (int)(2147483647 >> 2);

const int HV_DIST = (int)(0.954999983 * std::pow(2.0, 16.0));
const int DIAG_DIST = (int)(1.36930001 * std::pow(2.0, 16.0));
const float scale = 1.f / (1 << 16);

// Some global constants
typedef uint8_t K_COL_IDX_T;
typedef uint16_t COL_IDX_T; // Support upto 65,535
typedef uint16_t ROW_IDX_T; // Support upto 65,535
typedef uint32_t SIZE_IDX_T;

template <int IN_PTR, int FW_PTR, int ROWS, int COLS, int USE_URAM>
class dt_kernel_fw_pass {
   public:
    // Internal regsiters/buffers
    xf::cv::LineBuffer<1, COLS + 2, ap_uint<FW_PTR>, (USE_URAM ? RAM_S2P_URAM : RAM_S2P_BRAM), 1> buff;
    int im_h, im_w;

    // Internal Registers
    COL_IDX_T num_clks_per_row; // No.of clocks required for processing one row
    SIZE_IDX_T rd_ptr;          // Read pointer
    SIZE_IDX_T wr_ptr;          // Write pointer

    // Default Constructor
    dt_kernel_fw_pass() {
        num_clks_per_row = 0;
        rd_ptr = 0;
        wr_ptr = 0;
    }

    dt_kernel_fw_pass(int rows, int cols) {
        im_h = rows;
        im_w = cols;
    }

    // Internal functions
    void initialize_f() {
// clang-format off
#pragma HLS INLINE
        // clang-format on

        // Computing no.of clocks required for processing a row of given image
        // dimensions
        num_clks_per_row = im_w;

        // Read/Write pointer set to start location of input image
        rd_ptr = 0;
        wr_ptr = 0;

        return;
    };

    int fl = 0;

    void apply_f(ap_uint<IN_PTR> _src_data,
                 ap_uint<FW_PTR>& local_fw_pass_data,
                 ap_uint<FW_PTR> patch_top_0,
                 ap_uint<FW_PTR> patch_top_1,
                 ap_uint<FW_PTR> patch_top_2,
                 ap_uint<FW_PTR> patch_left) {
// clang-format off
#pragma HLS INLINE
        // clang-format on

        ap_uint<FW_PTR> tmp = 0;
        if (!_src_data)
            tmp = (ap_uint<FW_PTR>)0;
        else {
            int pt0 = patch_top_0 + DIAG_DIST;
            int pt1 = patch_top_1 + HV_DIST;
            int pt2 = patch_top_2 + DIAG_DIST;
            int pl = patch_left + HV_DIST;

            int t0 = (pt0 > pt1) ? pt1 : pt0;
            int t1 = (pt2 > pl) ? pl : pt2;
            tmp = (ap_uint<FW_PTR>)((t0 > t1) ? t1 : t0);
        }
        local_fw_pass_data = tmp;
    };

    void process_row_f(ROW_IDX_T r, ap_uint<IN_PTR>* _src, ap_uint<FW_PTR>* _fw_pass) {
// clang-format off
#pragma HLS INLINE
        // clang-format on

        // --------------------------------------
        // Constants
        // --------------------------------------
        const uint32_t _TC = COLS; // MAX Trip Count per row

        // --------------------------------------
        // Internal variables
        // --------------------------------------
        COL_IDX_T col_loop_cnt = num_clks_per_row;

        ap_uint<FW_PTR> patch_top[_WINDOW_SIZE_], patch_left;
        patch_top[1] = buff.val[0][0];
        patch_top[2] = buff.val[0][1];
        patch_left = SET_MAX_VAL;

    // --------------------------------------
    // Process columns of the row
    // --------------------------------------
    COL_LOOP:
        for (COL_IDX_T c = 0; c < col_loop_cnt; c++) {
// clang-format off
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=_TC
            // clang-format on
            patch_top[0] = patch_top[1];
            patch_top[1] = patch_top[2];
            patch_top[2] = buff.val[0][c + 2];

            ap_uint<FW_PTR> local_fw_pass_data;
            ap_uint<IN_PTR> in_data = _src[rd_ptr++];

            apply_f(in_data, local_fw_pass_data, patch_top[0], patch_top[1], patch_top[2], patch_left);

            buff.val[0][c + 1] = local_fw_pass_data;
            patch_left = local_fw_pass_data;
            _fw_pass[wr_ptr++] = local_fw_pass_data;
        }

        return;
    };

    bool process_image_f(ap_uint<IN_PTR>* _src, ap_uint<FW_PTR>* _fw_pass) {
// clang-format off
#pragma HLS INLINE OFF
        // clang-format on
        // Constant declaration
        const uint32_t _TC = COLS;

        // ----------------------------------
        // Start process with initialization
        // ----------------------------------
        initialize_f();

    // ----------------------------------
    // Initialize Line Buffer
    // ----------------------------------
    BORDER_INIT:
        for (COL_IDX_T c = 0; c < num_clks_per_row + 2; c++) {
// clang-format off
#pragma HLS PIPELINE
#pragma HLS LOOP_TRIPCOUNT min=1 max=_TC
            // clang-format on

            buff.val[0][c] = SET_MAX_VAL;
        }

    // ----------------------------------
    // Processing each row of the image
    // ----------------------------------
    ROW_LOOP:
        for (ROW_IDX_T r = 0; r < im_h; r++) {
// clang-format off
#pragma HLS LOOP_TRIPCOUNT min=1 max=ROWS
            // clang-format on

            process_row_f(r, _src, _fw_pass);
        }

        return 1;
    };
};

// ======================================================================================

template <int FW_PTR, int ROWS, int COLS, int USE_URAM>
class dt_kernel_bk_pass {
   public:
    // Internal regsiters/buffers
    xf::cv::LineBuffer<1, COLS + 2, ap_uint<FW_PTR>, (USE_URAM ? RAM_S2P_URAM : RAM_S2P_BRAM), 1> buff;
    int im_h, im_w;
    bool flag;

    // Internal Registers
    COL_IDX_T num_clks_per_row; // No.of clocks required for processing one row
    SIZE_IDX_T rd_ptr;          // Read pointer
    SIZE_IDX_T wr_ptr;          // Write pointer

    // ping-pong BRAMs for forward-pass data and backward distance information
    ap_uint<FW_PTR> fw_ram1[COLS], fw_ram2[COLS];
    float dist_ram1[COLS], dist_ram2[COLS];

    // Default Constructor
    dt_kernel_bk_pass() {
        num_clks_per_row = 0;
        rd_ptr = 0;
        wr_ptr = 0;
    }

    dt_kernel_bk_pass(int rows, int cols) {
        im_h = rows;
        im_w = cols;
    }

    // Internal functions
    void initialize_b() {
// clang-format off
#pragma HLS INLINE
        // clang-format on

        // Computing no.of clocks required for processing a row of given image
        // dimensions
        num_clks_per_row = im_w;

        int total_copy = im_h * im_w;

        // Read/Write pointer set to start location of input image
        rd_ptr = total_copy;
        wr_ptr = total_copy;

        // Initializing the flags to '0'
        flag = 0;

        return;
    };

    void apply_b(ap_uint<FW_PTR> _fw_data,
                 ap_uint<FW_PTR>& local_dist_data,
                 ap_uint<FW_PTR> patch_top_0,
                 ap_uint<FW_PTR> patch_top_1,
                 ap_uint<FW_PTR> patch_top_2,
                 ap_uint<FW_PTR> patch_left) {
// clang-format off
#pragma HLS INLINE
        // clang-format on

        int dist = (int)_fw_data;
        if (dist > HV_DIST) {
            int t_d = dist;
            int pt0 = patch_top_0 + DIAG_DIST;
            int pt1 = patch_top_1 + HV_DIST;
            int pt2 = patch_top_2 + DIAG_DIST;
            int pl = patch_left + HV_DIST;

            int t0 = (pt0 > pt1) ? pt1 : pt0;
            int t1 = (pt2 > pl) ? pl : pt2;
            int t2 = (t0 > t1) ? t1 : t0;
            dist = (t_d > t2) ? t2 : t_d;
        }
        local_dist_data = (ap_uint<FW_PTR>)dist;
    };

    void process_row_b(ap_uint<FW_PTR>* _fw_ram, float* _dist_ram) {
// clang-format off
#pragma HLS INLINE OFF
        // clang-format on

        // --------------------------------------
        // Constants
        // --------------------------------------
        const uint32_t _TC = COLS; // MAX Trip Count per row

        // --------------------------------------
        // Internal variables
        // --------------------------------------
        COL_IDX_T col_loop_cnt = num_clks_per_row;

        ap_uint<FW_PTR> patch_top[_WINDOW_SIZE_], patch_left;
        patch_top[1] = buff.val[0][0];
        patch_top[2] = buff.val[0][1];
        patch_left = SET_MAX_VAL;

    // --------------------------------------
    // Process columns of the row
    // --------------------------------------
    COL_LOOP:
        for (COL_IDX_T c = 0; c < col_loop_cnt; c++) {
// clang-format off
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=_TC
            // clang-format on

            patch_top[0] = patch_top[1];
            patch_top[1] = patch_top[2];
            patch_top[2] = buff.val[0][c + 2];

            ap_uint<FW_PTR> local_dist_data;
            ap_uint<FW_PTR> fw_data = _fw_ram[im_w - c - 1];

            apply_b(fw_data, local_dist_data, patch_top[0], patch_top[1], patch_top[2], patch_left);

            buff.val[0][c + 1] = local_dist_data;
            patch_left = local_dist_data;
            float tmp = ((float)local_dist_data * scale);
            _dist_ram[im_w - c - 1] = tmp;
        }

        return;
    };

    void read_fw_to_ram(ap_uint<FW_PTR>* _fw_pass, ap_uint<FW_PTR>* ram, int _rd_ptr) {
// clang-format off
#pragma HLS INLINE OFF
        // clang-format on
        int ptr = _rd_ptr;
        for (int j = 0; j < im_w; j++) {
// clang-format off
#pragma HLS LOOP_TRIPCOUNT min=1 max=COLS		
#pragma HLS PIPELINE II=1
            // clang-format on
            ram[j] = _fw_pass[ptr++];
        }
    };

    void write_dist_to_mem(float* ram, float* _dst, int _wr_ptr) {
// clang-format off
#pragma HLS INLINE OFF
        // clang-format on
        int ptr = _wr_ptr;
        for (int j = 0; j < im_w; j++) {
// clang-format off
#pragma HLS LOOP_TRIPCOUNT min=1 max=COLS		
#pragma HLS PIPELINE II=1
            // clang-format on
            _dst[ptr++] = ram[j];
        }
    };

    void process_image_b(ap_uint<FW_PTR>* _fw_pass, float* _dst) {
// clang-format off
#pragma HLS INLINE OFF
        // clang-format on

        // Constant declaration
        const uint32_t _TC = COLS;

        // ----------------------------------
        // Start process with initialization
        // ----------------------------------
        initialize_b();

    // ----------------------------------
    // Initialize Line Buffer
    // ----------------------------------
    BORDER_INIT:
        for (COL_IDX_T c = 0; c < num_clks_per_row + 2; c++) {
// clang-format off
#pragma HLS PIPELINE
#pragma HLS LOOP_TRIPCOUNT min=1 max=_TC
            // clang-format on

            buff.val[0][c] = SET_MAX_VAL;
        }

        // ----------------------------------
        // Processing each row of the image
        // ----------------------------------
        rd_ptr -= im_w;
        read_fw_to_ram(_fw_pass, fw_ram1, rd_ptr);

        rd_ptr -= im_w;
        read_fw_to_ram(_fw_pass, fw_ram2, rd_ptr);
        process_row_b(fw_ram1, dist_ram1);

    ROW_LOOP:
        for (ROW_IDX_T r = 0; r < im_h - 2; r++) {
// clang-format off
#pragma HLS LOOP_TRIPCOUNT min=1 max=ROWS
            // clang-format on

            rd_ptr -= im_w;
            wr_ptr -= im_w;
            if (flag == 0) {
                read_fw_to_ram(_fw_pass, fw_ram1, rd_ptr);
                process_row_b(fw_ram2, dist_ram2);
                write_dist_to_mem(dist_ram1, _dst, wr_ptr);
                flag = 1;
            } else {
                read_fw_to_ram(_fw_pass, fw_ram2, rd_ptr);
                process_row_b(fw_ram1, dist_ram1);
                write_dist_to_mem(dist_ram2, _dst, wr_ptr);
                flag = 0;
            }
        }

        wr_ptr -= im_w;
        if (flag == 0) {
            process_row_b(fw_ram2, dist_ram2);
            write_dist_to_mem(dist_ram1, _dst, wr_ptr);
            flag = 1;
        } else {
            process_row_b(fw_ram1, dist_ram1);
            write_dist_to_mem(dist_ram2, _dst, wr_ptr);
            flag = 0;
        }

        wr_ptr -= im_w;
        if (flag == 0) {
            write_dist_to_mem(dist_ram1, _dst, wr_ptr);
        } else {
            write_dist_to_mem(dist_ram2, _dst, wr_ptr);
        }

        return;
    };
};

// ======================================================================================

template <int IN_PTR, int FW_PTR, int ROWS, int COLS, int USE_URAM>
void distanceTransform(ap_uint<IN_PTR>* _src, float* _dst, ap_uint<FW_PTR>* _fw_pass, int rows, int cols) {
// clang-format off
#pragma HLS INLINE OFF
    // clang-format on

    assert(((rows <= ROWS) && (cols <= COLS)) &&
           "ROWS and COLS must be greater or equal torows and cols respectively.");
    assert((IN_PTR == 8) &&
           "The input must be a grayscale image, encoded with "
           "binary values (0 or 255), which means the pointer "
           "width must be '8'.");
    assert((FW_PTR == 32) && "FW_PTR, is the forwards-pass datawidth, which must be '32'.");

    xf::cv::dt_kernel_fw_pass<IN_PTR, FW_PTR, ROWS, COLS, USE_URAM> dt_fw(rows, cols);
    xf::cv::dt_kernel_bk_pass<FW_PTR, ROWS, COLS, USE_URAM> dt_bk(rows, cols);

    for (int i = 0; i < 2; i++) {
        if (i == 0)
            dt_fw.process_image_f(_src, _fw_pass);
        else
            dt_bk.process_image_b(_fw_pass, _dst);
    }

    return;
}

} // namespace cv
} // namespace xf

#endif //__XF_VITIS_DISTANCETRANSFORM_HPP__
