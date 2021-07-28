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

#ifndef __XF_EXTRA_UTILITY_H__
#define __XF_EXTRA_UTILITY_H__

#include "common/xf_common.hpp"
#include "common/xf_video_mem.hpp"

namespace xf {
namespace cv {

// ======================================================================================
// Function to split hls::stream into 2 hls::stream
// ======================================================================================
template <int SRC_T, int ROWS, int COLS, int NPC>
void DuplicateStrm(hls::stream<XF_TNAME(SRC_T, NPC)>& _src_mat,
                   hls::stream<XF_TNAME(SRC_T, NPC)>& _dst1_mat,
                   hls::stream<XF_TNAME(SRC_T, NPC)>& _dst2_mat,
                   uint16_t img_height,
                   uint16_t img_width) {
    img_width = img_width >> XF_BITSHIFT(NPC);
    ap_uint<13> row, col;

Row_Loop:
    for (row = 0; row < img_height; row++) {
#pragma HLS LOOP_TRIPCOUNT min = ROWS max = ROWS
#pragma HLS LOOP_FLATTEN off
    Col_Loop:
        for (col = 0; col < img_width; col++) {
#pragma HLS LOOP_TRIPCOUNT min = COLS / NPC max = COLS / NPC
#pragma HLS pipeline
            XF_TNAME(SRC_T, NPC) tmp_src;
            tmp_src = _src_mat.read();
            _dst1_mat.write(tmp_src);
            _dst2_mat.write(tmp_src);
        }
    }
} // End of DuplicateStrm()
// ======================================================================================

// ======================================================================================
// Function to split hls::stream into 3 hls::stream
// ======================================================================================
template <int SRC_T, int ROWS, int COLS, int NPC>
void DuplicateStrm_3(hls::stream<XF_TNAME(SRC_T, NPC)>& _src_mat,
                     hls::stream<XF_TNAME(SRC_T, NPC)>& _dst1_mat,
                     hls::stream<XF_TNAME(SRC_T, NPC)>& _dst2_mat,
                     hls::stream<XF_TNAME(SRC_T, NPC)>& _dst3_mat,
                     uint16_t img_height,
                     uint16_t img_width) {
    img_width = img_width >> XF_BITSHIFT(NPC);
    ap_uint<13> row, col;

Row_Loop:
    for (row = 0; row < img_height; row++) {
#pragma HLS LOOP_TRIPCOUNT min = ROWS max = ROWS
#pragma HLS LOOP_FLATTEN off
    Col_Loop:
        for (col = 0; col < img_width; col++) {
#pragma HLS LOOP_TRIPCOUNT min = COLS / NPC max = COLS / NPC
#pragma HLS pipeline
            XF_TNAME(SRC_T, NPC) tmp_src;
            tmp_src = _src_mat.read();
            _dst1_mat.write(tmp_src);
            _dst2_mat.write(tmp_src);
            _dst3_mat.write(tmp_src);
        }
    }
} // End of DuplicateStrm_3()
// ======================================================================================

// ======================================================================================
// Function to split hls::stream into 4 hls::stream
// ======================================================================================
template <int SRC_T, int ROWS, int COLS, int NPC>
void DuplicateStrm_4(hls::stream<XF_TNAME(SRC_T, NPC)>& _src_mat,
                     hls::stream<XF_TNAME(SRC_T, NPC)>& _dst1_mat,
                     hls::stream<XF_TNAME(SRC_T, NPC)>& _dst2_mat,
                     hls::stream<XF_TNAME(SRC_T, NPC)>& _dst3_mat,
                     hls::stream<XF_TNAME(SRC_T, NPC)>& _dst4_mat,
                     uint16_t img_height,
                     uint16_t img_width) {
    img_width = img_width >> XF_BITSHIFT(NPC);
    ap_uint<13> row, col;

Row_Loop:
    for (row = 0; row < img_height; row++) {
#pragma HLS LOOP_TRIPCOUNT min = ROWS max = ROWS
#pragma HLS LOOP_FLATTEN off
    Col_Loop:
        for (col = 0; col < img_width; col++) {
#pragma HLS LOOP_TRIPCOUNT min = COLS / NPC max = COLS / NPC
#pragma HLS pipeline
            XF_TNAME(SRC_T, NPC) tmp_src;
            tmp_src = _src_mat.read();
            _dst1_mat.write(tmp_src);
            _dst2_mat.write(tmp_src);
            _dst3_mat.write(tmp_src);
            _dst4_mat.write(tmp_src);
        }
    }
} // End of DuplicateStrm_4()
// ======================================================================================

// ======================================================================================
// Function to split xf::cv::Mat into 2 xf::cv::Mat
// ======================================================================================
template <int SRC_T, int ROWS, int COLS, int NPC>
void duplicateMat(xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _src,
                  xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _dst1,
                  xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _dst2) {
#pragma HLS inline off

#pragma HLS dataflow

    int _rows = _src.rows;
    int _cols = _src.cols;

    hls::stream<XF_TNAME(SRC_T, NPC)> src;
    hls::stream<XF_TNAME(SRC_T, NPC)> dst;
    hls::stream<XF_TNAME(SRC_T, NPC)> dst1;

    for (int i = 0; i < _rows; i++) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = ROWS
        for (int j = 0; j<(_cols)>> (XF_BITSHIFT(NPC)); j++) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = COLS / NPC
#pragma HLS PIPELINE
#pragma HLS loop_flatten off
            src.write(_src.read(i * (_cols >> (XF_BITSHIFT(NPC))) + j));
        }
    }

    DuplicateStrm<SRC_T, ROWS, COLS, NPC>(src, dst, dst1, _rows, _cols);

    for (int i = 0; i < _rows; i++) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = ROWS
        for (int j = 0; j<(_cols)>> (XF_BITSHIFT(NPC)); j++) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = COLS / NPC
#pragma HLS PIPELINE
#pragma HLS loop_flatten off
            _dst1.write((i * (_cols >> (XF_BITSHIFT(NPC))) + j), dst.read());
            _dst2.write((i * (_cols >> (XF_BITSHIFT(NPC))) + j), dst1.read());
        }
    }
} // End of duplicateMat()
// ======================================================================================

// ======================================================================================
// Function to split xf::cv::Mat into 3 xf::cv::Mat
// ======================================================================================
template <int SRC_T, int ROWS, int COLS, int NPC>
void duplicateMat_3(xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _src,
                    xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _dst1,
                    xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _dst2,
                    xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _dst3) {
#pragma HLS inline off

#pragma HLS dataflow

    int _rows = _src.rows;
    int _cols = _src.cols;

    hls::stream<XF_TNAME(SRC_T, NPC)> src;
    hls::stream<XF_TNAME(SRC_T, NPC)> dst;
    hls::stream<XF_TNAME(SRC_T, NPC)> dst1;
    hls::stream<XF_TNAME(SRC_T, NPC)> dst2;

    for (int i = 0; i < _rows; i++) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = ROWS
        for (int j = 0; j<(_cols)>> (XF_BITSHIFT(NPC)); j++) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = COLS / NPC
#pragma HLS PIPELINE
#pragma HLS loop_flatten off
            src.write(_src.read(i * (_cols >> (XF_BITSHIFT(NPC))) + j));
        }
    }

    DuplicateStrm_3<SRC_T, ROWS, COLS, NPC>(src, dst, dst1, dst2, _rows, _cols);

    for (int i = 0; i < _rows; i++) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = ROWS
        for (int j = 0; j<(_cols)>> (XF_BITSHIFT(NPC)); j++) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = COLS / NPC
#pragma HLS PIPELINE
#pragma HLS loop_flatten off
            _dst1.write((i * (_cols >> (XF_BITSHIFT(NPC))) + j), dst.read());
            _dst2.write((i * (_cols >> (XF_BITSHIFT(NPC))) + j), dst1.read());
            _dst3.write((i * (_cols >> (XF_BITSHIFT(NPC))) + j), dst2.read());
        }
    }
} // End of duplicateMat_3()
// ======================================================================================

// ======================================================================================
// Function to split xf::cv::Mat into 4 xf::cv::Mat
// ======================================================================================
template <int SRC_T, int ROWS, int COLS, int NPC>
void duplicateMat_4(xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _src,
                    xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _dst1,
                    xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _dst2,
                    xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _dst3,
                    xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _dst4) {
#pragma HLS inline off

#pragma HLS dataflow

    int _rows = _src.rows;
    int _cols = _src.cols;

    hls::stream<XF_TNAME(SRC_T, NPC)> src;
    hls::stream<XF_TNAME(SRC_T, NPC)> dst;
    hls::stream<XF_TNAME(SRC_T, NPC)> dst1;
    hls::stream<XF_TNAME(SRC_T, NPC)> dst2;
    hls::stream<XF_TNAME(SRC_T, NPC)> dst3;

    for (int i = 0; i < _rows; i++) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = ROWS
        for (int j = 0; j<(_cols)>> (XF_BITSHIFT(NPC)); j++) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = COLS / NPC
#pragma HLS PIPELINE
#pragma HLS loop_flatten off
            src.write(_src.read(i * (_cols >> (XF_BITSHIFT(NPC))) + j));
        }
    }

    DuplicateStrm_4<SRC_T, ROWS, COLS, NPC>(src, dst, dst1, dst2, dst3, _rows, _cols);

    for (int i = 0; i < _rows; i++) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = ROWS
        for (int j = 0; j<(_cols)>> (XF_BITSHIFT(NPC)); j++) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = COLS / NPC
#pragma HLS PIPELINE
#pragma HLS loop_flatten off
            _dst1.write((i * (_cols >> (XF_BITSHIFT(NPC))) + j), dst.read());
            _dst2.write((i * (_cols >> (XF_BITSHIFT(NPC))) + j), dst1.read());
            _dst3.write((i * (_cols >> (XF_BITSHIFT(NPC))) + j), dst2.read());
            _dst4.write((i * (_cols >> (XF_BITSHIFT(NPC))) + j), dst3.read());
        }
    }
} // End of duplicateMat_4()
// ======================================================================================

// ======================================================================================
// Function to split xf::cv::Mat into 2 xf::cv::Mat
// ======================================================================================
template <int TYPE, int ROWS, int COLS, int NPPC>
void xFDuplicateMat(xf::cv::Mat<TYPE, ROWS, COLS, NPPC>& in_mat,
                    xf::cv::Mat<TYPE, ROWS, COLS, NPPC>& out_mat1,
                    xf::cv::Mat<TYPE, ROWS, COLS, NPPC>& out_mat2) {
#pragma HLS INLINE OFF

    const int c_TRIP_COUNT = ROWS * COLS;
    int loopcount = in_mat.rows * (in_mat.cols >> XF_BITSHIFT(NPPC));

    for (int i = 0; i < loopcount; i++) {
#pragma HLS LOOP_TRIPCOUNT min = c_TRIP_COUNT max = c_TRIP_COUNT
#pragma HLS pipeline II = 1
        XF_TNAME(TYPE, NPPC) tmp = in_mat.read(i);
        out_mat1.write(i, tmp);
        out_mat2.write(i, tmp);
    }

} // End of xFDuplicateMat()
// ======================================================================================

// ======================================================================================
// Function to split xf::cv::Mat into 3 xf::cv::Mat
// ======================================================================================
template <int TYPE, int ROWS, int COLS, int NPPC>
void xFDuplicateMat_3(xf::cv::Mat<TYPE, ROWS, COLS, NPPC>& in_mat,
                      xf::cv::Mat<TYPE, ROWS, COLS, NPPC>& out_mat1,
                      xf::cv::Mat<TYPE, ROWS, COLS, NPPC>& out_mat2,
                      xf::cv::Mat<TYPE, ROWS, COLS, NPPC>& out_mat3) {
#pragma HLS INLINE OFF

    const int c_TRIP_COUNT = ROWS * COLS;
    int loopcount = in_mat.rows * (in_mat.cols >> XF_BITSHIFT(NPPC));

    for (int i = 0; i < loopcount; i++) {
#pragma HLS LOOP_TRIPCOUNT min = c_TRIP_COUNT max = c_TRIP_COUNT
#pragma HLS pipeline II = 1
        XF_TNAME(TYPE, NPPC) tmp = in_mat.read(i);
        out_mat1.write(i, tmp);
        out_mat2.write(i, tmp);
        out_mat3.write(i, tmp);
    }

} // End of xFDuplicateMat_3()
// ======================================================================================

// ======================================================================================
// Function to split xf::cv::Mat into 4 xf::cv::Mat
// ======================================================================================
template <int TYPE, int ROWS, int COLS, int NPPC>
void xFDuplicateMat_4(xf::cv::Mat<TYPE, ROWS, COLS, NPPC>& in_mat,
                      xf::cv::Mat<TYPE, ROWS, COLS, NPPC>& out_mat1,
                      xf::cv::Mat<TYPE, ROWS, COLS, NPPC>& out_mat2,
                      xf::cv::Mat<TYPE, ROWS, COLS, NPPC>& out_mat3,
                      xf::cv::Mat<TYPE, ROWS, COLS, NPPC>& out_mat4) {
#pragma HLS INLINE OFF

    const int c_TRIP_COUNT = ROWS * COLS;
    int loopcount = in_mat.rows * (in_mat.cols >> XF_BITSHIFT(NPPC));

    for (int i = 0; i < loopcount; i++) {
#pragma HLS LOOP_TRIPCOUNT min = c_TRIP_COUNT max = c_TRIP_COUNT
#pragma HLS pipeline II = 1
        XF_TNAME(TYPE, NPPC) tmp = in_mat.read(i);
        out_mat1.write(i, tmp);
        out_mat2.write(i, tmp);
        out_mat3.write(i, tmp);
        out_mat4.write(i, tmp);
    }

} // End of xFDuplicateMat_4()
// ======================================================================================

// ======================================================================================
// Function to set border in the extracted kernel sized block
// ======================================================================================
template <int K_ROWS, int K_COLS, typename SRC_T, int BORDER_T>
void xFSetBorder(xf::cv::Window<K_ROWS, K_COLS, SRC_T>& src_blk,
                 uint16_t _row,
                 uint16_t _col,
                 uint16_t _src_rows,
                 uint16_t _src_cols) {
#pragma HLS INLINE OFF

    uint16_t blk_t_idx, blk_b_idx;
    uint16_t blk_l_idx, blk_r_idx;

    blk_t_idx = (K_ROWS - _row - 1);
    blk_b_idx = (K_ROWS - (_row - _src_rows + 1) - 1);

    blk_l_idx = (K_COLS - _col - 1);
    blk_r_idx = (K_COLS - (_col - _src_cols + 1) - 1);

    for (uint16_t r = 0; r < K_ROWS; r++) {
#pragma HLS UNROLL
        for (uint16_t c = 0; c < K_COLS; c++) {
#pragma HLS UNROLL

            bool top_border = ((r < blk_t_idx) && (_row < K_ROWS - 1)) ? true : false;
            bool bottom_border = ((r > blk_b_idx) && (_row >= _src_rows)) ? true : false;
            bool left_border = ((c < blk_l_idx) && (_col < K_COLS - 1)) ? true : false;
            bool right_border = ((c > blk_r_idx) && (_col >= _src_cols)) ? true : false;

            uint16_t r_idx = r, c_idx = c;

            if (BORDER_T == XF_BORDER_REPLICATE) {
                r_idx = top_border ? blk_t_idx : bottom_border ? blk_b_idx : r;

            } else if (BORDER_T == XF_BORDER_CONSTANT) {
                r_idx = top_border ? (2 * blk_t_idx - r) : bottom_border ? (2 * blk_b_idx - r) : r;

            } else if (BORDER_T == XF_BORDER_CONSTANT) {
                r_idx = top_border ? (2 * blk_t_idx - r - 1) : bottom_border ? (2 * blk_b_idx - r + 1) : r;

            } else { // TODO: Need to add other modes support
                r_idx = r;
            }

            if (BORDER_T == XF_BORDER_REPLICATE) {
                c_idx = left_border ? blk_l_idx : right_border ? blk_r_idx : c;

            } else if (BORDER_T == XF_BORDER_CONSTANT) {
                c_idx = left_border ? (2 * blk_l_idx - c) : right_border ? (2 * blk_r_idx - c) : c;

            } else if (BORDER_T == XF_BORDER_CONSTANT) {
                c_idx = left_border ? (2 * blk_l_idx - c - 1) : right_border ? (2 * blk_r_idx - c + 1) : c;

            } else { // TODO: Need to add other modes support
                c_idx = c;
            }

            if ((top_border | bottom_border | left_border | right_border) && (BORDER_T == XF_BORDER_CONSTANT)) {
                src_blk.val[r][c] = 0;
            } else {
                src_blk.val[r][c] = src_blk.val[r_idx][c_idx];
            }
        }
    }

} // End of xFSetBorder()
  // ======================================================================================

} // namespace cv
} // namespace xf

#endif //__XF_EXTRA_UTILITY_H__
