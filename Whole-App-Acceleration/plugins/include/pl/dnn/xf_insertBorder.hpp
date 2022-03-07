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

#ifndef _XF_INSERTBORDER_
#define _XF_INSERTBORDER_

#include "hls_stream.h"
#include "ap_int.h"
#include "common/xf_common.hpp"
#include "common/xf_utility.hpp"

/**
 * @file xf_insertBorder.hpp
 * This file is part of Vitis Vision Library.
 */

namespace xf {
namespace cv {

/**
 * @tparam TYPE input and ouput type
 * @tparam SRC_ROWS rows of the input image
 * @tparam SRC_COLS cols of the input image
 * @tparam DST_ROWS rows of the output image
 * @tparam DST_COLS cols of the output image
 * @tparam NPC number of pixels processed per cycle
 * @param _src input image
 * @param _dst output image
 * @param insert_pad_val insert pad value
 */

template <int TYPE, int SRC_ROWS, int SRC_COLS, int DST_ROWS, int DST_COLS, int NPC>
void insertBorder(xf::cv::Mat<TYPE, SRC_ROWS, SRC_COLS, NPC>& _src,
                  xf::cv::Mat<TYPE, DST_ROWS, DST_COLS, NPC>& _dst,
                  int insert_pad_val) {
// clang-format off
#pragma HLS INLINE OFF
    // clang-format on

    enum { DEPTH = TYPE, PLANES = XF_CHANNELS(TYPE, NPC) };

    unsigned short in_height = _src.rows;
    unsigned short in_width = _src.cols;
    unsigned short out_height = _dst.rows;
    unsigned short out_width = _dst.cols;

    unsigned short dx = (out_width - in_width) >> 1;
    unsigned short dy = (out_height - in_height) >> 1;

    unsigned short imgInput_ncpr = (in_width + (NPC - 1)) >> XF_BITSHIFT(NPC);
    unsigned short imgInput_width_align_npc = imgInput_ncpr << XF_BITSHIFT(NPC);
    unsigned short imgOutput_ncpr = (out_width + (NPC - 1)) >> XF_BITSHIFT(NPC);
    unsigned short imgOutput_width_align_npc = imgOutput_ncpr << XF_BITSHIFT(NPC);

    unsigned short padded_in_col_due_2_xfmat = imgInput_width_align_npc - in_width;
    unsigned short non_padded_in_col_due_2_xfmat = NPC - padded_in_col_due_2_xfmat;

    unsigned short row_loop_bound;
    bool row_pad_enable;
    if (out_height > in_height) {
        row_loop_bound = out_height;
        row_pad_enable = 1;
    } else {
        row_loop_bound = in_height;
        row_pad_enable = 0;
    }

    unsigned short col_loop_bound;
    bool col_pad_enable;
    if (imgOutput_ncpr > imgInput_ncpr) {
        col_loop_bound = imgOutput_ncpr;
        col_pad_enable = 1;
    } else {
        col_loop_bound = imgInput_ncpr;
        col_pad_enable = 0;
    }

    //##DDR index
    uint32_t read_index = 0;
    ap_uint<32> write_index = 0;
    ap_uint<16> write_col_index = 0;

    short pad_col_count_left = dx;
    short pad_row_count_top = dy;
    unsigned short input_col_index = 0;
    unsigned short input_col_index_prev = 0;
    unsigned short input_row_index = 0;

    XF_TNAME(DEPTH, NPC) read_word_init;
    for (int pixel = 0, bit1 = 0; pixel < NPC; pixel++, bit1 += (PLANES * 8)) {
// clang-format off
#pragma HLS unroll
        // clang-format on
        for (int channel = 0, bit2 = 0; channel < PLANES; channel++, bit2 += 8) {
// clang-format off
#pragma HLS unroll
            // clang-format on
            read_word_init.range(bit1 + (bit2 + 7), bit1 + bit2) = insert_pad_val;
        }
    }

    XF_TNAME(DEPTH, NPC) read_word_tmp = 0;
    XF_TNAME(DEPTH, NPC) read_word = 0;
    XF_TNAME(DEPTH, NPC) read_word_prev = read_word_init;
    XF_TNAME(DEPTH, NPC) write_word = 0;

    short col_index = 0;
    short row_index = 0;

LOOP_ROW_COL:
    for (int row_col_index = 0; row_col_index < row_loop_bound * col_loop_bound; row_col_index++) {
// clang-format off
#pragma HLS LOOP_TRIPCOUNT min=1 max=SRC_COLS
#pragma HLS pipeline II=1
        // clang-format on
        bool input_read_en = pad_col_count_left < NPC && input_col_index < imgInput_ncpr && pad_row_count_top <= 0 &&
                             (row_index - dy) < in_height;
        if (input_read_en == 1) {
            read_word_tmp = _src.read(read_index++);
        } else
            read_word_tmp = read_word_init;

        for (int pixel = 0, bit1 = 0; pixel < NPC; pixel++, bit1 += (PLANES * 8)) {
// clang-format off
#pragma HLS unroll
            // clang-format on
            for (int channel = 0, bit2 = 0; channel < PLANES; channel++, bit2 += 8) {
// clang-format off
#pragma HLS unroll
                // clang-format on

                if ((pixel >= non_padded_in_col_due_2_xfmat) && (input_col_index == (imgInput_ncpr - 1)))
                    read_word.range(bit1 + (bit2 + 7), bit1 + bit2) = insert_pad_val;
                else
                    read_word.range(bit1 + (bit2 + 7), bit1 + bit2) =
                        read_word_tmp.range(bit1 + (bit2 + 7), bit1 + bit2);
            }
        }

        short prev_valid_elements = pad_col_count_left;
        short current_valid_elements = NPC - pad_col_count_left;

        ap_uint<8> read_word_extract[NPC][PLANES];
        ap_uint<8> read_word_extract_prev[NPC][PLANES];
// clang-format off
#pragma HLS array_partition variable=read_word_extract dim=0
#pragma HLS array_partition variable=read_word_extract_prev dim=0
        // clang-format on
        for (int pixel = 0, bit1 = 0; pixel < NPC; pixel++, bit1 += (PLANES * 8)) {
// clang-format off
#pragma HLS unroll
            // clang-format on
            for (int channel = 0, bit2 = 0; channel < PLANES; channel++, bit2 += 8) {
// clang-format off
#pragma HLS unroll
                // clang-format on
                read_word_extract[pixel][channel] = read_word.range(bit1 + (bit2 + 7), bit1 + bit2);
                read_word_extract_prev[pixel][channel] = read_word_prev.range(bit1 + (bit2 + 7), bit1 + bit2);
            }
        }

        bool out_write_en;
        if (current_valid_elements < NPC && input_col_index_prev == (imgInput_ncpr - 1))
            out_write_en = pad_col_count_left < NPC && input_col_index < (imgInput_ncpr + 1) &&
                           pad_row_count_top <= 0 && (row_index - dy) < in_height;
        else
            out_write_en = input_read_en;

        ap_uint<8> DDR_write_data[NPC][PLANES];
// clang-format off
#pragma HLS array_partition variable=DDR_write_data dim=0
        // clang-format on
        for (int pixel = 0; pixel < NPC; pixel++) {
            for (int channel = 0; channel < PLANES; channel++) {
                if (out_write_en == 1) {
                    if ((pixel + current_valid_elements) < NPC)
                        DDR_write_data[pixel][channel] =
                            read_word_extract_prev[pixel + current_valid_elements][channel];
                    else
                        DDR_write_data[pixel][channel] = read_word_extract[pixel - prev_valid_elements][channel];
                } else {
                    DDR_write_data[pixel][channel] = insert_pad_val;
                }
            }
        }

        XF_TNAME(DEPTH, NPC) out_pix;
        ap_uint<PLANES * 8> plane_tmp;
        for (int pixel = 0, bit1 = 0; pixel < NPC; pixel++, bit1 += (PLANES * 8)) {
// clang-format off
#pragma HLS unroll
            // clang-format on
            for (int channel = 0, bit2 = 0; channel < PLANES; channel++, bit2 += 8) {
// clang-format off
#pragma HLS unroll
                // clang-format on
                plane_tmp.range(bit2 + 7, bit2) = DDR_write_data[pixel][channel];
            }
            out_pix.range(bit1 + (PLANES * 8) - 1, bit1) = plane_tmp;
        }
        _dst.write(write_index++, out_pix);

        // Last

        if (pad_col_count_left >= NPC)
            pad_col_count_left -= NPC;
        else if (col_index == (col_loop_bound - 1)) {
            pad_col_count_left = dx;
            pad_row_count_top--;
        }
        input_col_index_prev = input_col_index;
        if (col_index == (col_loop_bound - 1)) {
            input_col_index = 0;
            read_word_prev = read_word_init;
        } else if (input_read_en == 1) {
            input_col_index++;
            read_word_prev = read_word;
        }

        if (col_index < (col_loop_bound - 1)) {
            col_index++;
        } else {
            col_index = 0;
            row_index++;
        }

    } // LOOP_COL_row
}

} // namespace cv
} // namespace xf

#endif
