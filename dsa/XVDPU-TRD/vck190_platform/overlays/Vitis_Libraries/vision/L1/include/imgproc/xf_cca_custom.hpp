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

#ifndef __XF_CCA_CUSTOM_HPP__
#define __XF_CCA_CUSTOM_HPP__

#include "ap_int.h"
#include "common/xf_common.hpp"
#include "common/xf_utility.hpp"

namespace xf {
namespace cv {

template <int WIDTH>
void process_row(uint8_t* in_ptr, uint8_t* tmp_out_ptr, bool* lab_arr, int& obj_pix, int width) {
// clang-format off
#pragma HLS INLINE
    // clang-format on

    bool a, b, c, d;
    a = 1;
    b = lab_arr[0];
    c = lab_arr[1];
    d = 1;

PROC_ROW_LOOP:
    for (int j = 0; j < width; j++) {
// clang-format off
#pragma HLS LOOP_TRIPCOUNT min=1 max=WIDTH
#pragma HLS PIPELINE II=1
        // clang-format on
        bool lab;
        unsigned char out;

        if (in_ptr[j] != 0) {
            if ((a || b || c || d) == 1) {
                lab = 1;
                out = 0;
            } else {
                lab = 0;
                out = 255;
            }
        } else {
            lab = 0;
            out = 0;
            obj_pix++;
        }

        lab_arr[j] = lab;
        tmp_out_ptr[j] = out;
        a = b;
        b = c;
        c = lab_arr[j + 2];
        d = lab;
    }
}

template <int HEIGHT, int WIDTH>
void fw_cca(uint8_t* in_ptr, uint8_t* tmp_out_ptr, int& obj_pix, int height, int width) {
// clang-format off
#pragma HLS INLINE OFF
    // clang-format on

    obj_pix = 0;
    int offset = 0;
    bool lab_arr[WIDTH + 2];
    for (int i = 0; i < width + 2; i++) {
// clang-format off
#pragma HLS LOOP_TRIPCOUNT min=1 max=WIDTH
#pragma HLS PIPELINE II=1
        // clang-format on
        lab_arr[i] = 1;
    }

    for (int i = 0; i < height; i++) {
// clang-format off
#pragma HLS LOOP_TRIPCOUNT min=1 max=HEIGHT
        // clang-format on
        process_row<WIDTH>(in_ptr + offset, tmp_out_ptr + offset, lab_arr, obj_pix, width);
        offset += width;
    }
}

template <int WIDTH>
void read_row_to_ram(uint8_t* _fw_pass, uint8_t* ram, int width) {
// clang-format off
#pragma HLS INLINE OFF
    // clang-format on

    for (int j = 0; j < width; j++) {
// clang-format off
#pragma HLS LOOP_TRIPCOUNT min=1 max=WIDTH
#pragma HLS PIPELINE II=1
        // clang-format on
        ram[j] = _fw_pass[j];
    }
}

template <int WIDTH>
void write_row_to_mem(uint8_t* ram, uint8_t* _dst, int width) {
// clang-format off
#pragma HLS INLINE OFF
    // clang-format on

    for (int j = 0; j < width; j++) {
// clang-format off
#pragma HLS LOOP_TRIPCOUNT min=1 max=WIDTH		
#pragma HLS PIPELINE II=1
        // clang-format on
        _dst[j] = ram[j];
    }
}

template <int HEIGHT, int WIDTH>
void rev_cca(uint8_t* in_ptr, uint8_t* tmp_out_ptr, int height, int width) {
// clang-format off
#pragma HLS INLINE OFF
    // clang-format on

    bool lab_arr[WIDTH + 2], flag = 0;
    int obj_pix;

    for (int i = 0; i < width + 2; i++) {
// clang-format off
#pragma HLS LOOP_TRIPCOUNT min=1 max=WIDTH
#pragma HLS PIPELINE II=1
        // clang-format on
        lab_arr[i] = 1;
    }

    uint8_t rd_linebuff1[WIDTH], rd_linebuff2[WIDTH];
    uint8_t wrt_linebuff1[WIDTH], wrt_linebuff2[WIDTH];

    int rd_offset = (height * width);
    int wrt_offset = (height * width);

    rd_offset -= width;
    read_row_to_ram<WIDTH>(in_ptr + rd_offset, rd_linebuff1, width);

    rd_offset -= width;
    read_row_to_ram<WIDTH>(in_ptr + rd_offset, rd_linebuff2, width);
    process_row<WIDTH>(rd_linebuff1, wrt_linebuff1, lab_arr, obj_pix, width);

REV_ROW_LOOP:
    for (int i = 0; i < height - 2; i++) {
// clang-format off
#pragma HLS LOOP_TRIPCOUNT min=1 max=HEIGHT
        // clang-format on
        rd_offset -= width;
        wrt_offset -= width;

        if (flag == 0) {
            read_row_to_ram<WIDTH>(in_ptr + rd_offset, rd_linebuff1, width);
            process_row<WIDTH>(rd_linebuff2, wrt_linebuff2, lab_arr, obj_pix, width);
            write_row_to_mem<WIDTH>(wrt_linebuff1, tmp_out_ptr + wrt_offset, width);
            flag = 1;
        } else {
            read_row_to_ram<WIDTH>(in_ptr + rd_offset, rd_linebuff2, width);
            process_row<WIDTH>(rd_linebuff1, wrt_linebuff1, lab_arr, obj_pix, width);
            write_row_to_mem<WIDTH>(wrt_linebuff2, tmp_out_ptr + wrt_offset, width);
            flag = 0;
        }
    }

    wrt_offset -= width;

    if (flag == 0) {
        process_row<WIDTH>(rd_linebuff2, wrt_linebuff2, lab_arr, obj_pix, width);
        write_row_to_mem<WIDTH>(wrt_linebuff1, tmp_out_ptr + wrt_offset, width);
        flag = 1;
    } else {
        process_row<WIDTH>(rd_linebuff1, wrt_linebuff1, lab_arr, obj_pix, width);
        write_row_to_mem<WIDTH>(wrt_linebuff2, tmp_out_ptr + wrt_offset, width);
        flag = 0;
    }

    wrt_offset -= width;

    if (flag == 0)
        write_row_to_mem<WIDTH>(wrt_linebuff1, tmp_out_ptr + wrt_offset, width);
    else
        write_row_to_mem<WIDTH>(wrt_linebuff2, tmp_out_ptr + wrt_offset, width);
}

template <int HEIGHT, int WIDTH>
void pass_1(uint8_t* in_ptr1,
            uint8_t* in_ptr2,
            uint8_t* tmp_out_ptr1,
            uint8_t* tmp_out_ptr2,
            int& obj_pix,
            int height,
            int width) {
// clang-format off
#pragma HLS INLINE OFF
// clang-format on							

	fw_cca<HEIGHT,WIDTH>(in_ptr1,tmp_out_ptr1,obj_pix,height,width);
	rev_cca<HEIGHT,WIDTH>(in_ptr2,tmp_out_ptr2,height,width);
}

template <int HEIGHT, int WIDTH>
void pass_2 (uint8_t* tmp_out_ptr1, uint8_t* tmp_out_ptr2, uint8_t* out_ptr, int& def_pix, int height, int width) {
// clang-format off
#pragma HLS INLINE OFF
    // clang-format on

    int idx = 0;
    def_pix = 0;
    for (int i = 0; i < height; i++) {
// clang-format off
#pragma HLS LOOP_TRIPCOUNT min=1 max=HEIGHT
#pragma HLS LOOP_FLATTEN
        // clang-format on
        for (int j = 0; j < width; j++) {
// clang-format off
#pragma HLS LOOP_TRIPCOUNT min=1 max=WIDTH
#pragma HLS PIPELINE II=1
            // clang-format on
            uint8_t tmp = tmp_out_ptr1[idx] & tmp_out_ptr2[idx];
            out_ptr[idx++] = tmp;
            if (tmp != 0) def_pix++;
        }
    }
}

template <int HEIGHT, int WIDTH>
void ccaCustom(uint8_t* in_ptr1,      // input image pointer for forward pass
               uint8_t* in_ptr2,      // input image pinter for the parallel
                                      // computation of reverse pass
               uint8_t* tmp_out_ptr1, // pointer to store and read from the
                                      // temporary buffer in DDR for the forward
                                      // pass
               uint8_t* tmp_out_ptr2, // pointer to store and read from the
                                      // temporary buffer in DDR for the reverse
                                      // pass
               uint8_t* out_ptr,      // output defects image
               int& obj_pix,          // output - object pixels without defects
               int& def_pix,          // output - defect pixels
               int height,
               int width) {
// clang-format off
#pragma HLS INLINE OFF
    // clang-format on

    for (int i = 0; i < 2; i++) {
        if (i == 0)
            pass_1<HEIGHT, WIDTH>(in_ptr1, in_ptr2, tmp_out_ptr1, tmp_out_ptr2, obj_pix, height, width);
        else
            pass_2<HEIGHT, WIDTH>(tmp_out_ptr1, tmp_out_ptr2, out_ptr, def_pix, height, width);
    }

    return;
}
// ======================================================================================

} // end of cv
} // end of xf

#endif // end of __XF_CCA_CUSTOM_HPP__
