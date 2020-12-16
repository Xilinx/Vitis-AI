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

#ifndef _XF_PYR_DOWN_GAUSSIAN_DOWN_
#define _XF_PYR_DOWN_GAUSSIAN_DOWN_

#include "ap_int.h"
#include "hls_stream.h"
#include "common/xf_common.hpp"

template <int NPC, int DEPTH, int WIN_SZ, int WIN_SZ_SQ, int PLANES>
void xFPyrDownApplykernel(XF_PTUNAME(DEPTH) OutputValues[XF_NPIXPERCYCLE(NPC)],
                          XF_PTUNAME(DEPTH) src_buf[WIN_SZ][XF_NPIXPERCYCLE(NPC) + (WIN_SZ - 1)],
                          ap_uint<8> win_size) {
// clang-format off
    #pragma HLS INLINE
    // clang-format on
    ap_uint<32> array[WIN_SZ_SQ];
// clang-format off
    #pragma HLS ARRAY_PARTITION variable=array complete dim=1
    // clang-format on

    int array_ptr = 0;

Compute_Grad_Loop:
    for (int copy_arr = 0; copy_arr < WIN_SZ; copy_arr++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=1 max=WIN_SZ
        #pragma HLS UNROLL
        // clang-format on
        for (int copy_in = 0; copy_in < WIN_SZ; copy_in++) {
// clang-format off
            #pragma HLS LOOP_TRIPCOUNT min=1 max=WIN_SZ
            #pragma HLS UNROLL
            // clang-format on
            array[array_ptr] = src_buf[copy_arr][copy_in];
            array_ptr++;
        }
    }
    ap_uint<32> out_pixel = 0;
    int k[25] = {1, 4, 6, 4, 1, 4, 16, 24, 16, 4, 6, 24, 36, 24, 6, 4, 16, 24, 16, 4, 1, 4, 6, 4, 1};
    for (int i = 0, k = 0; i < PLANES; i++, k += 8) {
// clang-format off
        #pragma HLS PIPELINE II=1
        // clang-format on
        out_pixel = array[0 * 5 + 0].range(k + 7, k) + array[0 * 5 + 4].range(k + 7, k) +
                    array[4 * 5 + 0].range(k + 7, k) + array[4 * 5 + 4].range(k + 7, k);
        out_pixel += (array[0 * 5 + 1].range(k + 7, k) + array[0 * 5 + 3].range(k + 7, k) +
                      array[1 * 5 + 0].range(k + 7, k) + array[1 * 5 + 4].range(k + 7, k))
                     << 2;
        out_pixel += (array[4 * 5 + 1].range(k + 7, k) + array[4 * 5 + 3].range(k + 7, k) +
                      array[3 * 5 + 0].range(k + 7, k) + array[3 * 5 + 4].range(k + 7, k))
                     << 2;
        out_pixel += (array[0 * 5 + 2].range(k + 7, k) + array[2 * 5 + 0].range(k + 7, k) +
                      array[2 * 5 + 4].range(k + 7, k) + array[4 * 5 + 2].range(k + 7, k))
                     << 2;
        out_pixel += (array[0 * 5 + 2].range(k + 7, k) + array[2 * 5 + 0].range(k + 7, k) +
                      array[2 * 5 + 4].range(k + 7, k) + array[4 * 5 + 2].range(k + 7, k))
                     << 1;
        out_pixel += (array[1 * 5 + 1].range(k + 7, k) + array[1 * 5 + 3].range(k + 7, k) +
                      array[3 * 5 + 1].range(k + 7, k) + array[3 * 5 + 3].range(k + 7, k))
                     << 4;
        out_pixel += (array[1 * 5 + 2].range(k + 7, k) + array[2 * 5 + 1].range(k + 7, k) +
                      array[2 * 5 + 3].range(k + 7, k) + array[3 * 5 + 2].range(k + 7, k))
                     << 4;
        out_pixel += (array[1 * 5 + 2].range(k + 7, k) + array[2 * 5 + 1].range(k + 7, k) +
                      array[2 * 5 + 3].range(k + 7, k) + array[3 * 5 + 2].range(k + 7, k))
                     << 3;
        out_pixel += (array[2 * 5 + 2].range(k + 7, k)) << 5;
        out_pixel += (array[2 * 5 + 2].range(k + 7, k)) << 2;

        OutputValues[0].range(k + 7, k) = (unsigned char)((out_pixel + 128) >> 8);
    }
    return;
}

template <int ROWS, int COLS, int DEPTH, int NPC, int WORDWIDTH, int TC, int WIN_SZ, int WIN_SZ_SQ, int PLANES>
void xFPyrDownprocessgaussian(hls::stream<XF_TNAME(DEPTH, NPC)>& _src_mat,
                              hls::stream<XF_TNAME(DEPTH, NPC)>& _out_mat,
                              XF_TNAME(DEPTH, NPC) buf[WIN_SZ][(COLS >> XF_BITSHIFT(NPC)) + (WIN_SZ >> 1)],
                              XF_PTUNAME(DEPTH) src_buf[WIN_SZ][XF_NPIXPERCYCLE(NPC) + (WIN_SZ - 1)],
                              XF_PTUNAME(DEPTH) OutputValues[XF_NPIXPERCYCLE(NPC)],
                              XF_PTUNAME(DEPTH) & P0,
                              uint16_t img_width,
                              uint16_t img_height,
                              uint16_t& shift_x,
                              ap_uint<13> row_ind[WIN_SZ],
                              ap_uint<13> row,
                              ap_uint<8> win_size) {
// clang-format off
    #pragma HLS INLINE
    // clang-format on

    XF_TNAME(DEPTH, NPC) buf_cop[WIN_SZ];
// clang-format off
    #pragma HLS ARRAY_PARTITION variable=buf_cop complete dim=1
    // clang-format on

    uint16_t npc = XF_NPIXPERCYCLE(NPC);
Col_Loop:
    for (ap_uint<13> col = 0; col < img_width + (WIN_SZ >> 1); col++) {
// clang-format off
        #pragma HLS LOOP_FLATTEN OFF
        #pragma HLS LOOP_TRIPCOUNT min=1 max=TC
        #pragma HLS pipeline
        // clang-format on
        if (row < img_height && col < img_width)
            buf[row_ind[win_size - 1]][col] = _src_mat.read(); // Read data
        else
            buf[row_ind[win_size - 1]][col] = 0;

        for (int copy_buf_var = 0; copy_buf_var < WIN_SZ; copy_buf_var++) {
// clang-format off
            #pragma HLS LOOP_TRIPCOUNT min=1 max=WIN_SZ
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
            #pragma HLS LOOP_TRIPCOUNT min=1 max=WIN_SZ
            #pragma HLS UNROLL
            // clang-format on
            if (col < img_width) {
                src_buf[extract_px][win_size - 1] = buf_cop[extract_px];
            } else {
                src_buf[extract_px][win_size - 1] = src_buf[extract_px][win_size - 2];
            }
        }

        xFPyrDownApplykernel<NPC, DEPTH, WIN_SZ, WIN_SZ_SQ, PLANES>(OutputValues, src_buf, win_size);
        if (col >= (win_size >> 1)) {
            _out_mat.write(OutputValues[0]);
        }

        for (int wrap_buf = 0; wrap_buf < WIN_SZ; wrap_buf++) {
// clang-format off
            #pragma HLS UNROLL
            #pragma HLS LOOP_TRIPCOUNT min=1 max=WIN_SZ
            // clang-format on
            for (int col_warp = 0; col_warp < WIN_SZ - 1; col_warp++) {
// clang-format off
                #pragma HLS UNROLL
                #pragma HLS LOOP_TRIPCOUNT min=1 max=WIN_SZ
                // clang-format on
                if (col == 0) {
                    src_buf[wrap_buf][col_warp] = src_buf[wrap_buf][win_size - 1];
                } else {
                    src_buf[wrap_buf][col_warp] = src_buf[wrap_buf][col_warp + 1];
                }
            }
        }
    } // Col_Loop
}

template <int ROWS, int COLS, int DEPTH, int NPC, int WORDWIDTH, int TC, int WIN_SZ, int WIN_SZ_SQ, int PLANES>
void xf_pyrdown_gaussian_nxn(hls::stream<XF_TNAME(DEPTH, NPC)>& _src_mat,
                             hls::stream<XF_TNAME(DEPTH, NPC)>& _out_mat,
                             ap_uint<8> win_size,
                             uint16_t img_height,
                             uint16_t img_width) {
    ap_uint<13> row_ind[WIN_SZ];
// clang-format off
    #pragma HLS ARRAY_PARTITION variable=row_ind complete dim=1
    // clang-format on

    ap_uint<8> buf_size = XF_NPIXPERCYCLE(NPC) + (WIN_SZ - 1);
    uint16_t shift_x = 0;
    ap_uint<13> row, col;

    XF_TNAME(DEPTH, NPC) OutputValues[XF_NPIXPERCYCLE(NPC)];
// clang-format off
    #pragma HLS ARRAY_PARTITION variable=OutputValues complete dim=1
    // clang-format on

    XF_PTUNAME(DEPTH) src_buf[WIN_SZ][XF_NPIXPERCYCLE(NPC) + (WIN_SZ - 1)];
// clang-format off
    #pragma HLS ARRAY_PARTITION variable=src_buf complete dim=1
    #pragma HLS ARRAY_PARTITION variable=src_buf complete dim=2
    // clang-format on
    // src_buf1 et al merged
    XF_TNAME(DEPTH, NPC) P0;

    XF_TNAME(DEPTH, NPC) buf[WIN_SZ][(COLS >> XF_BITSHIFT(NPC)) + (WIN_SZ >> 1)];
// clang-format off
    #pragma HLS ARRAY_PARTITION variable=buf complete dim=1
    #pragma HLS RESOURCE variable=buf core=RAM_S2P_BRAM
    // clang-format on

    // initializing row index

    for (int init_row_ind = 0; init_row_ind < win_size; init_row_ind++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=1 max=WIN_SZ
        // clang-format on
        row_ind[init_row_ind] = init_row_ind;
    }

read_lines:
    for (int init_buf = row_ind[win_size >> 1]; init_buf < row_ind[win_size - 1]; init_buf++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=1 max=WIN_SZ
        // clang-format on
        for (col = 0; col < img_width; col++) {
// clang-format off
            #pragma HLS LOOP_TRIPCOUNT min=1 max=TC
            #pragma HLS LOOP_FLATTEN OFF
            #pragma HLS pipeline
            // clang-format on
            buf[init_buf][col] = _src_mat.read();
        }
    }

    // takes care of top borders
    for (col = 0; col < img_width; col++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=1 max=TC
        // clang-format on
        for (int init_buf = 0; init_buf<WIN_SZ>> 1; init_buf++) {
// clang-format off
            #pragma HLS LOOP_TRIPCOUNT min=1 max=WIN_SZ
            #pragma HLS UNROLL
            // clang-format on
            buf[init_buf][col] = buf[row_ind[win_size >> 1]][col];
        }
    }

Row_Loop:
    for (row = (win_size >> 1); row < img_height + (win_size >> 1); row++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=1 max=ROWS
        // clang-format on
        P0 = 0;
        xFPyrDownprocessgaussian<ROWS, COLS, DEPTH, NPC, WORDWIDTH, TC, WIN_SZ, WIN_SZ_SQ, PLANES>(
            _src_mat, _out_mat, buf, src_buf, OutputValues, P0, img_width, img_height, shift_x, row_ind, row, win_size);

        // update indices
        ap_uint<13> zero_ind = row_ind[0];
        for (int init_row_ind = 0; init_row_ind < WIN_SZ - 1; init_row_ind++) {
// clang-format off
            #pragma HLS LOOP_TRIPCOUNT min=1 max=WIN_SZ
            #pragma HLS UNROLL
            // clang-format on
            row_ind[init_row_ind] = row_ind[init_row_ind + 1];
        }
        row_ind[win_size - 1] = zero_ind;

    } // Row_Loop
}

template <int ROWS,
          int COLS,
          int DEPTH,
          int NPC,
          int WORDWIDTH,
          int PIPELINEFLAG,
          int WIN_SZ,
          int WIN_SZ_SQ,
          int PLANES>
void xFPyrDownGaussianBlur(hls::stream<XF_TNAME(DEPTH, NPC)>& _src,
                           hls::stream<XF_TNAME(DEPTH, NPC)>& _dst,
                           ap_uint<8> win_size,
                           int _border_type,
                           uint16_t imgheight,
                           uint16_t imgwidth) {
#ifndef __SYNTHESIS__
    assert(((imgheight <= ROWS) && (imgwidth <= COLS)) && "ROWS and COLS should be greater than input image");

    assert((win_size <= WIN_SZ) && "win_size must not be greater than WIN_SZ");
#endif
    imgwidth = imgwidth >> XF_BITSHIFT(NPC);

    xf_pyrdown_gaussian_nxn<ROWS, COLS, DEPTH, NPC, WORDWIDTH, (COLS >> XF_BITSHIFT(NPC)) + (WIN_SZ >> 1), WIN_SZ,
                            WIN_SZ_SQ, PLANES>(_src, _dst, WIN_SZ, imgheight, imgwidth);
}

#endif
