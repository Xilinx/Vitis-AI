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

#ifndef _XF_MEDIAN_BLUR_
#define _XF_MEDIAN_BLUR_

#include "ap_int.h"
#include "hls_stream.h"
#include "common/xf_common.hpp"
#include "common/xf_utility.hpp"

namespace xf {
namespace cv {

template <int PLANES, int NPC, int DEPTH, int WIN_SZ, int WIN_SZ_SQ>
void xFMedianProc(XF_PTUNAME(DEPTH) OutputValues[XF_NPIXPERCYCLE(NPC)],
                  XF_PTUNAME(DEPTH) src_buf[WIN_SZ][XF_NPIXPERCYCLE(NPC) + (WIN_SZ - 1)],
                  ap_uint<8> win_size) {
// clang-format off
    #pragma HLS INLINE off
    // clang-format on
    XF_PTUNAME(DEPTH) out_val;
    XF_PTUNAME(DEPTH) array[WIN_SZ_SQ];
    XF_PTUNAME(DEPTH) array_channel[WIN_SZ_SQ];
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
    for (int channel = 0, k = 0; channel < PLANES; channel++, k += 8) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=1 max=PLANES
        #pragma HLS UNROLL
        // clang-format on
        for (int p = 0; p < WIN_SZ_SQ; p++) {
// clang-format off
            #pragma HLS LOOP_TRIPCOUNT min=1 max=WIN_SZ_SQ
            #pragma HLS UNROLL
            // clang-format on
            array_channel[p] = array[p].range(k + 7, k);
        }

    xFApplyMaskLoop:
        for (int16_t j = 0; j <= WIN_SZ_SQ - 1; j++) {
// clang-format off
            #pragma HLS LOOP_TRIPCOUNT min=1 max=WIN_SZ
            #pragma HLS LOOP_FLATTEN off
            // clang-format on
            int16_t tmp = j & 0x0001;
            if (tmp == 0) {
            xFSortLoop1:
                for (int i = 0; i <= ((WIN_SZ_SQ >> 1) - 1); i++) // even sort
                {
// clang-format off
                    #pragma HLS LOOP_TRIPCOUNT min=1 max=WIN_SZ
                    #pragma HLS unroll
                    // clang-format on
                    int c = (i * 2);
                    int c1 = (c + 1);

                    if (array_channel[c] < array_channel[c1]) {
                        XF_PTUNAME(DEPTH) temp = array_channel[c];
                        array_channel[c] = array_channel[c1];
                        array_channel[c1] = temp;
                    }
                }
            } else {
            xFSortLoop2:
                for (int i = 0; i <= ((WIN_SZ_SQ >> 1) - 1); i++) // odd sort WINDOW_SIZE_H>>1 -1
                {
// clang-format off
                    #pragma HLS LOOP_TRIPCOUNT min=1 max=WIN_SZ
                    #pragma HLS unroll
                    // clang-format on
                    int c = (i * 2);
                    int c1 = (c + 1);
                    int c2 = (c + 2);
                    if (array_channel[c1] < array_channel[c2]) {
                        XF_PTUNAME(DEPTH) temp = array_channel[c1];
                        array_channel[c1] = array_channel[c2];
                        array_channel[c2] = temp;
                    }
                }
            }
        }

        out_val.range(k + 7, k) = array_channel[(WIN_SZ_SQ) >> 1];
    }

    OutputValues[0] = out_val;
    return;
}

template <int ROWS, int COLS, int PLANES, int TYPE, int NPC, int WORDWIDTH, int TC, int WIN_SZ, int WIN_SZ_SQ>
void ProcessMedianNxN(xf::cv::Mat<TYPE, ROWS, COLS, NPC>& _src_mat,
                      xf::cv::Mat<TYPE, ROWS, COLS, NPC>& _out_mat,
                      XF_TNAME(TYPE, NPC) buf[WIN_SZ][(COLS >> XF_BITSHIFT(NPC))],
                      XF_PTUNAME(TYPE) src_buf[WIN_SZ][XF_NPIXPERCYCLE(NPC) + (WIN_SZ - 1)],
                      XF_PTUNAME(TYPE) OutputValues[XF_NPIXPERCYCLE(NPC)],
                      XF_TNAME(TYPE, NPC) & P0,
                      uint16_t img_width,
                      uint16_t img_height,
                      uint16_t& shift_x,
                      ap_uint<13> row_ind[WIN_SZ],
                      ap_uint<13> row,
                      ap_uint<8> win_size,
                      int& rd_ind,
                      int& wr_ind) {
// clang-format off
    #pragma HLS INLINE
    // clang-format on

    XF_TNAME(TYPE, NPC) buf_cop[WIN_SZ];
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
        #pragma HLS LOOP_TRIPCOUNT min=1 max=WIN_SZ
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
        #pragma HLS LOOP_TRIPCOUNT min=1 max=TC/NPC
        #pragma HLS pipeline
        #pragma HLS LOOP_FLATTEN OFF
        // clang-format on
        if (col == (WIN_SZ >> 1) + 1662) int a = 0;
        if (row < img_height && col < (img_width >> XF_BITSHIFT(NPC))) {
            buf[row_ind[win_size - 1]][col] =
                _src_mat.read(rd_ind); //.data[(row*img_width>>XF_BITSHIFT(NPC)) + col]; // Read data
            rd_ind++;
        }

        if (NPC != XF_NPPC1) {
            for (int copy_buf_var = 0; copy_buf_var < WIN_SZ; copy_buf_var++) {
// clang-format off
                #pragma HLS LOOP_TRIPCOUNT min=1 max=WIN_SZ
                #pragma HLS UNROLL
                // clang-format on
                if ((row > (img_height - 1)) && (copy_buf_var > (win_size - 1 - (row - (img_height - 1))))) {
                    buf_cop[copy_buf_var] = buf[(row_ind[win_size - 1 - (row - (img_height - 1))])][col];
                } else {
                    if (col < (img_width >> XF_BITSHIFT(NPC)))
                        buf_cop[copy_buf_var] = buf[(row_ind[copy_buf_var])][col];
                    else
                        buf_cop[copy_buf_var] = buf_cop[copy_buf_var];
                }
            }

            XF_PTUNAME(TYPE) src_buf_temp_copy[WIN_SZ][XF_NPIXPERCYCLE(NPC)];
            XF_PTUNAME(TYPE) src_buf_temp_copy_extract[XF_NPIXPERCYCLE(NPC)];

            for (int extract_px = 0; extract_px < WIN_SZ; extract_px++) {
// clang-format off
                #pragma HLS LOOP_TRIPCOUNT min=1 max=WIN_SZ
                #pragma HLS unroll
                // clang-format on
                XF_TNAME(TYPE, NPC) toextract = buf_cop[extract_px];

                xfExtractPixels<NPC, XF_WORDWIDTH(TYPE, NPC), XF_DEPTH(TYPE, NPC)>(src_buf_temp_copy_extract, toextract,
                                                                                   0);
                for (int ext_copy = 0; ext_copy < npc; ext_copy++) {
// clang-format off
                    #pragma HLS unroll
                    // clang-format on
                    src_buf_temp_copy[extract_px][ext_copy] = src_buf_temp_copy_extract[ext_copy];
                }
            }
            for (int extract_px = 0; extract_px < WIN_SZ; extract_px++) {
// clang-format off
                #pragma HLS LOOP_TRIPCOUNT min=1 max=WIN_SZ
                // clang-format on
                for (int col_warp = 0; col_warp < (WIN_SZ >> 1); col_warp++) {
// clang-format off
                    #pragma HLS UNROLL
                    #pragma HLS LOOP_TRIPCOUNT min=1 max=WIN_SZ
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
                    #pragma HLS LOOP_TRIPCOUNT min=1 max=WIN_SZ
                    // clang-format on
                    for (int col_warp = 0; col_warp < npc + (WIN_SZ >> 1); col_warp++) {
// clang-format off
                        #pragma HLS UNROLL
                        #pragma HLS LOOP_TRIPCOUNT min=1 max=WIN_SZ
                        // clang-format on
                        src_buf[extract_px][col_warp] = src_buf_temp_copy[extract_px][0];
                    }
                }
            }

            XF_PTUNAME(TYPE) src_buf_temp_med_apply[WIN_SZ][XF_NPIXPERCYCLE(NPC) + (WIN_SZ - 1)];
            for (int applymedian = 0; applymedian < npc; applymedian++) {
// clang-format off
                #pragma HLS UNROLL
                // clang-format on
                for (int copyi = 0; copyi < WIN_SZ; copyi++) {
                    for (int copyj = 0; copyj < WIN_SZ; copyj++) {
                        src_buf_temp_med_apply[copyi][copyj] = src_buf[copyi][copyj + applymedian];
                    }
                }
                XF_PTUNAME(TYPE) OutputValues_percycle[XF_NPIXPERCYCLE(NPC)];
                xFMedianProc<PLANES, NPC, TYPE, WIN_SZ, WIN_SZ_SQ>(OutputValues_percycle, src_buf_temp_med_apply,
                                                                   WIN_SZ);
                OutputValues[applymedian] = OutputValues_percycle[0];
            }
            if (col >= 1) {
                shift_x = 0;
                P0 = 0;
                xfPackPixels<NPC, XF_WORDWIDTH(TYPE, NPC), XF_DEPTH(TYPE, NPC)>(OutputValues, P0, 0, npc, shift_x);
                _out_mat.write(wr_ind, P0);
                wr_ind++;
            }

            for (int extract_px = 0; extract_px < WIN_SZ; extract_px++) {
// clang-format off
                #pragma HLS LOOP_TRIPCOUNT min=1 max=WIN_SZ
                // clang-format on
                for (int col_warp = 0; col_warp < (WIN_SZ >> 1); col_warp++) {
// clang-format off
                    #pragma HLS UNROLL
                    #pragma HLS LOOP_TRIPCOUNT min=1 max=WIN_SZ
                    // clang-format on
                    src_buf[extract_px][col_warp] = src_buf[extract_px][col_warp + npc];
                }
            }

            for (int extract_px = 0; extract_px < WIN_SZ; extract_px++) {
// clang-format off
                #pragma HLS LOOP_TRIPCOUNT min=1 max=WIN_SZ
                // clang-format on
                for (int col_warp = 0; col_warp < npc; col_warp++) {
// clang-format off
                    #pragma HLS UNROLL
                    #pragma HLS LOOP_TRIPCOUNT min=1 max=WIN_SZ
                    // clang-format on
                    src_buf[extract_px][col_warp + (WIN_SZ >> 1)] = src_buf_temp_copy[extract_px][col_warp];
                }
            }

        } else {
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
            xFMedianProc<PLANES, NPC, TYPE, WIN_SZ, WIN_SZ_SQ>(OutputValues, src_buf, win_size);

            if (col >= (WIN_SZ >> 1)) {
                _out_mat.write(wr_ind, OutputValues[0]);
                if (wr_ind == 2073341) int a = 0;
                //				if (row == (img_height+(win_size>>1)-1))
                //					printf("%d ",(int)OutputValues[0]);
                wr_ind++;
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
        }
    } // Col_Loop
}

template <int ROWS, int COLS, int PLANES, int TYPE, int NPC, int WORDWIDTH, int TC, int WIN_SZ, int WIN_SZ_SQ>
void xFMedianNxN(xf::cv::Mat<TYPE, ROWS, COLS, NPC>& _src,
                 xf::cv::Mat<TYPE, ROWS, COLS, NPC>& _dst,
                 ap_uint<8> win_size,
                 uint16_t img_height,
                 uint16_t img_width) {
    ap_uint<13> row_ind[WIN_SZ];
// clang-format off
    #pragma HLS ARRAY_PARTITION variable=row_ind complete dim=1
    // clang-format on

    uint16_t shift_x = 0;
    ap_uint<13> row, col;
    XF_PTUNAME(TYPE) OutputValues[XF_NPIXPERCYCLE(NPC)];
// clang-format off
    #pragma HLS ARRAY_PARTITION variable=OutputValues complete dim=1
    // clang-format on

    XF_PTUNAME(TYPE) src_buf[WIN_SZ][XF_NPIXPERCYCLE(NPC) + (WIN_SZ - 1)];
// clang-format off
    #pragma HLS ARRAY_PARTITION variable=src_buf complete dim=1
    #pragma HLS ARRAY_PARTITION variable=src_buf complete dim=2
    // clang-format on
    // src_buf1 et al merged
    XF_TNAME(TYPE, NPC) P0;

    XF_TNAME(TYPE, NPC) buf[WIN_SZ][(COLS >> XF_BITSHIFT(NPC))];
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

    int rd_ind = 0;
read_lines:
    for (int init_buf = row_ind[win_size >> 1]; init_buf < row_ind[win_size - 1]; init_buf++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=1 max=WIN_SZ
        // clang-format on
        for (col = 0; col<img_width>> XF_BITSHIFT(NPC); col++) {
// clang-format off
            #pragma HLS LOOP_TRIPCOUNT min=1 max=TC/NPC
            #pragma HLS pipeline
            #pragma HLS LOOP_FLATTEN OFF
            // clang-format on
            buf[init_buf][col] = _src.read(rd_ind); //.data[(init_buf*(img_width>>XF_BITSHIFT(NPC))) + col];
            rd_ind++;
        }
    }

    // takes care of top borders
    for (col = 0; col<img_width>> XF_BITSHIFT(NPC); col++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=1 max=TC/NPC
        // clang-format on
        for (int init_buf = 0; init_buf<WIN_SZ>> 1; init_buf++) {
// clang-format off
            #pragma HLS LOOP_TRIPCOUNT min=1 max=WIN_SZ
            #pragma HLS UNROLL
            // clang-format on
            buf[init_buf][col] = buf[row_ind[win_size >> 1]][col];
        }
    }

    int wr_ind = 0;
Row_Loop:
    for (row = (win_size >> 1); row < img_height + (win_size >> 1); row++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=1 max=ROWS
        // clang-format on

        if (row == (img_height + (win_size >> 1) - 1)) int a = 0;
        P0 = 0;
        ProcessMedianNxN<ROWS, COLS, PLANES, TYPE, NPC, WORDWIDTH, TC, WIN_SZ, WIN_SZ_SQ>(
            _src, _dst, buf, src_buf, OutputValues, P0, img_width, img_height, shift_x, row_ind, row, win_size, rd_ind,
            wr_ind);

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

template <int FILTER_SIZE, int BORDER_TYPE, int TYPE, int ROWS, int COLS, int NPC = 1>
void medianBlur(xf::cv::Mat<TYPE, ROWS, COLS, NPC>& _src, xf::cv::Mat<TYPE, ROWS, COLS, NPC>& _dst) {
// clang-format off
    #pragma HLS INLINE OFF
    // clang-format on

    unsigned short imgheight = _src.rows;
    unsigned short imgwidth = _src.cols;

    assert(BORDER_TYPE == XF_BORDER_REPLICATE && "Only XF_BORDER_REPLICATE is supported");

    assert(((imgheight <= ROWS) && (imgwidth <= COLS)) && "ROWS and COLS should be greater than input image");

    xFMedianNxN<ROWS, COLS, XF_CHANNELS(TYPE, NPC), TYPE, NPC, 0, (COLS >> XF_BITSHIFT(NPC)) + (FILTER_SIZE >> 1),
                FILTER_SIZE, FILTER_SIZE * FILTER_SIZE>(_src, _dst, FILTER_SIZE, imgheight, imgwidth);

    return;
}
} // namespace cv
} // namespace xf
#endif
