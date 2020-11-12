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

#include "common/xf_common.hpp"
#include "common/xf_utility.hpp"

template <int NPC, int DEPTH_SRC, int DEPTH_DST, int WIN_SZ, int WIN_SZ_SQ>
void xFSobelProcx5x5(XF_PTNAME(DEPTH_DST) OutputValues[XF_NPIXPERCYCLE(NPC)],
                     XF_PTNAME(DEPTH_SRC) src_buf[WIN_SZ][XF_NPIXPERCYCLE(NPC) + (WIN_SZ - 1)],
                     ap_uint<8> win_size) {
// clang-format off
    #pragma HLS INLINE
    // clang-format on

    XF_PTNAME(DEPTH_DST) g_x = 0;
    XF_PTNAME(DEPTH_DST) M00 = (XF_PTNAME(DEPTH_DST))(src_buf[0][1] + src_buf[4][1]) << 1;
    XF_PTNAME(DEPTH_DST) M01 = (XF_PTNAME(DEPTH_DST))(src_buf[0][4] + src_buf[4][4]) - (src_buf[0][0] + src_buf[4][0]);
    XF_PTNAME(DEPTH_DST) A00 = (XF_PTNAME(DEPTH_DST))(src_buf[0][3] + src_buf[4][3]) << 1;
    XF_PTNAME(DEPTH_DST) M02 = (XF_PTNAME(DEPTH_DST))(src_buf[1][0] + src_buf[3][0]) << 2;
    XF_PTNAME(DEPTH_DST) M03 = (XF_PTNAME(DEPTH_DST))(src_buf[1][1] + src_buf[3][1]) << 3;
    XF_PTNAME(DEPTH_DST) A01 = (XF_PTNAME(DEPTH_DST))(src_buf[1][3] + src_buf[3][3]) << 3;
    XF_PTNAME(DEPTH_DST) A02 = (XF_PTNAME(DEPTH_DST))(src_buf[1][4] + src_buf[3][4]) << 2;
    XF_PTNAME(DEPTH_DST) M04 = src_buf[2][0] * 6;
    XF_PTNAME(DEPTH_DST) M05 = src_buf[2][1] * 12;
    XF_PTNAME(DEPTH_DST) A03 = src_buf[2][3] * 12;
    XF_PTNAME(DEPTH_DST) A04 = src_buf[2][4] * 6;
    XF_PTNAME(DEPTH_DST) S00 = M00 + M02;
    XF_PTNAME(DEPTH_DST) S01 = M03 + M04 + M05;
    XF_PTNAME(DEPTH_DST) A0 = A00 + A01;
    XF_PTNAME(DEPTH_DST) A1 = A02 + A03;
    XF_PTNAME(DEPTH_DST) A2 = A04 + M01;
    XF_PTNAME(DEPTH_DST) FA = A0 + A1 + A2;
    XF_PTNAME(DEPTH_DST) FS = S00 + S01;
    g_x = FA - FS;

    OutputValues[0] = g_x;
    return;
}

template <int NPC, int DEPTH_SRC, int DEPTH_DST, int WIN_SZ, int WIN_SZ_SQ>
void xFSobelProcy5x5(XF_PTNAME(DEPTH_DST) OutputValues[XF_NPIXPERCYCLE(NPC)],
                     XF_PTNAME(DEPTH_SRC) src_buf[WIN_SZ][XF_NPIXPERCYCLE(NPC) + (WIN_SZ - 1)],
                     ap_uint<8> win_size) {
// clang-format off
    #pragma HLS INLINE
    // clang-format on

    XF_PTNAME(DEPTH_DST) g_y = 0;
    XF_PTNAME(DEPTH_DST) M00 = (src_buf[4][0] + src_buf[4][4]) - (src_buf[0][0] + src_buf[0][4]);
    XF_PTNAME(DEPTH_DST) M01 = (XF_PTNAME(DEPTH_DST))(src_buf[0][1] + src_buf[0][3]) << 2;
    XF_PTNAME(DEPTH_DST) A00 = (XF_PTNAME(DEPTH_DST))(src_buf[4][1] + src_buf[4][3]) << 2;
    XF_PTNAME(DEPTH_DST) M02 = (XF_PTNAME(DEPTH_DST))(src_buf[1][0] + src_buf[1][4]) << 1;
    XF_PTNAME(DEPTH_DST) A01 = (XF_PTNAME(DEPTH_DST))(src_buf[3][0] + src_buf[3][4]) << 1;
    XF_PTNAME(DEPTH_DST) M03 = (XF_PTNAME(DEPTH_DST))(src_buf[1][1] + src_buf[1][3]) << 3;
    XF_PTNAME(DEPTH_DST) A02 = (XF_PTNAME(DEPTH_DST))(src_buf[3][1] + src_buf[3][3]) << 3;
    XF_PTNAME(DEPTH_DST) M04 = src_buf[0][2] * 6;
    XF_PTNAME(DEPTH_DST) M05 = src_buf[1][2] * 12;
    XF_PTNAME(DEPTH_DST) A03 = src_buf[3][2] * 12;
    XF_PTNAME(DEPTH_DST) A04 = src_buf[4][2] * 6;
    XF_PTNAME(DEPTH_DST) S00 = M01 + M02 + M03;
    XF_PTNAME(DEPTH_DST) S01 = M04 + M05;
    XF_PTNAME(DEPTH_DST) A0 = A00 + A01;
    XF_PTNAME(DEPTH_DST) A1 = A02 + A03;
    XF_PTNAME(DEPTH_DST) A2 = A04 + M00;
    XF_PTNAME(DEPTH_DST) FA = A0 + A1 + A2;
    XF_PTNAME(DEPTH_DST) FS = S00 + S01;
    g_y = FA - FS;

    OutputValues[0] = g_y;

    return;
}

template <int NPC, int DEPTH_SRC, int DEPTH_DST, int WIN_SZ, int WIN_SZ_SQ>
void xFSobelProcx(XF_PTNAME(DEPTH_DST) OutputValues[XF_NPIXPERCYCLE(NPC)],
                  XF_PTNAME(DEPTH_SRC) src_buf[WIN_SZ][XF_NPIXPERCYCLE(NPC) + (WIN_SZ - 1)],
                  ap_uint<8> win_size) {
// clang-format off
    #pragma HLS INLINE
    // clang-format on

    XF_PTNAME(DEPTH_SRC) array[WIN_SZ_SQ];
// clang-format off
    #pragma HLS ARRAY_PARTITION variable=array complete dim=1
    // clang-format on

    XF_PTNAME(DEPTH_DST) g_x = 0;
    XF_PTNAME(DEPTH_DST) M00 = ((XF_PTNAME(DEPTH_DST))src_buf[1][0] << 1);
    XF_PTNAME(DEPTH_DST) M01 = ((XF_PTNAME(DEPTH_DST))src_buf[1][2] << 1);
    XF_PTNAME(DEPTH_DST) A00 = (src_buf[0][2] + src_buf[2][2]);
    XF_PTNAME(DEPTH_DST) S00 = (src_buf[0][0] + src_buf[2][0]);
    g_x = M01 - M00;
    g_x = g_x + A00;
    g_x = g_x - S00;

    OutputValues[0] = g_x;
    return;
}

template <int NPC, int DEPTH_SRC, int DEPTH_DST, int WIN_SZ, int WIN_SZ_SQ>
void xFSobelProcy(XF_PTNAME(DEPTH_DST) OutputValues[XF_NPIXPERCYCLE(NPC)],
                  XF_PTNAME(DEPTH_SRC) src_buf[WIN_SZ][XF_NPIXPERCYCLE(NPC) + (WIN_SZ - 1)],
                  ap_uint<8> win_size) {
// clang-format off
    #pragma HLS INLINE
    // clang-format on

    XF_PTNAME(DEPTH_DST) g_y = 0;
    XF_PTNAME(DEPTH_DST) M00, M01;
    XF_PTNAME(DEPTH_DST) A00, S00;
    M00 = ((XF_PTNAME(DEPTH_DST))src_buf[0][1] << 1);
    M01 = ((XF_PTNAME(DEPTH_DST))src_buf[2][1] << 1);
    A00 = (src_buf[2][0] + src_buf[2][2]); //(b0 + b2);
    S00 = (src_buf[0][0] + src_buf[0][2]); //(t0 + t2);
    g_y = (M01 - M00);
    g_y = (g_y + A00);
    g_y = (g_y - S00);

    OutputValues[0] = g_y;

    return;
}

template <int SRC_T,
          int DST_T,
          int ROWS,
          int COLS,
          int DEPTH_SRC,
          int DEPTH_DST,
          int NPC,
          int WORDWIDTH_SRC,
          int WORDWIDTH_DST,
          int TC,
          int WIN_SZ,
          int WIN_SZ_SQ>
void ProcessSobelfunc(xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _src_mat,  // hls::stream< XF_SNAME(WORDWIDTH_SRC) > &_src,
                      xf::cv::Mat<DST_T, ROWS, COLS, NPC>& _out_mat,  // hls::stream< XF_SNAME(WORDWIDTH_DST) > &_dst,
                      xf::cv::Mat<DST_T, ROWS, COLS, NPC>& _out_mat1, // hls::stream< XF_SNAME(WORDWIDTH_DST) > &_dst1,
                      XF_SNAME(WORDWIDTH_SRC) buf[WIN_SZ][(COLS >> XF_BITSHIFT(NPC))],
                      XF_PTNAME(DEPTH_SRC) src_buf[WIN_SZ][XF_NPIXPERCYCLE(NPC) + (WIN_SZ - 1)],
                      XF_PTNAME(DEPTH_DST) OutputValues[XF_NPIXPERCYCLE(NPC)],
                      XF_PTNAME(DEPTH_DST) OutputValues1[XF_NPIXPERCYCLE(NPC)],
                      XF_SNAME(WORDWIDTH_DST) & P0,
                      XF_SNAME(WORDWIDTH_DST) & P1,
                      uint16_t img_width,
                      uint16_t img_height,
                      uint16_t& shift_x,
                      ap_uint<13> row_ind[WIN_SZ],
                      ap_uint<13> row,
                      ap_uint<8> win_size,
                      int& readind,
                      int& writeind,
                      int& writeind1) {
// clang-format off
    #pragma HLS INLINE
    // clang-format on

    XF_SNAME(WORDWIDTH_SRC) buf_cop[WIN_SZ];
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
            buf[row_ind[win_size - 1]][col] = _src_mat.read(readind++); // Read data

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

            XF_PTNAME(DEPTH_SRC) src_buf_temp_copy[WIN_SZ][XF_NPIXPERCYCLE(NPC)];
            XF_PTNAME(DEPTH_SRC) src_buf_temp_copy_extract[XF_NPIXPERCYCLE(NPC)];

            for (int extract_px = 0; extract_px < WIN_SZ; extract_px++) {
// clang-format off
                #pragma HLS LOOP_TRIPCOUNT min=WIN_SZ max=WIN_SZ
                #pragma HLS unroll
                // clang-format on
                XF_SNAME(WORDWIDTH_SRC) toextract = buf_cop[extract_px];
                xfExtractPixels<NPC, WORDWIDTH_SRC, DEPTH_SRC>(src_buf_temp_copy_extract, toextract, 0);
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

            XF_PTNAME(DEPTH_SRC) src_buf_temp_med_apply[WIN_SZ][XF_NPIXPERCYCLE(NPC) + (WIN_SZ - 1)];
            for (int applySobel = 0; applySobel < npc; applySobel++) {
// clang-format off
                #pragma HLS UNROLL
                // clang-format on
                for (int copyi = 0; copyi < WIN_SZ; copyi++) {
                    for (int copyj = 0; copyj < WIN_SZ; copyj++) {
                        src_buf_temp_med_apply[copyi][copyj] = src_buf[copyi][copyj + applySobel];
                    }
                }
                XF_PTNAME(DEPTH_DST) OutputValues_percycle[XF_NPIXPERCYCLE(NPC)];
                XF_PTNAME(DEPTH_DST) OutputValues_percycle1[XF_NPIXPERCYCLE(NPC)];
                if (WIN_SZ == 3) {
                    xFSobelProcx<NPC, DEPTH_SRC, DEPTH_DST, WIN_SZ, WIN_SZ_SQ>(OutputValues_percycle,
                                                                               src_buf_temp_med_apply, WIN_SZ);
                    xFSobelProcy<NPC, DEPTH_SRC, DEPTH_DST, WIN_SZ, WIN_SZ_SQ>(OutputValues_percycle1,
                                                                               src_buf_temp_med_apply, WIN_SZ);
                }
                if (WIN_SZ == 5) {
                    xFSobelProcx5x5<NPC, DEPTH_SRC, DEPTH_DST, WIN_SZ, WIN_SZ_SQ>(OutputValues_percycle,
                                                                                  src_buf_temp_med_apply, WIN_SZ);
                    xFSobelProcy5x5<NPC, DEPTH_SRC, DEPTH_DST, WIN_SZ, WIN_SZ_SQ>(OutputValues_percycle1,
                                                                                  src_buf_temp_med_apply, WIN_SZ);
                }
                OutputValues[applySobel] = OutputValues_percycle[0];
                OutputValues1[applySobel] = OutputValues_percycle1[0];
            }
            if (col >= 1) {
                shift_x = 0;
                P0 = 0;
                xfPackPixels<NPC, WORDWIDTH_DST, DEPTH_DST>(OutputValues, P0, 0, npc, shift_x);
                shift_x = 0;
                P1 = 0;
                xfPackPixels<NPC, WORDWIDTH_DST, DEPTH_DST>(OutputValues1, P1, 0, npc, shift_x);
                _out_mat.write(writeind++, P0);
                _out_mat1.write(writeind1++, P1);
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
            if (WIN_SZ == 3) {
                xFSobelProcx<NPC, DEPTH_SRC, DEPTH_DST, WIN_SZ, WIN_SZ_SQ>(OutputValues, src_buf, win_size);
                xFSobelProcy<NPC, DEPTH_SRC, DEPTH_DST, WIN_SZ, WIN_SZ_SQ>(OutputValues1, src_buf, win_size);
            }
            if (WIN_SZ == 5) {
                xFSobelProcx5x5<NPC, DEPTH_SRC, DEPTH_DST, WIN_SZ, WIN_SZ_SQ>(OutputValues, src_buf, win_size);
                xFSobelProcy5x5<NPC, DEPTH_SRC, DEPTH_DST, WIN_SZ, WIN_SZ_SQ>(OutputValues1, src_buf, win_size);
            }
            if (col >= (WIN_SZ >> 1)) {
                _out_mat.write(writeind++, OutputValues[0]);
                _out_mat1.write(writeind1++, OutputValues1[0]);
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

template <int SRC_T,
          int DST_T,
          int ROWS,
          int COLS,
          int DEPTH_SRC,
          int DEPTH_DST,
          int NPC,
          int WORDWIDTH_SRC,
          int WORDWIDTH_DST,
          int TC,
          int WIN_SZ,
          int WIN_SZ_SQ,
          bool USE_URAM>
void xFSobel3x3(xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _src_mat,  // hls::stream< XF_SNAME(WORDWIDTH_SRC) > &_src,
                xf::cv::Mat<DST_T, ROWS, COLS, NPC>& _out_mat,  // hls::stream< XF_SNAME(WORDWIDTH_DST) > &_dst,
                xf::cv::Mat<DST_T, ROWS, COLS, NPC>& _out_mat1, // hls::stream< XF_SNAME(WORDWIDTH_DST) > &_dst1,
                ap_uint<8> win_size,
                uint16_t img_height,
                uint16_t img_width) {
    ap_uint<13> row_ind[WIN_SZ];
// clang-format off
    #pragma HLS ARRAY_PARTITION variable=row_ind complete dim=1
    // clang-format on

    int readind = 0, writeind = 0, writeind1 = 0;

    uint16_t shift_x = 0;
    ap_uint<13> row, col;
    XF_PTNAME(DEPTH_DST) OutputValues[XF_NPIXPERCYCLE(NPC)];
    XF_PTNAME(DEPTH_DST) OutputValues1[XF_NPIXPERCYCLE(NPC)];
// clang-format off
    #pragma HLS ARRAY_PARTITION variable=OutputValues complete dim=1
    #pragma HLS ARRAY_PARTITION variable=OutputValues1 complete dim=1
    // clang-format on

    XF_PTNAME(DEPTH_SRC) src_buf[WIN_SZ][XF_NPIXPERCYCLE(NPC) + (WIN_SZ - 1)];
// clang-format off
    #pragma HLS ARRAY_PARTITION variable=src_buf complete dim=1
    #pragma HLS ARRAY_PARTITION variable=src_buf complete dim=2
    // clang-format on
    // src_buf1 et al merged
    XF_SNAME(WORDWIDTH_DST) P0;
    XF_SNAME(WORDWIDTH_DST) P1;

    XF_SNAME(WORDWIDTH_SRC) buf[WIN_SZ][(COLS >> XF_BITSHIFT(NPC))];
    if (USE_URAM) {
// clang-format off
        #pragma HLS array reshape variable=buf dim=1 factor=WIN_SZ cyclic
        #pragma HLS RESOURCE variable=buf core=RAM_S2P_URAM
        // clang-format on
    } else {
// clang-format off
        #pragma HLS RESOURCE variable=buf core=RAM_S2P_BRAM
        #pragma HLS ARRAY_PARTITION variable=buf complete dim=1
        // clang-format on
    }
    // initializing row index

    for (int init_row_ind = 0; init_row_ind < win_size; init_row_ind++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=WIN_SZ max=WIN_SZ
        // clang-format on
        row_ind[init_row_ind] = init_row_ind;
    }

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
            buf[init_buf][col] = _src_mat.read(readind++);
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
        P1 = 0;
        ProcessSobelfunc<SRC_T, DST_T, ROWS, COLS, DEPTH_SRC, DEPTH_DST, NPC, WORDWIDTH_SRC, WORDWIDTH_DST, TC, WIN_SZ,
                         WIN_SZ_SQ>(_src_mat, _out_mat, _out_mat1, buf, src_buf, OutputValues, OutputValues1, P0, P1,
                                    img_width, img_height, shift_x, row_ind, row, win_size, readind, writeind,
                                    writeind1);

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

template <int SRC_T,
          int DST_T,
          int ROWS,
          int COLS,
          int DEPTH_SRC,
          int DEPTH_DST,
          int NPC,
          int WORDWIDTH_SRC,
          int WORDWIDTH_DST,
          int WIN_SZ,
          bool USE_URAM = false>
void xFSobel(xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _src,  // hls::stream< XF_SNAME(WORDWIDTH_SRC) > &_src,
             xf::cv::Mat<DST_T, ROWS, COLS, NPC>& _dst,  // hls::stream< XF_SNAME(WORDWIDTH_DST) > &_dst,
             xf::cv::Mat<DST_T, ROWS, COLS, NPC>& _dst1, // hls::stream< XF_SNAME(WORDWIDTH_DST) > &_dst1,
             int _border_type,
             uint16_t imgheight,
             uint16_t imgwidth) {
#ifndef __SYNTHESIS__
    assert(((imgheight <= ROWS) && (imgwidth <= COLS)) && "ROWS and COLS should be greater than input image");
#endif
    xFSobel3x3<SRC_T, DST_T, ROWS, COLS, DEPTH_SRC, DEPTH_DST, NPC, WORDWIDTH_SRC, WORDWIDTH_DST,
               (COLS >> XF_BITSHIFT(NPC)) + (WIN_SZ >> 1), WIN_SZ, WIN_SZ * WIN_SZ, USE_URAM>(_src, _dst, _dst1, WIN_SZ,
                                                                                              imgheight, imgwidth);
}
