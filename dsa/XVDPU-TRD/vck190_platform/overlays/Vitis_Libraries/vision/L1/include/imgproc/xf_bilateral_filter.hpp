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

#ifndef __BILATERAL_FILTER__
#define __BILATERAL_FILTER__
#include "hls_stream.h"
#include "ap_int.h"
#include "common/xf_common.hpp"
#include "common/xf_utility.hpp"
#include <cmath>
#include <cstdlib>
namespace xf {
namespace cv {

static float xFBilateralFloatInv(float in_val) {
    return float(1.0 / (in_val));
}

static float xFBilateralFloatMul(float in_val1, float in_val2) {
    return float(in_val1 * in_val2);
}
static float xFBilateralExpf(float in_val) {
    float result;
    result = expf(in_val);
    return result;
}

template <int NPC, int DEPTH, int PLANES, int WIN_SZ, int WIN_SZ_SQ, int NUM_DIST, int FPRES_SC>
static void xFBilateralProc(XF_DTUNAME(DEPTH, NPC) OutputValues[XF_NPIXPERCYCLE(NPC)],
                            XF_DTUNAME(DEPTH, NPC) src_buf[WIN_SZ][XF_NPIXPERCYCLE(NPC) + (WIN_SZ - 1)],
                            ap_uint<8> win_size,
                            ap_ufixed<FPRES_SC, 1> exp_lut_sigma_color[WIN_SZ * WIN_SZ][NUM_DIST][256 * PLANES],
                            ap_int<8> distances_array_revmap[(WIN_SZ >> 1) * (WIN_SZ >> 1) + 1]) {
// clang-format off
    #pragma HLS INLINE
    // clang-format on
    XF_DTUNAME(DEPTH, NPC) pixel_mat[WIN_SZ][WIN_SZ];
// clang-format off
    #pragma HLS ARRAY_PARTITION variable=pixel_mat complete dim=1
    #pragma HLS ARRAY_PARTITION variable=pixel_mat complete dim=2
    // clang-format on
    for (int i = 0; i < WIN_SZ; i++) {
        for (int j = 0; j < WIN_SZ; j++) {
// clang-format off
            #pragma HLS UNROLL
            // clang-format on
            pixel_mat[i][j] = src_buf[i][j];
        }
    }
    ap_ufixed<16, 1> color_weights;
    ap_ufixed<32, 16> weight_sum = 0;
    ap_ufixed<32, 16> px_sum = 0;
    int buf_indx = 0;
    XF_DTUNAME(DEPTH, NPC) tmp;
    ap_int<24> diffpx;
    for (ap_uint<5> c = 0, k = 0; c < PLANES; c++, k += 8) {
// clang-format off
        #pragma HLS unroll
        // clang-format on
        weight_sum = 0;
        px_sum = 0;
        buf_indx = 0;
        for (ap_uint<5> i = 0; i < WIN_SZ; i++) {
            for (ap_uint<5> j = 0; j < WIN_SZ; j++) {
// clang-format off
                #pragma HLS unroll
                // clang-format on
                ap_uint<8> sub = WIN_SZ >> 1;
                ap_uint<8> sub_sq = sub * sub;
                ap_int<8> ei = i - sub;
                ap_int<8> ej = j - sub;
                ap_uint<8> comp = ei * ei;
                comp += ej * ej;
                if (comp > sub_sq) {
                    continue;
                } else {
                    if (PLANES == 3) {
                        diffpx =
                            std::abs(pixel_mat[i][j].range(7, 0) - pixel_mat[WIN_SZ >> 1][WIN_SZ >> 1].range(7, 0)) +
                            std::abs(pixel_mat[i][j].range(15, 8) - pixel_mat[WIN_SZ >> 1][WIN_SZ >> 1].range(15, 8)) +
                            std::abs(pixel_mat[i][j].range(23, 16) - pixel_mat[WIN_SZ >> 1][WIN_SZ >> 1].range(23, 16));
                    } else {
                        diffpx = (pixel_mat[i][j] - pixel_mat[WIN_SZ >> 1][WIN_SZ >> 1]);
                    }

                    if (diffpx < 0) {
                        diffpx = -diffpx;
                    }
                    if (comp == 0)
                        color_weights = 1;
                    else
                        color_weights = (ap_ufixed<16, 1>)(exp_lut_sigma_color[buf_indx >> 1]
                                                                              [distances_array_revmap[comp]][diffpx]);
                    px_sum += (color_weights) * (ap_uint<16>)(pixel_mat[i][j].range(k + 7, k));
                    weight_sum += color_weights;
                    buf_indx++;
                }
            }
        }

        float val = (float)1.0 / (float)weight_sum;
        float mul_val = (float)px_sum * (float)val;
        OutputValues[0].range(k + 7, k) = ((ap_ufixed<32, 16>)mul_val + (ap_ufixed<32, 16>)(0.5));
        //		OutputValues[0].range(k+7,k) =
        //(XF_DTUNAME(DEPTH,NPC))((ap_ufixed<32,16>)(xf::cv::xFBilateralFloatMul(px_sum,xf::cv::xFBilateralFloatInv(weight_sum)))
        //+ (ap_ufixed<32,16>)(0.5));
    }
    //	OutputValues[0]=weight_sum;
    return;
}

template <int TYPE,
          int ROWS,
          int COLS,
          int PLANES,
          int DEPTH,
          int NPC,
          int WORDWIDTH,
          int TC,
          int WIN_SZ,
          int WIN_SZ_SQ,
          int NUM_DIST,
          int FPRES_SC>
static void ProcessBilateralNXN(xf::cv::Mat<TYPE, ROWS, COLS, NPC>& _src_mat,
                                xf::cv::Mat<TYPE, ROWS, COLS, NPC>& _dst_mat,
                                XF_TNAME(DEPTH, NPC) buf[WIN_SZ][(COLS >> XF_BITSHIFT(NPC))],
                                XF_DTUNAME(DEPTH, NPC) src_buf[WIN_SZ][XF_NPIXPERCYCLE(NPC) + (WIN_SZ - 1)],
                                XF_DTUNAME(DEPTH, NPC) OutputValues[XF_NPIXPERCYCLE(NPC)],
                                XF_TNAME(DEPTH, NPC) & P0,
                                uint16_t img_width,
                                uint16_t img_height,
                                uint16_t& shift_x,
                                ap_uint<13> row_ind[WIN_SZ],
                                ap_uint<13> row,
                                ap_uint<8> win_size,
                                ap_ufixed<FPRES_SC, 1> exp_lut_sigma_color[WIN_SZ * WIN_SZ][NUM_DIST][256 * PLANES],
                                ap_int<8> distances_array_revmap[(WIN_SZ >> 1) * (WIN_SZ >> 1) + 1],
                                int& rd_ind,
                                int& wr_ind) {
// clang-format off
    #pragma HLS INLINE
    // clang-format on

    XF_TNAME(DEPTH, NPC) buf_cop[WIN_SZ];
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
        #pragma HLS LOOP_TRIPCOUNT min=3 max=WIN_SZ
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
        #pragma HLS LOOP_TRIPCOUNT min=1 max=TC
        #pragma HLS pipeline
        #pragma HLS LOOP_FLATTEN OFF
        // clang-format on
        if (row < img_height && col < (img_width >> XF_BITSHIFT(NPC)))
            buf[row_ind[win_size - 1]][col] = _src_mat.read(rd_ind++); // data[rd_ind++]; // Read data

        if (NPC != XF_NPPC1) {
            for (int copy_buf_var = 0; copy_buf_var < WIN_SZ; copy_buf_var++) {
// clang-format off
                #pragma HLS LOOP_TRIPCOUNT min=3 max=WIN_SZ
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

            XF_DTUNAME(DEPTH, NPC) src_buf_temp_copy[WIN_SZ][XF_NPIXPERCYCLE(NPC)];
            XF_DTUNAME(DEPTH, NPC) src_buf_temp_copy_extract[XF_NPIXPERCYCLE(NPC)];

            for (int extract_px = 0; extract_px < WIN_SZ; extract_px++) {
// clang-format off
                #pragma HLS LOOP_TRIPCOUNT min=3 max=WIN_SZ
                #pragma HLS unroll
                // clang-format on
                XF_TNAME(DEPTH, NPC) toextract = buf_cop[extract_px];
                xfExtractPixels<NPC, XF_WORDWIDTH(DEPTH, NPC), XF_DEPTH(DEPTH, NPC)>(src_buf_temp_copy_extract,
                                                                                     toextract, 0);
                // xfExtractPixels(src_buf_temp_copy_extract, toextract, 0);
                for (int ext_copy = 0; ext_copy < npc; ext_copy++) {
// clang-format off
                    #pragma HLS unroll
                    // clang-format on
                    src_buf_temp_copy[extract_px][ext_copy] = src_buf_temp_copy_extract[ext_copy];
                }
            }
            for (int extract_px = 0; extract_px < WIN_SZ; extract_px++) {
// clang-format off
                #pragma HLS LOOP_TRIPCOUNT min=3 max=WIN_SZ
                // clang-format on
                for (int col_warp = 0; col_warp < (WIN_SZ >> 1); col_warp++) {
// clang-format off
                    #pragma HLS UNROLL
                    #pragma HLS LOOP_TRIPCOUNT min=3 max=WIN_SZ
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
                    #pragma HLS LOOP_TRIPCOUNT min=3 max=WIN_SZ
                    // clang-format on
                    for (int col_warp = 0; col_warp < npc + (WIN_SZ >> 1); col_warp++) {
// clang-format off
                        #pragma HLS UNROLL
                        #pragma HLS LOOP_TRIPCOUNT min=3 max=WIN_SZ
                        // clang-format on
                        src_buf[extract_px][col_warp] = src_buf_temp_copy[extract_px][0];
                    }
                }
            }

            XF_DTUNAME(DEPTH, NPC) src_buf_temp_med_apply[WIN_SZ][XF_NPIXPERCYCLE(NPC) + (WIN_SZ - 1)];
            for (int applymedian = 0; applymedian < npc; applymedian++) {
// clang-format off
                #pragma HLS UNROLL
                // clang-format on
                for (int copyi = 0; copyi < WIN_SZ; copyi++) {
                    for (int copyj = 0; copyj < WIN_SZ; copyj++) {
                        src_buf_temp_med_apply[copyi][copyj] = src_buf[copyi][copyj + applymedian];
                    }
                }
                XF_DTUNAME(DEPTH, NPC) OutputValues_percycle[XF_NPIXPERCYCLE(NPC)];
                xFBilateralProc<NPC, DEPTH, PLANES, WIN_SZ, WIN_SZ_SQ, NUM_DIST, FPRES_SC>(
                    OutputValues_percycle, src_buf_temp_med_apply, WIN_SZ, exp_lut_sigma_color, distances_array_revmap);
                OutputValues[applymedian] = OutputValues_percycle[0];
            }
            if (col >= 1) {
                shift_x = 0;
                P0 = 0;
                xfPackPixels<NPC, XF_WORDWIDTH(DEPTH, NPC), XF_DEPTH(DEPTH, NPC)>(OutputValues, P0, 0, npc, shift_x);

                _dst_mat.write(wr_ind++, P0);
            }

            for (int extract_px = 0; extract_px < WIN_SZ; extract_px++) {
// clang-format off
                #pragma HLS LOOP_TRIPCOUNT min=3 max=WIN_SZ
                // clang-format on
                for (int col_warp = 0; col_warp < (WIN_SZ >> 1); col_warp++) {
// clang-format off
                    #pragma HLS UNROLL
                    #pragma HLS LOOP_TRIPCOUNT min=3 max=WIN_SZ
                    // clang-format on
                    src_buf[extract_px][col_warp] = src_buf[extract_px][col_warp + npc];
                }
            }

            for (int extract_px = 0; extract_px < WIN_SZ; extract_px++) {
// clang-format off
                #pragma HLS LOOP_TRIPCOUNT min=3 max=WIN_SZ
                // clang-format on
                for (int col_warp = 0; col_warp < npc; col_warp++) {
// clang-format off
                    #pragma HLS UNROLL
                    #pragma HLS LOOP_TRIPCOUNT min=3 max=WIN_SZ
                    // clang-format on
                    src_buf[extract_px][col_warp + (WIN_SZ >> 1)] = src_buf_temp_copy[extract_px][col_warp];
                }
            }

        } else {
            for (int copy_buf_var = 0; copy_buf_var < WIN_SZ; copy_buf_var++) {
// clang-format off
                #pragma HLS LOOP_TRIPCOUNT min=3 max=WIN_SZ
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
                #pragma HLS LOOP_TRIPCOUNT min=3 max=WIN_SZ
                #pragma HLS UNROLL
                // clang-format on
                if (col < img_width) {
                    src_buf[extract_px][win_size - 1] = buf_cop[extract_px];
                } else {
                    src_buf[extract_px][win_size - 1] = src_buf[extract_px][win_size - 2];
                }
            }
            xFBilateralProc<NPC, DEPTH, PLANES, WIN_SZ, WIN_SZ_SQ, NUM_DIST, FPRES_SC>(
                OutputValues, src_buf, win_size, exp_lut_sigma_color, distances_array_revmap);
            if (col >= (WIN_SZ >> 1)) {
                _dst_mat.write(wr_ind++, OutputValues[0]);
            }
            for (int wrap_buf = 0; wrap_buf < WIN_SZ; wrap_buf++) {
// clang-format off
                #pragma HLS UNROLL
                #pragma HLS LOOP_TRIPCOUNT min=3 max=WIN_SZ
                // clang-format on
                for (int col_warp = 0; col_warp < WIN_SZ - 1; col_warp++) {
// clang-format off
                    #pragma HLS UNROLL
                    #pragma HLS LOOP_TRIPCOUNT min=3 max=WIN_SZ
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

template <int TYPE,
          int ROWS,
          int COLS,
          int PLANES,
          int DEPTH,
          int NPC,
          int WORDWIDTH,
          int TC,
          int WIN_SZ,
          int WIN_SZ_SQ,
          int NUM_DIST,
          int FPRES_SC>
static void xFBilateralFilterNXN(xf::cv::Mat<TYPE, ROWS, COLS, NPC>& _src_mat,
                                 xf::cv::Mat<TYPE, ROWS, COLS, NPC>& _dst_mat,
                                 ap_uint<8> win_size,
                                 uint16_t img_height,
                                 uint16_t img_width,
                                 ap_ufixed<FPRES_SC, 1> exp_lut_sigma_color[WIN_SZ * WIN_SZ][NUM_DIST][256 * PLANES],
                                 ap_int<8> distances_array_revmap[(WIN_SZ >> 1) * (WIN_SZ >> 1) + 1]) {
// clang-format off
    #pragma HLS INLINE
    // clang-format on
    ap_uint<13> row_ind[WIN_SZ];
// clang-format off
    #pragma HLS ARRAY_PARTITION variable=row_ind complete dim=1
    // clang-format on

    uint16_t shift_x = 0;
    ap_uint<13> row, col;
    int rd_ind = 0, wr_ind = 0;
    XF_DTUNAME(DEPTH, NPC) OutputValues[XF_NPIXPERCYCLE(NPC)];
// clang-format off
    #pragma HLS ARRAY_PARTITION variable=OutputValues complete dim=1
    // clang-format on

    XF_DTUNAME(DEPTH, NPC) src_buf[WIN_SZ][XF_NPIXPERCYCLE(NPC) + (WIN_SZ - 1)];
// clang-format off
    #pragma HLS ARRAY_PARTITION variable=src_buf complete dim=1
    #pragma HLS ARRAY_PARTITION variable=src_buf complete dim=2
    // clang-format on
    // src_buf1 et al merged
    XF_TNAME(DEPTH, NPC) P0;

    XF_TNAME(DEPTH, NPC) buf[WIN_SZ][(COLS >> XF_BITSHIFT(NPC))];
// clang-format off
    #pragma HLS ARRAY_PARTITION variable=buf complete dim=1
    #pragma HLS RESOURCE variable=buf core=RAM_S2P_BRAM
    // clang-format on

    // initializing row index

    for (int init_row_ind = 0; init_row_ind < win_size; init_row_ind++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=3 max=WIN_SZ
        // clang-format on
        row_ind[init_row_ind] = init_row_ind;
    }

read_lines:
    for (int init_buf = row_ind[win_size >> 1]; init_buf < row_ind[win_size - 1]; init_buf++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=3 max=WIN_SZ
        // clang-format on
        for (col = 0; col<img_width>> XF_BITSHIFT(NPC); col++) {
// clang-format off
            #pragma HLS LOOP_TRIPCOUNT min=1 max=TC
            #pragma HLS pipeline
            #pragma HLS LOOP_FLATTEN OFF
            // clang-format on
            buf[init_buf][col] = _src_mat.read(rd_ind++); //_src_mat.data[rd_ind++];
        }
    }

    // takes care of top borders
    for (col = 0; col<img_width>> XF_BITSHIFT(NPC); col++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=1 max=TC
        // clang-format on
        for (int init_buf = 0; init_buf<WIN_SZ>> 1; init_buf++) {
// clang-format off
            #pragma HLS LOOP_TRIPCOUNT min=3 max=WIN_SZ
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
        ProcessBilateralNXN<TYPE, ROWS, COLS, PLANES, DEPTH, NPC, WORDWIDTH, TC, WIN_SZ, WIN_SZ_SQ, NUM_DIST, FPRES_SC>(
            _src_mat, _dst_mat, buf, src_buf, OutputValues, P0, img_width, img_height, shift_x, row_ind, row, win_size,
            exp_lut_sigma_color, distances_array_revmap, rd_ind, wr_ind);

        // update indices
        ap_uint<13> zero_ind = row_ind[0];
        for (int init_row_ind = 0; init_row_ind < WIN_SZ - 1; init_row_ind++) {
// clang-format off
            #pragma HLS LOOP_TRIPCOUNT min=3 max=WIN_SZ
            #pragma HLS UNROLL
            // clang-format on
            row_ind[init_row_ind] = row_ind[init_row_ind + 1];
        }
        row_ind[win_size - 1] = zero_ind;
    } // Row_Loop
}

template <int TYPE, int ROWS, int COLS, int PLANES, int DEPTH, int NPC, int WORDWIDTH, int WIN_SZ>
static void xFbilateralFilterKernel(xf::cv::Mat<TYPE, ROWS, COLS, NPC>& _src_mat,
                                    xf::cv::Mat<TYPE, ROWS, COLS, NPC>& _dst_mat,
                                    int _border_type,
                                    uint16_t imgheight,
                                    uint16_t imgwidth,
                                    float sigma_color,
                                    float sigma_space) {
// clang-format off
    #pragma HLS INLINE OFF
    #pragma HLS ALLOCATION function instances=xf::cv::xFBilateralFloatMul limit=1 
    #pragma HLS ALLOCATION function instances=xf::cv::xFBilateralFloatInv limit=1 
    #pragma HLS ALLOCATION function instances=xf::cv::xFBilateralExpf limit=1
    // clang-format on
    // assert(_border_type == XF_BORDER_REPLICATE && "Only XF_BORDER_REPLICATE is supported");

    // assert(((imgheight <= ROWS ) && (imgwidth <= COLS)) && "ROWS and COLS should be greater than input image");

    // assert(((WIN_SZ == 7 ) || (WIN_SZ == 5) || (WIN_SZ == 3 )) && "Window Size should only be 3, 5 or 7");

    // compute the inverse square of sigma_color and sigma_space
    float sigma_color_sqinv = xf::cv::xFBilateralFloatInv(xf::cv::xFBilateralFloatMul(sigma_color, sigma_color));
    float sigma_space_sqinv = xf::cv::xFBilateralFloatInv(xf::cv::xFBilateralFloatMul(sigma_space, sigma_space));

    // find the number of valied distances for the filter size
    // for 3x3, the euclidean distances to the neighborhood pixels, 0 and 1 are valid
    // for 5x5, the euclidean distances to the neighborhood pixels, 0, 1, sqrt(2) are valid
    // for 7x7, the euclidean distances to the neighborhood pixels, 0, 1, sqrt(2), 2, sqrt(5), sqrt(8) and 3 are valid
    // TODO: Come up with a formula to get the number of distances
    const int NUM_DIST = (WIN_SZ == 3) ? 1 : ((WIN_SZ == 5) ? 3 : 6);
    // fractional precision to store the weights. The actual precision will be the number below - 1
    const int FPRES_SC = 16;

    // as the bilateral filter has a circular kernel not a square kernel,
    // not all the weights in the square window are going to be valid.
    // distances greater than (WIN_SZ-1/2) need to be ignored
    // the following code computes the number of such distances per square
    ap_uint<8> sub = WIN_SZ >> 1;
    ap_uint<8> sub_sq = sub * sub;
    int comp_sz = 0;
    for (int i = 0; i < WIN_SZ; i++) {
        for (int j = 0; j < WIN_SZ; j++) {
            int temp_cmp = (i - sub) * (i - sub) + (j - sub) * (j - sub);
            if (temp_cmp <= sub_sq) comp_sz++;
        }
    }

    // the following code computes all the valid (squares of) distances
    // The size of the array is NUM_DIST for the given window size
    ap_uint<8> distances_array[(WIN_SZ >> 1) * (WIN_SZ >> 1) + 1];
// clang-format off
    #pragma HLS ARRAY_PARTITION variable=distances_array complete dim=1
    // clang-format on
    int dist_index = 0;
    for (int i = 0; i <= (WIN_SZ >> 1); i++) {
        for (int j = 0; j <= i; j++) {
            int temp_cmp = (i) * (i) + (j) * (j);
            if (temp_cmp <= sub_sq && temp_cmp != 0) {
                distances_array[dist_index] = temp_cmp;
                dist_index++;
            }
        }
    }

    // this array is to reverse map the index to each of the buffer location
    // Theis is necessary to reduce the BRAM utilization. The squares of distances for 7x7,
    // 1, 2, 4, 5, 8, 9 if directly index, take 9 locations while they just need 6
    // This reverse map stores the location for each of the square distances
    ap_int<8> distances_array_revmap[(WIN_SZ >> 1) * (WIN_SZ >> 1) + 1];
// clang-format off
    #pragma HLS ARRAY_PARTITION variable=distances_array_revmap complete dim=1
    // clang-format on
    int index = 0;
    for (int i = 0; i <= sub_sq; i++) {
        if (distances_array[index] == i) {
            distances_array_revmap[i] = index;
            index++;
        } else
            distances_array_revmap[i] = -1;
    }

    // this array stores the weights needed to filter the input image
    // the weight of a pixel in the neighborhood is exp(-0.5*d^2/sigma_space^2 -0.5*(I1-I2)^2/sigma_color^2)
    // where d is the euclidean distance between the center pixel and the neighborhood pixel
    // I1-I2 is the difference in pixel intensities of the center and neighborhood pixels
    // The array is replicated as many times as needed by the kernel.
    // for 3x3 - 4, for 5x5 - 12, for 7x7 - 28. Weight for the central pixel is always 1
    // The numbers 4, 12 and 28 are the Number of valid pixels for a circular window in the square window
    static const int array_size = comp_sz;
    ap_ufixed<FPRES_SC, 1> exp_lut_sigma_color[WIN_SZ * WIN_SZ][NUM_DIST][256 * PLANES];
// clang-format off
    #pragma HLS ARRAY_PARTITION variable=exp_lut_sigma_color complete dim=1
    #pragma HLS ARRAY_PARTITION variable=exp_lut_sigma_color complete dim=2
    // clang-format on

    if (NPC == 8) {
// clang-format off
        #pragma HLS ARRAY_PARTITION variable=exp_lut_sigma_color complete dim=3
        // clang-format on
    }

    for (unsigned int m = 0; m < (256 * PLANES); m++) {
        ap_uint<32> jsq = (ap_uint<16>)m * (ap_uint<16>)m;
        for (int i = 0; i < NUM_DIST; i++) {
// clang-format off
            #pragma HLS PIPELINE
            // clang-format on
            for (unsigned short k = 0; k < array_size; k++) {
// clang-format off
                #pragma HLS UNROLL
                // clang-format on
                exp_lut_sigma_color[k][i][m] = xf::cv::xFBilateralFloatMul(
                    xf::cv::xFBilateralExpf(
                        xf::cv::xFBilateralFloatMul(-0.5, xf::cv::xFBilateralFloatMul(sigma_color_sqinv, jsq))),
                    xf::cv::xFBilateralExpf(xf::cv::xFBilateralFloatMul(
                        -0.5, xf::cv::xFBilateralFloatMul(sigma_space_sqinv, distances_array[i]))));
            }
        }
    }
    xFBilateralFilterNXN<TYPE, ROWS, COLS, PLANES, DEPTH, NPC, WORDWIDTH, (COLS >> (XF_BITSHIFT(NPC))) + (WIN_SZ >> 1),
                         WIN_SZ, WIN_SZ * WIN_SZ, NUM_DIST, FPRES_SC>(_src_mat, _dst_mat, WIN_SZ, imgheight, imgwidth,
                                                                      exp_lut_sigma_color, distances_array_revmap);
}

template <int WINDOW_SIZE, int BORDER_TYPE, int TYPE, int ROWS, int COLS, int NPC = 1>
void bilateralFilter(xf::cv::Mat<TYPE, ROWS, COLS, NPC>& _src_mat,
                     xf::cv::Mat<TYPE, ROWS, COLS, NPC>& _dst_mat,
                     float sigma_color,
                     float sigma_space) {
// clang-format off
    #pragma HLS INLINE OFF
    // clang-format on

    xFbilateralFilterKernel<TYPE, ROWS, COLS, XF_CHANNELS(TYPE, NPC), TYPE, NPC, (TYPE << (XF_BITSHIFT(NPC))),
                            WINDOW_SIZE>(_src_mat, _dst_mat, BORDER_TYPE, _src_mat.rows, _src_mat.cols, sigma_color,
                                         sigma_space);
}
} // namespace cv
} // namespace xf
#endif
