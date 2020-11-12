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

#ifndef _XF_EDGE_TRACING_HPP_
#define _XF_EDGE_TRACING_HPP_

#ifndef __cplusplus
#error C++ is needed to use this file!
#endif

#include "hls_stream.h"
#include "common/xf_common.hpp"
#include "common/xf_utility.hpp"
#include <ap_int.h>
#include <string.h>

#define INTRA_ITERATIONS 8
#define INTER_ITERATIONS 2
#define SLICES 4
#define PIXELS 34
#define MIN_OVERLAP 10
#define PIXEL_PROCESS_BITS 68

#define MIN(x, y) (((x) < (y)) ? (x) : (y))
#define DIV_CEIL(x, y) (((x) + (y)-1) / (y))
#define ADJUST_MULTIPLE(x, y) (DIV_CEIL(x, y) * (y))

namespace xf {
namespace cv {

static void applyEqn(ap_uint<2>& x0,
                     ap_uint<2>& x1,
                     ap_uint<2>& x2,
                     ap_uint<2>& x3,
                     ap_uint<2>& a,
                     ap_uint<2>& x4,
                     ap_uint<2>& x5,
                     ap_uint<2>& x6,
                     ap_uint<2>& x7) {
// clang-format off
    #pragma HLS inline
    // clang-format on

    //# Apply equations
    bool a0 = a.range(1, 1);
    bool a1 = a.range(0, 0);

    a0 = (x0.range(1, 1) | x1.range(1, 1) | x2.range(1, 1) | x3.range(1, 1) | x4.range(1, 1) | x5.range(1, 1) |
          x6.range(1, 1) | x7.range(1, 1) | a.range(1, 1)) &
         (a.range(0, 0));

    x0.range(1, 1) = (a0 & x0.range(0, 0)) | x0.range(1, 1);
    x1.range(1, 1) = (a0 & x1.range(0, 0)) | x1.range(1, 1);
    x2.range(1, 1) = (a0 & x2.range(0, 0)) | x2.range(1, 1);
    x3.range(1, 1) = (a0 & x3.range(0, 0)) | x3.range(1, 1);
    x4.range(1, 1) = (a0 & x4.range(0, 0)) | x4.range(1, 1);
    x5.range(1, 1) = (a0 & x5.range(0, 0)) | x5.range(1, 1);
    x6.range(1, 1) = (a0 & x6.range(0, 0)) | x6.range(1, 1);
    x7.range(1, 1) = (a0 & x7.range(0, 0)) | x7.range(1, 1);

    //# Center pixel update
    a.range(1, 1) = a0;
    a.range(0, 0) = a1;
}

template <int n>
void PixelProcessNew(ap_uint<PIXEL_PROCESS_BITS> k1,
                     ap_uint<PIXEL_PROCESS_BITS> k2,
                     ap_uint<PIXEL_PROCESS_BITS> k3,
                     ap_uint<PIXEL_PROCESS_BITS>& l1,
                     ap_uint<PIXEL_PROCESS_BITS>& l2,
                     ap_uint<PIXEL_PROCESS_BITS>& l3) {
// clang-format off
    #pragma HLS inline off
    // clang-format on

    ap_uint<2> x1[PIXELS], x2[PIXELS], x3[PIXELS];
    ap_uint<2> y1[PIXELS], y2[PIXELS], y3[PIXELS];
    ap_uint<2> z1[PIXELS], z2[PIXELS], z3[PIXELS];

    for (int i = 0, j = 0; i < PIXEL_PROCESS_BITS; i += 2, j++) {
// clang-format off
        #pragma HLS unroll
        // clang-format on
        x1[j] = k1.range(i + 1, i);
        x2[j] = k2.range(i + 1, i);
        x3[j] = k3.range(i + 1, i);
    }

PL_1:
    for (int i = 1; i < PIXELS - 1; i += 3) {
// clang-format off
        #pragma HLS unroll
        // clang-format on
        applyEqn(x1[i - 1], x1[i], x1[i + 1], x2[i - 1], x2[i], x2[i + 1], x3[i - 1], x3[i], x3[i + 1]);
    }

    for (int i = 0; i < PIXELS; i++) {
// clang-format off
        #pragma HLS unroll
        // clang-format on
        y1[i] = x1[i];
        y2[i] = x2[i];
        y3[i] = x3[i];
    }

PL_2:
    for (int i = 2; i < PIXELS; i += 3) {
// clang-format off
        #pragma HLS unroll
        // clang-format on
        applyEqn(y1[i - 1], y1[i], y1[i + 1], y2[i - 1], y2[i], y2[i + 1], y3[i - 1], y3[i], y3[i + 1]);
    }

    for (int i = 0; i < PIXELS; i++) {
// clang-format off
        #pragma HLS unroll
        // clang-format on
        z1[i] = y1[i];
        z2[i] = y2[i];
        z3[i] = y3[i];
    }

PL_3:
    for (int i = 3; i < PIXELS - 1; i += 3) {
// clang-format off
        #pragma HLS unroll
        // clang-format on
        applyEqn(z1[i - 1], z1[i], z1[i + 1], z2[i - 1], z2[i], z2[i + 1], z3[i - 1], z3[i], z3[i + 1]);
    }

    for (int i = 0, j = 0; i < PIXEL_PROCESS_BITS; i += 2, j++) {
        l1.range(i + 1, i) = z1[j];
        l2.range(i + 1, i) = z2[j];
        l3.range(i + 1, i) = z3[j];
    }
}

template <int BRAMS, int BRAMS_SETS_BY_3, int DEPTH>
void TopDown(ap_uint<64> iBuff[BRAMS][DEPTH],
             uint16_t width,
             uint16_t height,
             int bramsetsval,
             int bramtotal,
             int bdrows,
             int ram_row_depth) {
    ap_uint<64> arr1[BRAMS], arr2[BRAMS], arr4[BRAMS];
// clang-format off
    #pragma HLS array_partition variable=arr1 complete
    #pragma HLS array_partition variable=arr2 complete
    #pragma HLS array_partition variable=arr4 complete
    // clang-format on

    ap_uint<4> arr3[BRAMS], arr5[BRAMS];
// clang-format off
    #pragma HLS array_partition variable=arr3 complete
    #pragma HLS array_partition variable=arr5 complete
    // clang-format on

    int countind = 0;

    for (int j = 0; j < 3; j++) {
    RD_INIT:
        for (int i = 0; i < BRAMS; i++) {
// clang-format off
            #pragma HLS unroll
            #pragma HLS loop_tripcount min=38 max=38
            // clang-format on

            arr1[i] = iBuff[i][0];
            arr3[i] = arr1[i].range(3, 0);
        }

    // Elements per RAM
    ELEMENTS_P_RAM:
        for (int el = 1; el < (ram_row_depth * bdrows);
             el++) { // (width/32)*number of rows possible in one bram(512 depth)//(ram_row_depth * bdrows)
                     // clang-format off
                     #pragma HLS loop_tripcount min=480 max=480
                     #pragma HLS pipeline II=1
                     #pragma HLS loop_flatten off
                     #pragma HLS DEPENDENCE variable=arr1 inter false
                     #pragma HLS DEPENDENCE variable=arr2 inter false
                     // clang-format on

        RD:
            for (int i = 0; i < BRAMS; i++) {
// clang-format off
                #pragma HLS unroll
                #pragma HLS loop_tripcount min=38 max=38
                // clang-format on

                arr2[i].range(3, 0) = arr3[i];
                arr2[i].range(63, 4) = arr1[i].range(63, 4);
                arr1[i] = iBuff[i][el];
                arr3[i] = arr1[i].range(3, 0);
                arr4[i] = arr2[i];
            }

        RD1:
            for (int i = 1, k = 0; i < BRAMS - 3; i += 3, k++) {
// clang-format off
                #pragma HLS unroll
                #pragma HLS loop_tripcount min=38 max=38
                // clang-format on

                ap_uint<PIXEL_PROCESS_BITS> k1, k2, k3;
                ap_uint<PIXEL_PROCESS_BITS> l1, l2, l3;

                k1.range(63, 0) = arr2[i + j - 1];
                k2.range(63, 0) = arr2[i + j + 0];
                k3.range(63, 0) = arr2[i + j + 1];

                k1.range(PIXEL_PROCESS_BITS - 1, 64) = arr1[i + j - 1].range(3, 0);
                k2.range(PIXEL_PROCESS_BITS - 1, 64) = arr1[i + j + 0].range(3, 0);
                k3.range(PIXEL_PROCESS_BITS - 1, 64) = arr1[i + j + 1].range(3, 0);

                PixelProcessNew<1>(k1, k2, k3, l1, l2, l3);

                arr4[i + j - 1] = l1.range(63, 0);
                arr4[i + j + 0] = l2.range(63, 0);
                arr4[i + j + 1] = l3.range(63, 0);

                arr3[i + j - 1] = l1.range(PIXEL_PROCESS_BITS - 1, 64);
                arr3[i + j + 0] = l2.range(PIXEL_PROCESS_BITS - 1, 64);
                arr3[i + j + 1] = l3.range(PIXEL_PROCESS_BITS - 1, 64);
            }

        RD2:
            for (int ii = 0; ii < BRAMS; ii++) {
// clang-format off
                #pragma HLS unroll
                #pragma HLS loop_tripcount min=38 max=38
                // clang-format on

                if ((ii == 0) && (el <= ram_row_depth * (bdrows - 1))) {
                    iBuff[0][ram_row_depth + el - 1] = arr4[bramtotal - 1];
                } else {
                    iBuff[ii][el - 1] = arr4[ii];
                }
            }
        }
    }
}

/**
 * xfEdgeTracing : Connects edge
 */
template <int SRC_T, int DST_T, int NPC_SRC, int NPC_DST, int HEIGHT, int WIDTH, bool USE_URAM>
static void xfEdgeTracing(xf::cv::Mat<DST_T, HEIGHT, WIDTH, NPC_DST>& _dst,
                          xf::cv::Mat<SRC_T, HEIGHT, WIDTH, NPC_SRC>& _src,
                          uint16_t height,
                          uint16_t width) {
// clang-format off
#pragma HLS INLINE
// clang-format on
#define BRAM_DEPTH (USE_URAM ? 4096 : 1024)

    enum {
        RAM_ROW_DEPTH = (WIDTH / 32),                // 64-bit width = 32 pixels; Gives depth of ram a row occupies
        NUM_ROWS_RAM = (BRAM_DEPTH / RAM_ROW_DEPTH), // Gives No.of rows per BRAM
        SLICE_H = (HEIGHT / SLICES),                 // Gives height of each Slice
        BRAM_SETS_TEMP = DIV_CEIL((SLICE_H + MIN_OVERLAP), NUM_ROWS_RAM),
        BRAMS_SETS = ADJUST_MULTIPLE(BRAM_SETS_TEMP, 3), // Making BRAM_CNT divisible by 3
        ACTUAL_ROWS = BRAMS_SETS * NUM_ROWS_RAM,
        OVERLAP = ACTUAL_ROWS - SLICE_H,
        BRAMS_TOTAL = BRAMS_SETS + 2
    };

    ap_uint<64> iBuff[BRAMS_TOTAL][BRAM_DEPTH];

    if (USE_URAM) {
// clang-format off
        #pragma HLS RESOURCE variable=iBuff core=RAM_T2P_URAM
        #pragma HLS array_partition variable=iBuff dim=1
        // clang-format on
    } else {
// clang-format off
        #pragma HLS RESOURCE variable=iBuff core=RAM_T2P_BRAM
        #pragma HLS array_partition variable=iBuff dim=1
        // clang-format on
    }

    //# I/P & O/P Registers
    ap_uint<64> iReg[1];
    ap_uint<64> oReg[1];

    int slice_h = (height >> 2);      // = height / SLICES
    int ram_row_depth = (width >> 5); // = width / 32
    int bdrows = BRAM_DEPTH / ram_row_depth;
    int bramsetsval = ADJUST_MULTIPLE(DIV_CEIL((slice_h + MIN_OVERLAP), bdrows), 3);
    int overlap = (bramsetsval * bdrows) - slice_h;
    int bramtotal = bramsetsval + 2;

//# Inter Iterations

INTER_ITERATION_LOOP:
    for (int inter_i = 0; inter_i < INTER_ITERATIONS; inter_i++) {
        //# Loop for Reading chunks of NMS output
        unsigned int offset = 0;
        unsigned int lBound = 0;

    SLICE_LOOP:
        for (int slice = 0; slice < SLICES; slice++) {
            lBound = ram_row_depth * (slice_h + ((slice == 3) ? 0 : overlap));

            if (inter_i == 0) {
                offset = slice * slice_h * ram_row_depth;
            } else {
                offset = (slice == 3) ? 0 : ((((3 - slice) * slice_h) - overlap) * ram_row_depth);
            }

            ap_uint<16> idx1 = 0, dep = 0;
            ap_uint<16> idx2 = 1;
            int cnt = 0;

        Read_N_Arrange:
            for (unsigned int i = 0; i < lBound; i++) {
// clang-format off
                #pragma HLS loop_tripcount min=16200 max=16800
                #pragma HLS pipeline II=1
                #pragma HLS DEPENDENCE variable=iBuff inter false
                #pragma HLS DEPENDENCE variable=iBuff intra false
                // clang-format on
                int ind_1 = 0, ind_2 = 0, val_ind = 0;
                iReg[0] = _src.read(offset + i); // Reading Input

                if (idx1 == ram_row_depth) {
                    idx1 = 0;
                    idx2++;
                }

                if (idx2 == bramsetsval + 1) {
                    idx2 = 1;
                    dep += ram_row_depth;
                }

                ap_uint<16> index = idx1 + dep;
                iBuff[idx2][index] = iReg[0];

                // Filling edge row buffers (i.e., iBuff[0] and iBuff[bramsetsval+1])
                // This is done by replicating the rows
                if (idx2 == 1) {
                    if (dep == 0) {
                        iBuff[0][index] = 0;
                    } else {
                        iBuff[bramsetsval + 1][index - ram_row_depth] = iReg[0];
                    }
                } else if (idx2 == bramsetsval) {
                    if (dep == ((bdrows - 1) * ram_row_depth)) {
                        iBuff[bramsetsval + 1][index] = 0;

                    } else {
                        iBuff[0][index + ram_row_depth] = iReg[0];
                    }
                }

                idx1++;
            }

        //# Intra Iterations
        INTRA_ITERATION_LOOP:
            for (int intra_i = 0; intra_i < INTRA_ITERATIONS; intra_i++) {
                TopDown<BRAMS_TOTAL, BRAMS_SETS / 3, BRAM_DEPTH>(iBuff, width, height, bramsetsval, bramtotal, bdrows,
                                                                 ram_row_depth);
            }

            idx1 = 0;
            idx2 = 1;
            dep = 0;

        Write:
            for (unsigned int i = 0; i < lBound; i++) {
// clang-format off
                #pragma HLS loop_tripcount min=16200 max=16800
                #pragma HLS pipeline
                // clang-format on

                if (idx1 == ram_row_depth) {
                    idx1 = 0;
                    idx2++;
                }
                if (idx2 == bramsetsval + 1) {
                    idx2 = 1;
                    dep += ram_row_depth;
                }

                oReg[0] = iBuff[idx2][idx1 + dep];
                _src.write((offset + i), oReg[0]);

                idx1++;
            }
        }
    }

    ap_uint<64> oBuff[RAM_ROW_DEPTH], oRegF[1];
//# Write the final output as 8-bit / pixel
FIN_WR_LOOP:
    for (int ii = 0; ii < height; ii++) {
// memcpy(oBuff, nms_in + ii * (width >> 2), width << 1);
// clang-format off
        #pragma HLS loop_tripcount min=HEIGHT max=HEIGHT
        // clang-format on
        for (int k = 0; k < ram_row_depth; k++) {
// clang-format off
            #pragma HLS pipeline
            #pragma HLS loop_tripcount min=RAM_ROW_DEPTH max=RAM_ROW_DEPTH
            // clang-format on
            oBuff[k] = _src.read((ii * ram_row_depth) + k);
        }

        ap_uint<3> id = 0;
        ap_uint<9> pixel = 0;
    WR_FIN_PIPE:
        for (int j = 0, bit = 0; j < width / 8; j++, bit += 16) {
// clang-format off
            #pragma HLS loop_tripcount min=WIDTH/8 max=WIDTH/8
            #pragma HLS pipeline
            // clang-format on
            if (id == 4) {
                id = 0;
                pixel++;
                bit = 0;
            }
            for (int k = 0, l = 0; k < 16; k += 2, l += 8) {
                ap_uint<2> pix = oBuff[pixel].range(bit + k + 1, bit + k);
                if (pix == 3)
                    oRegF[0].range(l + 7, l) = 255;
                else
                    oRegF[0].range(l + 7, l) = 0;
            }
            id++;
            _dst.write((ii * width / 8 + j), oRegF[0]);
        }
    }
}

template <int SRC_T, int DST_T, int ROWS, int COLS, int NPC_SRC, int NPC_DST, bool USE_URAM = false>
void EdgeTracing(xf::cv::Mat<SRC_T, ROWS, COLS, NPC_SRC>& _src, xf::cv::Mat<DST_T, ROWS, COLS, NPC_DST>& _dst) {
// clang-format off
    #pragma HLS INLINE
    // clang-format on
    xfEdgeTracing<SRC_T, DST_T, NPC_SRC, NPC_DST, ROWS, COLS, USE_URAM>(_dst, _src, _dst.rows, _dst.cols);
}

} // namespace cv
} // namespace xf
#endif
