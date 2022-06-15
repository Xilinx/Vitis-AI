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

#ifndef __XF_WARP_TRANSFORM__
#define __XF_WARP_TRANSFORM__
#include "ap_int.h"
#include "hls_stream.h"
#include "assert.h"
#include "common/xf_common.hpp"

// Number of fractional bits used for interpolation
#define INTER_BITS 5
#define MAX_BITS(x, y) (((x) > (y)) ? (x) : (y))
#define INTER_TAB_SIZE (1 << INTER_BITS)
#define INTER_SCALE 1.f / INTER_TAB_SIZE

#define AB_BITS MAX_BITS(10, INTER_BITS)
#define AB_SCALE (1 << AB_BITS)
// Number of bits used to linearly interpolate
#define INTER_REMAP_COEF_BITS 15
#define INTER_REMAP_COEF_SCALE (1 << INTER_REMAP_COEF_BITS)
#define ROUND_DELTA (1 << (AB_BITS - INTER_BITS - 1))
typedef float image_comp;

namespace xf {
namespace cv {

// function to store the image in 4 memories of a combined size of (0:STORE_LINES-1,0:COLS-1)
template <int COLS, int STORE_LINES, int DEPTH, int NPC>
void store_EvOd_image1(XF_TNAME(DEPTH, NPC) in_pixel,
                       ap_uint<16> i,
                       ap_uint<16> j,
                       XF_TNAME(DEPTH, NPC) store1_pt_2EvR_EvC[(STORE_LINES >> 2)][COLS],
                       XF_TNAME(DEPTH, NPC) store1_pt_2OdR_EvC[(STORE_LINES >> 2)][COLS],
                       XF_TNAME(DEPTH, NPC) store1_pt_2EvR_OdC[(STORE_LINES >> 2)][COLS],
                       XF_TNAME(DEPTH, NPC) store1_pt_2OdR_OdC[(STORE_LINES >> 2)][COLS]) {
// clang-format off
    #pragma HLS INLINE
    // clang-format on
    //	const int COLS = COLS;
    //	const int STORE_LINES = Num_Store_Rows;
    // finding if i and j are even or odd to access the appropriate memory
    bool i_a1 = (i & 0x00000002) >> 1;
    bool i_a = i & 0x00000001;
    bool j_a = j & 0x00000001;
    ap_uint<16> I = ap_uint<16>(i >> 2);
    ap_uint<16> J = 0;
    if (i_a1) {
        J = ap_uint<16>(j >> 1) + (COLS >> 1);
    } else {
        J = (j >> 1);
    }
    switch (j_a << 1 | i_a) {
        case 0:
            store1_pt_2EvR_EvC[I][J] = in_pixel;
            break;
        case 1:
            store1_pt_2OdR_EvC[I][J] = in_pixel;
            break;
        case 2:
            store1_pt_2EvR_OdC[I][J] = in_pixel;
            break;
        case 3:
            store1_pt_2OdR_OdC[I][J] = in_pixel;
            break;
    }
};

template <int COLS, int PLANES, int STORE_LINES, int DEPTH, int NPC>
XF_TNAME(DEPTH, NPC)
retrieve_EvOd_image1(int i,
                     int j,
                     XF_TNAME(DEPTH, NPC) store1_pt_2EvR_EvC[(STORE_LINES >> 2)][COLS],
                     XF_TNAME(DEPTH, NPC) store1_pt_2OdR_EvC[(STORE_LINES >> 2)][COLS],
                     XF_TNAME(DEPTH, NPC) store1_pt_2EvR_OdC[(STORE_LINES >> 2)][COLS],
                     XF_TNAME(DEPTH, NPC) store1_pt_2OdR_OdC[(STORE_LINES >> 2)][COLS]) {
// clang-format off
    #pragma HLS INLINE
    // clang-format on
    //	const int COLS = COLS;
    //	const int STORE_LINES = Num_Store_Rows;
    // finding if i and j are even or odd to access the appropriate memory
    XF_TNAME(DEPTH, NPC) return_val = 0;

    // temporary variables to compute the indices for memory access
    int I, J, temp1;

    // snapping the row number from 0:STORE_LINES-1 and wrapping around the indices
    // for i>STORE_LINES and i<0
    temp1 = i > (STORE_LINES - 1) ? (i - STORE_LINES) : ((i < 0) ? (i + STORE_LINES) : i);

    bool i_a1 = (temp1 & 0x00000002) >> 1;
    bool i_a = temp1 & 0x00000001;
    bool j_a = j & 0x00000001;
    I = temp1 >> 2;
    if (i_a1) {
        J = (j >> 1) + (COLS >> 1);
    } else {
        J = (j >> 1);
    }

    switch (j_a << 1 | i_a) {
        case 0:
            return_val = store1_pt_2EvR_EvC[I][J];
            break;
        case 1:
            return_val = store1_pt_2OdR_EvC[I][J];
            break;
        case 2:
            return_val = store1_pt_2EvR_OdC[I][J];
            break;
        case 3:
            return_val = store1_pt_2OdR_OdC[I][J];
            break;
    }
    return return_val;
};
// function to store the image in 4 memories of a combined size of (0:STORE_LINES-1,0:COLS-1)
template <int COLS, int PLANES, int STORE_LINES, int DEPTH, int NPC>
XF_TNAME(DEPTH, NPC)
retrieve_EvOd_image4x1(int i,
                       int j,
                       int A,
                       int B,
                       int C,
                       int D,
                       XF_TNAME(DEPTH, NPC) store1_pt_2EvR_EvC[(STORE_LINES >> 2)][COLS],
                       XF_TNAME(DEPTH, NPC) store1_pt_2OdR_EvC[(STORE_LINES >> 2)][COLS],
                       XF_TNAME(DEPTH, NPC) store1_pt_2EvR_OdC[(STORE_LINES >> 2)][COLS],
                       XF_TNAME(DEPTH, NPC) store1_pt_2OdR_OdC[(STORE_LINES >> 2)][COLS]) {
// clang-format off
    #pragma HLS INLINE
    // clang-format on
    //	const int COLS = COLS;
    //	const int STORE_LINES = Num_Store_Rows;
    // inlin-ing the function to enable the tool to pipeline the memory accesses
    // finding the LSB of i and j to find even/odd conditions for memory reads

    // output value computed in fixed point (17,15)
    ap_uint<32> op_val = 0;
    int op_val1 = 0;
    // temporary variables to compute the indices for 4 memories for linear interpolation
    // computing i/2 (i+1)/2 j/2 and (j+1)/2 to access the 4 memories
    int I, J, I1, J1, temp1, temp2, Ja, Ja1;

    // snapping the row number from 0:STORE_LINES-1 and wrapping around the indices
    // for i>STORE_LINES and i<0
    temp1 = (i > (STORE_LINES - 1)) ? (i - STORE_LINES) : ((i < 0) ? (i + STORE_LINES) : i);

    // snapping the row number from 0:STORE_LINES-1 and wrapping around the indices
    // for i+1>STORE_LINES and i+1<0
    temp2 = ((i + 1) > (STORE_LINES - 1)) ? (i + 1 - STORE_LINES) : ((i + 1 < 0) ? (i + 1 + STORE_LINES) : i + 1);
    // dividing the indices by 2 to get the indices for the 4 memories
    XF_TNAME(DEPTH, NPC) px00 = 0, px01 = 0, px10 = 0, px11 = 0;

    ap_uint<2> i_a1 = (temp1 & 0x00000003);
    ap_uint<2> i_a2 = (temp2 & 0x00000003);
    bool i_a = i & 0x00000001;
    bool j_a = j & 0x00000001;
    I = temp1 >> 2;
    I1 = temp2 >> 2;

    J = j >> 1;
    J1 = (j + 1) >> 1;
    Ja = (j >> 1) + (COLS >> 1);
    Ja1 = ((j + 1) >> 1) + (COLS >> 1);
    int EvR_EvC_colAddr, EvR_OdC_colAddr, OdR_EvC_colAddr, OdR_OdC_colAddr;
    int EvR_rowAddr, OdR_rowAddr;

    if ((i_a1 == 0) && (j_a == 0)) {
        EvR_EvC_colAddr = J;
        EvR_OdC_colAddr = J1;
        OdR_EvC_colAddr = J;
        OdR_OdC_colAddr = J1;
    } else if ((i_a1 == 0) && (j_a == 1)) {
        EvR_EvC_colAddr = J1;
        EvR_OdC_colAddr = J;
        OdR_EvC_colAddr = J1;
        OdR_OdC_colAddr = J;
    } else if ((i_a1 == 1) && (j_a == 0)) {
        EvR_EvC_colAddr = Ja;
        EvR_OdC_colAddr = Ja1;
        OdR_EvC_colAddr = J;
        OdR_OdC_colAddr = J1;
    } else if ((i_a1 == 1) && (j_a == 1)) {
        EvR_EvC_colAddr = Ja1;
        EvR_OdC_colAddr = Ja;
        OdR_EvC_colAddr = J1;
        OdR_OdC_colAddr = J;
    } else if ((i_a1 == 2) && (j_a == 0)) {
        EvR_EvC_colAddr = Ja;
        EvR_OdC_colAddr = Ja1;
        OdR_EvC_colAddr = Ja;
        OdR_OdC_colAddr = Ja1;
    } else if ((i_a1 == 2) && (j_a == 1)) {
        EvR_EvC_colAddr = Ja1;
        EvR_OdC_colAddr = Ja;
        OdR_EvC_colAddr = Ja1;
        OdR_OdC_colAddr = Ja;
    } else if ((i_a1 == 3) && (j_a == 0)) {
        EvR_EvC_colAddr = J;
        EvR_OdC_colAddr = J1;
        OdR_EvC_colAddr = Ja;
        OdR_OdC_colAddr = Ja1;
    } else {
        EvR_EvC_colAddr = J1;
        EvR_OdC_colAddr = J;
        OdR_EvC_colAddr = Ja1;
        OdR_OdC_colAddr = Ja;
    }

    if ((i_a1 == 0) || (i_a1 == 2)) {
        EvR_rowAddr = I;
        OdR_rowAddr = I1;
    } else {
        EvR_rowAddr = I1;
        OdR_rowAddr = I;
    }
    XF_TNAME(DEPTH, NPC) pop_bram_EvR_EvC = store1_pt_2EvR_EvC[EvR_rowAddr][EvR_EvC_colAddr];
    XF_TNAME(DEPTH, NPC) pop_bram_EvR_OdC = store1_pt_2EvR_OdC[EvR_rowAddr][EvR_OdC_colAddr];
    XF_TNAME(DEPTH, NPC) pop_bram_OdR_EvC = store1_pt_2OdR_EvC[OdR_rowAddr][OdR_EvC_colAddr];
    XF_TNAME(DEPTH, NPC) pop_bram_OdR_OdC = store1_pt_2OdR_OdC[OdR_rowAddr][OdR_OdC_colAddr];

    if (((i_a1 == 0) || (i_a1 == 2))) {
        if (j_a == 0) {
            px00 = pop_bram_EvR_EvC;
            px01 = pop_bram_EvR_OdC;
            px10 = pop_bram_OdR_EvC;
            px11 = pop_bram_OdR_OdC;
        } else {
            px00 = pop_bram_EvR_OdC;
            px01 = pop_bram_EvR_EvC;
            px10 = pop_bram_OdR_OdC;
            px11 = pop_bram_OdR_EvC;
        }
    } else {
        if (j_a == 0) {
            px00 = pop_bram_OdR_EvC;
            px01 = pop_bram_OdR_OdC;
            px10 = pop_bram_EvR_EvC;
            px11 = pop_bram_EvR_OdC;
        } else {
            px00 = pop_bram_OdR_OdC;
            px01 = pop_bram_OdR_EvC;
            px10 = pop_bram_EvR_OdC;
            px11 = pop_bram_EvR_EvC;
        }
    }
    for (ap_uint<10> c = 0, k = 0; c < PLANES; c++, k += 8) {
// clang-format off
        #pragma HLS UNROLL
        // clang-format on
        op_val1 = (A * px00.range(k + 7, k));
        op_val1 += (B * px01.range(k + 7, k));
        op_val1 += (C * px10.range(k + 7, k));
        op_val1 += (D * px11.range(k + 7, k));
        op_val.range(k + 7, k) = ((op_val1 + (1 << (INTER_REMAP_COEF_BITS - 1))) >> INTER_REMAP_COEF_BITS);
    }
    // returning the computed interpolated output after rounding off the op_val by adding 0.5
    // and shifting to right by INTER_REMAP_COEF_BITS
    return XF_TNAME(DEPTH, NPC)(op_val);
};

template <int COLS, int PLANES, int STORE_LINES, int DEPTH, int NPC>
void store_in_UramNN(XF_TNAME(DEPTH, NPC) in_pixel,
                     ap_uint<16> i,
                     ap_uint<16> j,
                     ap_uint<64> bufUram[PLANES][STORE_LINES][(COLS + 7) / 8]) {
// clang-format off
    #pragma HLS INLINE
    // clang-format on

    static XF_TNAME(DEPTH, NPC) sx8[8];
// clang-format off
    #pragma HLS ARRAY_PARTITION variable=sx8 complete dim=1
    // clang-format on
    sx8[j % 8] = in_pixel;
    for (int pl = 0, bit = 0; pl < PLANES; pl++, bit += 8)
        for (int k = 0; k < 8; k++) bufUram[pl][i][j / 8](k * 8 + 7, k * 8) = sx8[k](bit + 7, bit);
};

template <int COLS, int PLANES, int STORE_LINES, int DEPTH, int NPC>
void store_in_UramBL(hls::stream<XF_TNAME(DEPTH, NPC)>& input_image,
                     ap_uint<16> i,
                     ap_uint<16> j,
                     ap_uint<72> bufUram[PLANES][(STORE_LINES + 1) / 2][(COLS + 1) / 2],
                     short img_cols,
                     bool store_row,
                     bool store_col,
                     ap_uint<16> temppix[PLANES],
                     ap_uint<24> pixval[PLANES],
                     ap_uint<48> pixval_2[PLANES],
                     ap_uint<24> prev_pixval[PLANES]) {
// clang-format off
    #pragma HLS INLINE
    // clang-format on

    /*ap_uint<16> temppix_t[PLANES];//     = *temppix;
    ap_uint<24> pixval_t[PLANES];//      = *pixval;
    ap_uint<48> pixval_2_t[PLANES];//    = *pixval_2;
    ap_uint<24> prev_pixval_t[PLANES];// = *prev_pixval;

    for(int pl=0;pl<PLANES;pl++)
    {
// clang-format off
            #pragma HLS UNROLL
// clang-format on

    temppix_t[pl]= temppix[pl];
    pixval_t[pl]= pixval[pl];
    pixval_2_t[pl]= pixval_2[pl];
    prev_pixval_t[pl]= prev_pixval[pl];
    }*/

    ap_uint<24> lineBuf[PLANES][(COLS + 1) / 2];
// clang-format off
    #pragma HLS resource variable=lineBuf core=RAM_S2P_BRAM latency=1
    #pragma HLS ARRAY_PARTITION variable=lineBuf dim=1
    // clang-format on
    ap_int<16> i_hlf_mns1 = i / 2 - 1;
    i_hlf_mns1 = i_hlf_mns1 + (i_hlf_mns1 < 0 ? (STORE_LINES + 1) / 2 : 0);

    static XF_TNAME(DEPTH, NPC) in_pixel;
    if (j < img_cols) in_pixel = input_image.read();
    for (int pl = 0, bit = 0; pl < PLANES; pl++, bit += 8) {
// clang-format off
        #pragma HLS UNROLL
        // clang-format on
        if (store_col && (j != 0)) {
            pixval[pl].range(15, 0) = temppix[pl];
            pixval[pl].range(23, 16) = in_pixel.range(bit + 7, bit);

            if (store_row) {
                // Store every 3rd row in a buffer
                lineBuf[pl][(j / 2) - 1] = pixval[pl];
            } else {
                // Read the stored row and fill in
                prev_pixval[pl] = lineBuf[pl][(j / 2) - 1];
            }

            if (i != 0) {
                if (store_row) {
                    bufUram[pl][/*i_hlf_mns1*/ ((i - 1) / 2) % (STORE_LINES / 2)][(j / 2) - 1].range(71, 48) =
                        pixval[pl];
                } else {
                    pixval_2[pl].range(23, 0) = prev_pixval[pl];
                    pixval_2[pl].range(47, 24) = pixval[pl];
                    bufUram[pl][/*i_hlf_mns1*/ ((i - 1) / 2) % (STORE_LINES / 2)][(j / 2) - 1].range(47, 0) =
                        pixval_2[pl];
                }
            }
        }

        if (store_col) {
            temppix[pl].range(7, 0) = in_pixel.range(bit + 7, bit);
        } else {
            temppix[pl].range(15, 8) = in_pixel.range(bit + 7, bit);
        }

        /* temppix[pl] = temppix_t[pl];
        pixval[pl] =  pixval_t[pl];
         pixval_2[pl] = pixval_2_t[pl];
        prev_pixval[pl]  = prev_pixval_t[pl];*/
    }
};

template <int COLS, int PLANES, int STORE_LINES, int DEPTH, int NPC>
XF_TNAME(DEPTH, NPC)
retrieve_UramNN(int i, int j, ap_uint<64> bufUram[PLANES][STORE_LINES][(COLS + 7) / 8]) {
// clang-format off
    #pragma HLS INLINE
    // clang-format on

    i = i > (STORE_LINES - 1) ? (i - STORE_LINES) : ((i < 0) ? (i + STORE_LINES) : i);
    XF_TNAME(DEPTH, NPC) dx8[8];
// clang-format off
    #pragma HLS ARRAY_PARTITION variable=dx8 complete dim=1
    // clang-format on
    for (int pl = 0, bit = 0; pl < PLANES; pl++, bit += 8)
        for (int k = 0; k < 8; k++) dx8[k](bit + 7, bit) = bufUram[pl][i][j / 8](k * 8 + 7, k * 8);
    return dx8[j % 8];
};

template <int COLS, int PLANES, int STORE_LINES, int DEPTH, int NPC>
XF_TNAME(DEPTH, NPC)
retrieve_UramBL(
    int i, int j, int A, int B, int C, int D, ap_uint<72> bufUram[PLANES][(STORE_LINES + 1) / 2][(COLS + 1) / 2]) {
// clang-format off
    #pragma HLS INLINE
    // clang-format on

    i = (i > (STORE_LINES - 1)) ? (i - STORE_LINES) : ((i < 0) ? (i + STORE_LINES) : i);

    XF_TNAME(DEPTH, NPC) d3x3[9];
    XF_TNAME(DEPTH, NPC) op_val;
// clang-format off
    #pragma HLS ARRAY_PARTITION variable=d3x3 complete dim=0
    // clang-format on

    for (int pl = 0, k = 0; pl < PLANES; pl++, k += 8) {
        for (int k = 0; k < 9; k++) d3x3[k] = bufUram[pl][i / 2][j / 2](k * 8 + 7, k * 8);
        XF_TNAME(DEPTH, NPC) const px00 = d3x3[(i % 2) * 3 + j % 2];
        XF_TNAME(DEPTH, NPC) const px01 = d3x3[(i % 2) * 3 + j % 2 + 1];
        XF_TNAME(DEPTH, NPC) const px10 = d3x3[(i % 2 + 1) * 3 + j % 2];
        XF_TNAME(DEPTH, NPC) const px11 = d3x3[(i % 2 + 1) * 3 + j % 2 + 1];

        int const op_val1 = (A * px00) + (B * px01) + (C * px10) + (D * px11);

        op_val.range(k + 7, k) = ((op_val1 + (1 << (INTER_REMAP_COEF_BITS - 1))) >> INTER_REMAP_COEF_BITS);
        // returning the computed interpolated output after rounding off the op_val by adding 0.5
        // and shifting to right by INTER_REMAP_COEF_BITS
    }
    return XF_TNAME(DEPTH, NPC)(op_val);
};

// AK(ZoTech): rounding function to substitute one from math.h, consuming 2 BRAMs per call; not used as it is not
// bitexact with the math.h.
// template<class T>
// int round(T x)
// {
// #pragma HLS INLINE
// 	return (x + (x>=T(0) ? T(0.5) : T(-0.5)));
// };

// AK(ZoTech): floor function to substitute one from math.h, consuming 2 BRAMs per call; not used as it is not
// synthesisable if biexact.
// template<class T>
// int floor(T x)
// {
// #pragma HLS INLINE
//     return (x - (x>=T(0) ? T(0) : T(1)-std::numeric_limits<T>::epsilon() ));
// };

template <int NPC,
          int ROWS,
          int COLS,
          int PLANES,
          int DEPTH,
          int STORE_LINES,
          int START_ROW,
          int TRANSFORM,
          bool INTERPOLATION_TYPE,
          bool USE_URAM>
int xFwarpTransformKernel(hls::stream<XF_TNAME(DEPTH, NPC)>& input_image,
                          hls::stream<XF_TNAME(DEPTH, NPC)>& output_image,
                          float P_matrix[9],
                          short img_rows,
                          short img_cols) {
// clang-format off
    #pragma HLS INLINE
// clang-format on
// dividing memory (0:STORE_LINES-1,0:COLS-1) to ensure that the same memory
// is not accesses more than once per iteration
// removing intra and inter read dependencies between the reads and writes of memories
// declaring them as true port brams
#ifndef __SYNTHESIS__
    assert(((img_rows <= ROWS) && (img_cols <= COLS)) && "ROWS and COLS should be greater than input image");

    if (TRANSFORM == 0) {
        assert(((P_matrix[6] == 0) && (P_matrix[7] == 0) && (P_matrix[8] == 0)) &&
               "Third row of the transformation matrix must be 0s for Affine");
    }
#endif
    XF_TNAME(DEPTH, NPC) store1_pt_2EvR_EvC[((STORE_LINES + 3) >> 2)][COLS];
    XF_TNAME(DEPTH, NPC) store1_pt_2EvR_OdC[((STORE_LINES + 3) >> 2)][COLS];
    XF_TNAME(DEPTH, NPC) store1_pt_2OdR_EvC[((STORE_LINES + 3) >> 2)][COLS];
    XF_TNAME(DEPTH, NPC) store1_pt_2OdR_OdC[((STORE_LINES + 3) >> 2)][COLS];
// clang-format off
    #pragma HLS ARRAY_PARTITION variable=store1_pt_2EvR_EvC complete dim=1
    #pragma HLS ARRAY_PARTITION variable=store1_pt_2EvR_OdC complete dim=1
    #pragma HLS ARRAY_PARTITION variable=store1_pt_2OdR_EvC complete dim=1
    #pragma HLS ARRAY_PARTITION variable=store1_pt_2OdR_OdC complete dim=1
    #pragma HLS RESOURCE variable=store1_pt_2EvR_EvC core=RAM_T2P_BRAM
    #pragma HLS RESOURCE variable=store1_pt_2EvR_OdC core=RAM_T2P_BRAM
    #pragma HLS RESOURCE variable=store1_pt_2OdR_EvC core=RAM_T2P_BRAM
    #pragma HLS RESOURCE variable=store1_pt_2OdR_OdC core=RAM_T2P_BRAM
    #pragma HLS DEPENDENCE variable=store1_pt_2EvR_EvC inter false
    #pragma HLS DEPENDENCE variable=store1_pt_2EvR_OdC inter false
    #pragma HLS DEPENDENCE variable=store1_pt_2OdR_EvC inter false
    #pragma HLS DEPENDENCE variable=store1_pt_2OdR_OdC inter false
    #pragma HLS DEPENDENCE variable=store1_pt_2EvR_EvC intra false
    #pragma HLS DEPENDENCE variable=store1_pt_2EvR_OdC intra false
    #pragma HLS DEPENDENCE variable=store1_pt_2OdR_EvC intra false
    #pragma HLS DEPENDENCE variable=store1_pt_2OdR_OdC intra false
    // clang-format on

    // URAM based storages
    ap_uint<64> bufUramNN[PLANES][STORE_LINES][(COLS + 7) / 8];
// clang-format off
    #pragma HLS RESOURCE   variable=bufUramNN core= RAM_T2P_URAM latency=2
    #pragma HLS dependence variable=bufUramNN inter false
    #pragma HLS ARRAY_PARTITION variable=bufUramNN complete dim=1
    // clang-format on
    // URAM storage garnularity for BL inerpolation is 3x3-pel block in 2x2-pel picture grid, it fits to one URAM word
    ap_uint<72> bufUramBL[PLANES][(STORE_LINES + 1) / 2][(COLS + 1) / 2];
// clang-format off
    #pragma HLS RESOURCE   variable=bufUramBL core=RAM_T2P_URAM latency=2
    #pragma HLS dependence variable=bufUramBL inter false
    #pragma HLS ARRAY_PARTITION variable=bufUramBL complete dim=1
    // clang-format on
    // additional separation of URAM buffer to single URAMs to exclude their built-in cascading and thus limited timing
    // due to inability of VHLS to schedule built-in cascade register (OREG_CAS)

    // varables for loop counters
    ap_uint<16> i = 0, j = 0, k = 0, l = 0, m = 0, n = 0, p = 0;

    // copying transformation matrix to a local variable
    float R[3][3];
// clang-format off
    #pragma HLS ARRAY_PARTITION variable=R complete dim=1
    #pragma HLS ARRAY_PARTITION variable=R complete dim=2
// clang-format on
COPY_MAT1:
    for (i = 0; i < 3; i++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=3 max=3 avg=3
    // clang-format on
    COPY_MAT2:
        for (j = 0; j < 3; j++) {
// clang-format off
            #pragma HLS LOOP_TRIPCOUNT min=3 max=3 avg=3
            // clang-format on
            R[i][j] = float(P_matrix[i * 3 + j]);
        }
    }
    // output of the transformation matrix multiplication
    // partitioning the array to avoid memory read/write operations
    image_comp output_vec[3];
// clang-format off
    #pragma HLS ARRAY_PARTITION variable=output_vec complete dim=1
    // clang-format on

    // variables for image indices, they can be negative
    // and can go out of bounds of the input image indices
    int I = 0, J = 0, I1 = 0;

    // variables to compute the transformations
    int X = 0, Y = 0, X_t = 0, Y_t = 0, X_t1 = 0, Y_t1 = 0, round_delta = 0;

    // variables to compute the interpolation weights
    int A = 0, B = 0, C = 0, D = 0;

    // variable to store the output of the memory read/interpolation
    // to avoid multiple function calls in conditional statements
    XF_TNAME(DEPTH, NPC) output_value = 0;

    // op_val for storing the interpolated pixel value.
    // a, b for finding the fractional pixel values
    XF_TNAME(DEPTH, NPC) op_val = 0;

    bool store_col = 1;
    bool store_row = 1;

    /*ap_uint<16> temppix     = 0;
    ap_uint<24> pixval      = 0;
    ap_uint<48> pixval_2    = 0;
    ap_uint<24> prev_pixval = 0;*/

    ap_uint<16> temppix[PLANES];
    ap_uint<24> pixval[PLANES];
    ap_uint<48> pixval_2[PLANES];
    ap_uint<24> prev_pixval[PLANES];

    for (int pl = 0; pl < PLANES; pl++) {
// clang-format off
        #pragma HLS UNROLL
        // clang-format on
        temppix[pl] = 0;
        pixval[pl] = 0;
        pixval_2[pl] = 0;
        prev_pixval[pl] = 0;
    }

// main loop
MAIN_ROWS:
    for (i = 0; i < (img_rows + START_ROW); i++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=1 max=ROWS
        // clang-format on
        // Initialize for every row
        store_col = 1;
    MAIN_COLS:
        for (j = 0; j < (img_cols + 1); j++) {
// clang-format off
            #pragma HLS LOOP_TRIPCOUNT min=1 max=COLS
            #pragma HLS PIPELINE II=1
            #pragma HLS LOOP_FLATTEN off
            // clang-format on

            // condition to store the input image in the image buffers
            // only until the number of rows are streamed in
            if (i < img_rows) {
                // compute n to wrap the buffer writes
                // to 0 once STORE_LINES rows are filled
                if (!(USE_URAM && INTERPOLATION_TYPE == 1)) {
                    n = i - l;
                    if (n >= STORE_LINES) {
                        l = l + STORE_LINES;
                    }
                }
                // function to store the input image stream to
                // a buffer of size STORE_LINES rows
                // computing i-l to snap the writes to STORE_LINES size buffer
                if (USE_URAM)
                    if (INTERPOLATION_TYPE)
                        store_in_UramBL<COLS, PLANES, STORE_LINES, DEPTH, NPC>(input_image, i - l, j, bufUramBL,
                                                                               img_cols, store_row, store_col, temppix,
                                                                               pixval, pixval_2, prev_pixval);
                    else {
                        if (j < img_cols)
                            store_in_UramNN<COLS, PLANES, STORE_LINES, DEPTH, NPC>(input_image.read(), i - l, j,
                                                                                   bufUramNN);
                    }
                else if (j < img_cols)
                    store_EvOd_image1<COLS, STORE_LINES, DEPTH, NPC>(input_image.read(), i - l, j, store1_pt_2EvR_EvC,
                                                                     store1_pt_2EvR_OdC, store1_pt_2OdR_EvC,
                                                                     store1_pt_2OdR_OdC);
                store_col = !(store_col);
            }

            // condition to compute and stream out the output image
            // after START_ROW number of rows
            if (i >= START_ROW && j < img_cols) {
                // computing k from i to index the output image from 0
                k = i - (START_ROW);

                if (TRANSFORM == 1) {
                    // transforming the output coordinate (kth row, jth column)
                    // to find the input coordinate using the transform matrix R
                    // destination X coordinate is output_vec[0][0]
                    // destination Y coordinate is output_vec[0][1]
                    // destination Z coordinate is output_vec[0][2]
                    output_vec[0] = image_comp(R[0][0]) * (j) + image_comp(R[0][1]) * (k) + image_comp(R[0][2]);
                    output_vec[1] = image_comp(R[1][0]) * (j) + image_comp(R[1][1]) * (k) + image_comp(R[1][2]);
                    output_vec[2] = image_comp(R[2][0]) * (j) + image_comp(R[2][1]) * (k) + image_comp(R[2][2]);

                    // find the inverse of the Z element of the transform
                    // and make it 0 if the z element is 0 to evade divide by 0
                    if (INTERPOLATION_TYPE == 0) {
                        output_vec[2] = (output_vec[2] != 0.0f) ? (1.0f / output_vec[2]) : 0.0f;
                    } else {
                        output_vec[2] = (output_vec[2] != 0.0f) ? (INTER_TAB_SIZE / output_vec[2]) : 0.0f;
                    }
                    // find the X and Y indices of the destination by dividing
                    // with the Z element of the transform
                    X = round(output_vec[0] * output_vec[2]);
                    Y = round(output_vec[1] * output_vec[2]);
                } else {
                    if (INTERPOLATION_TYPE == 0) {
                        X_t = round(image_comp(R[0][0]) * ((unsigned short)(j) << (AB_BITS)));
                        X_t1 = round((image_comp(R[0][1]) * k + image_comp(R[0][2])) * AB_SCALE);
                        Y_t = round(image_comp(R[1][0]) * ((unsigned short)(j) << (AB_BITS)));
                        Y_t1 = round((image_comp(R[1][1]) * k + image_comp(R[1][2])) * AB_SCALE);
                        round_delta = (AB_SCALE >> 1);
                        X = X_t + X_t1 + round_delta;
                        Y = Y_t + Y_t1 + round_delta;
                        X = round(X >> (AB_BITS));
                        Y = round(Y >> (AB_BITS));
                    } else {
                        X_t = round(image_comp(R[0][0]) * (int(j) << (AB_BITS)));
                        X_t1 = round((image_comp(R[0][1]) * k + image_comp(R[0][2])) * AB_SCALE);
                        Y_t = round(image_comp(R[1][0]) * (int(j) << (AB_BITS)));
                        Y_t1 = round((image_comp(R[1][1]) * k + image_comp(R[1][2])) * AB_SCALE);
                        X = X_t + X_t1 + ROUND_DELTA;
                        X = X >> (AB_BITS - INTER_BITS);
                        Y = Y_t + Y_t1 + ROUND_DELTA;
                        Y = Y >> (AB_BITS - INTER_BITS);
                    }
                }

                if (INTERPOLATION_TYPE == 0) {
                    I = Y;
                    J = X;
                } else {
                    // finding the integer part by shifting to the right
                    // by the number of fractional bits
                    I = Y >> INTER_BITS;
                    J = X >> INTER_BITS;

                    // finding the fractional part of the indices in fixed point
                    short a = Y & (INTER_TAB_SIZE - 1);
                    short b = X & (INTER_TAB_SIZE - 1);

                    // finding the fractional part of the indices in floating point
                    float taby = (float(INTER_REMAP_COEF_SCALE) / INTER_TAB_SIZE) * a;
                    float tabx = (1.0f / INTER_TAB_SIZE) * b;

                    // finding the coefficients to multiply with the 4 pixels for interpolation
                    //(I,J)*A + (I,J+1)*B + (I+1,J)*C, (I+1,J+1)*D
                    // converting to fixed point with 15 fractional bits
                    A = floor((float(INTER_REMAP_COEF_SCALE) - taby) * (1.0f - tabx));
                    B = floor((float(INTER_REMAP_COEF_SCALE) - taby) * tabx);
                    C = floor(taby * (1.0f - tabx));
                    D = floor(taby * tabx);
                }

                // compute k-m to wrap the buffer reads
                // to 0 once N_Store rows are reads
                if (k - m >= STORE_LINES) {
                    m = m + STORE_LINES;
                }

                // calling the read function with interpolation
                if ((J >= 0) && (J < img_cols - int(INTERPOLATION_TYPE)) && (I >= 0) &&
                    (I < img_rows - int(INTERPOLATION_TYPE))) {
                    // computing the row index for the stored STORE_LINES rows
                    I1 = I - m;
                    if (INTERPOLATION_TYPE == 0) {
                        if (USE_URAM)
                            op_val = retrieve_UramNN<COLS, PLANES, STORE_LINES, DEPTH, NPC>(I1, J, bufUramNN);
                        else
                            op_val = retrieve_EvOd_image1<COLS, PLANES, STORE_LINES, DEPTH, NPC>(
                                I1, J, store1_pt_2EvR_EvC, store1_pt_2EvR_OdC, store1_pt_2OdR_EvC, store1_pt_2OdR_OdC);
                    } else {
                        // calling the read function with interpolation
                        if (USE_URAM)
                            op_val =
                                retrieve_UramBL<COLS, PLANES, STORE_LINES, DEPTH, NPC>(I1, J, A, B, C, D, bufUramBL);
                        else
                            op_val = retrieve_EvOd_image4x1<COLS, PLANES, STORE_LINES, DEPTH, NPC>(
                                I1, J, A, B, C, D, store1_pt_2EvR_EvC, store1_pt_2EvR_OdC, store1_pt_2OdR_EvC,
                                store1_pt_2OdR_OdC);
                    }
                } else {
                    // change this to an input if the border
                    op_val = 0;
                }
                // streaming out the computed output value and incrementing the out address pointer
                output_image.write(op_val);
            }
        }
        store_row = !(store_row);
    }

    return 0;
};

template <int STORE_LINES,
          int START_ROW,
          int TRANSFORM,
          bool INTERPOLATION_TYPE,
          int TYPE,
          int ROWS,
          int COLS,
          int NPC,
          bool USE_URAM = false>
void warpTransformKrnl(xf::cv::Mat<TYPE, ROWS, COLS, NPC>& _src_mat,
                       xf::cv::Mat<TYPE, ROWS, COLS, NPC>& _dst_mat,
                       float P_matrix[9]) {
// clang-format off
#pragma HLS DATAFLOW
    // clang-format on
    hls::stream<XF_TNAME(TYPE, NPC)> in_stream;
    hls::stream<XF_TNAME(TYPE, NPC)> out_stream;

    for (int i = 0; i < _src_mat.rows; i++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=1 max=ROWS
        // clang-format on
        for (int j = 0; j<(_src_mat.cols)>> XF_BITSHIFT(NPC); j++) {
// clang-format off
            #pragma HLS LOOP_TRIPCOUNT min=1 max=COLS/NPC
            #pragma HLS PIPELINE
            // clang-format on
            // in_stream.write( *(_src_mat.data + i*(_src_mat.cols>>XF_BITSHIFT(NPC)) +j) );
            in_stream.write(_src_mat.read(i * (_src_mat.cols >> XF_BITSHIFT(NPC)) + j));
        }
    }

    xFwarpTransformKernel<NPC, ROWS, COLS, XF_CHANNELS(TYPE, NPC), TYPE, STORE_LINES, START_ROW, TRANSFORM,
                          INTERPOLATION_TYPE, USE_URAM>(in_stream, out_stream, P_matrix, _src_mat.rows, _src_mat.cols);

    for (int i = 0; i < _dst_mat.rows; i++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=1 max=ROWS
        // clang-format on
        for (int j = 0; j<(_dst_mat.cols)>> XF_BITSHIFT(NPC); j++) {
// clang-format off
            #pragma HLS LOOP_TRIPCOUNT min=1 max=COLS/NPC
            #pragma HLS PIPELINE
            // clang-format on
            //*(_dst_mat.data + i*(_dst_mat.cols>>XF_BITSHIFT(NPC)) +j) = out_stream.read();
            _dst_mat.write(i * (_dst_mat.cols >> XF_BITSHIFT(NPC)) + j, out_stream.read());
            //_dst_mat.write(i*(_dst_mat.cols>>XF_BITSHIFT(NPC)) +j,in_stream.read());
        }
    }
}

template <int STORE_LINES,
          int START_ROW,
          int TRANSFORM,
          bool INTERPOLATION_TYPE,
          int TYPE,
          int ROWS,
          int COLS,
          int NPC,
          bool USE_URAM = false>
void warpTransform(xf::cv::Mat<TYPE, ROWS, COLS, NPC>& _src_mat,
                   xf::cv::Mat<TYPE, ROWS, COLS, NPC>& _dst_mat,
                   float P_matrix[9]) {
// clang-format off
    #pragma HLS INLINE OFF

	warpTransformKrnl<STORE_LINES, START_ROW, TRANSFORM, INTERPOLATION_TYPE, TYPE, ROWS, COLS, NPC,
                          USE_URAM>(_src_mat, _dst_mat, P_matrix);
}
} // namespace cv
} // namespace xf
#endif
