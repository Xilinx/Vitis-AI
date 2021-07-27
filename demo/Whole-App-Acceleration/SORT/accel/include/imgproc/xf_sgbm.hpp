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

#ifndef _XF_SGBM_HPP_
#define _XF_SGBM_HPP_

#ifndef __cplusplus
#error C++ is needed to include this header
#endif

typedef unsigned short uint16_t;

typedef unsigned int uint32_t;

#include "hls_stream.h"
#include "common/xf_common.hpp"
#include "common/xf_utility.hpp"

#define MAX_UCHAR 255

namespace xf {
namespace cv {

/**
 * xFComputeTransform5x5: 5x5 transform
 */
template <int DEPTH_SRC, int DEPTH_DST>
XF_PTNAME(DEPTH_DST)
xFComputeTransform5x5(XF_PTNAME(DEPTH_SRC) src_buf[5][5]) {
// clang-format off
    #pragma HLS INLINE off
    // clang-format on

    XF_PTNAME(DEPTH_SRC) target = src_buf[2][2];
    XF_PTNAME(DEPTH_DST) val = 0;

    int idx = 0;
    for (int i = 0; i < 5; i++) {
// clang-format off
        #pragma HLS UNROLL
        // clang-format on
        for (int j = 0; j < 5; j++) {
// clang-format off
            #pragma HLS UNROLL
            // clang-format on

            XF_PTNAME(DEPTH_SRC) ref = src_buf[i][j];
            if ((i != 2) || (j != 2)) {
                val.range(23 - idx, 23 - idx) = (ref < target) ? 1 : 0;
                idx++;
            }
        }
    }

    return val;
}

template <int ROWS, int COLS, int DEPTH_SRC, int DEPTH_DST, int NPC, int WORDWIDTH_SRC, int WORDWIDTH_DST>
void xFProcessCensusTransform5x5(hls::stream<XF_SNAME(WORDWIDTH_SRC)>& _src_mat,
                                 hls::stream<XF_SNAME(WORDWIDTH_DST)>& _dst_mat,
                                 XF_SNAME(WORDWIDTH_SRC) buf[5][COLS],
                                 XF_PTNAME(DEPTH_SRC) src_buf[5][5],
                                 XF_PTNAME(DEPTH_DST) & CensusVal,
                                 uint16_t img_width,
                                 uint16_t img_height,
                                 ap_uint<13> row_ind,
                                 ap_uint<4> tp1,
                                 ap_uint<4> tp2,
                                 ap_uint<4> mid,
                                 ap_uint<4> bottom1,
                                 ap_uint<4> bottom2,
                                 ap_uint<13> row) {
// clang-format off
    #pragma HLS INLINE
    // clang-format on
    XF_SNAME(WORDWIDTH_SRC) buf0, buf1, buf2, buf3, buf4;

Col_Loop:
    for (ap_uint<13> col = 0; col < img_width; col++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=COLS max=COLS
        #pragma HLS pipeline
        // clang-format on
        if (row < img_height)
            buf[row_ind][col] = _src_mat.read();
        else
            buf[bottom2][col] = 0;

        src_buf[0][4] = buf[tp1][col];
        src_buf[1][4] = buf[tp2][col];
        src_buf[2][4] = buf[mid][col];
        src_buf[3][4] = buf[bottom1][col];
        src_buf[4][4] = buf[bottom2][col];

        CensusVal = xFComputeTransform5x5<DEPTH_SRC, DEPTH_DST>(src_buf);

        for (ap_uint<4> i = 0; i < 5; i++) {
            for (ap_uint<4> j = 0; j < 4; j++) {
// clang-format off
                #pragma HLS unroll
                // clang-format on
                src_buf[i][j] = src_buf[i][j + 1];
            }
        }

        if (col >= 2) {
            _dst_mat.write(CensusVal);
        }
    } // Col_Loop
}

/**
 * xFCensus5x5 : Compute a Box filter of size 5x5 over a window of the input image.
 * Inputs : _src_mat --> input image of type XF_8U, XF_16U or XF_16S
 * Output : _dst_mat --> output image of input type
 */
template <int ROWS, int COLS, int DEPTH_SRC, int DEPTH_DST, int NPC, int WORDWIDTH_SRC, int WORDWIDTH_DST>
void xFCensus5x5(hls::stream<XF_SNAME(WORDWIDTH_SRC)>& _src_mat,
                 hls::stream<XF_SNAME(WORDWIDTH_DST)>& _dst_mat,
                 uint16_t img_height,
                 uint16_t img_width) {
    ap_uint<13> row_ind, row, col;
    ap_uint<4> tp1, tp2, mid, bottom1, bottom2;
    XF_PTNAME(DEPTH_DST) censusVal;

    // Temporary buffers to hold image data from five rows
    XF_PTNAME(DEPTH_SRC) src_buf[5][5];
// clang-format off
    #pragma HLS ARRAY_PARTITION variable=src_buf complete dim=0
    // clang-format on

    // Temporary buffer to hold image data from five rows
    XF_SNAME(WORDWIDTH_SRC) buf[5][COLS];
// clang-format off
    #pragma HLS RESOURCE variable=buf core=RAM_S2P_BRAM
    #pragma HLS ARRAY_PARTITION variable=buf complete dim=1
    // clang-format on

    row_ind = 2;

Clear_Row_Loop:
    for (col = 0; col < img_width; col++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=COLS max=COLS
        #pragma HLS pipeline
        // clang-format on
        buf[0][col] = 0;
        buf[1][col] = 0;
        buf[row_ind][col] = _src_mat.read();
    }
    row_ind++;

Read_Row2_Loop:
    for (col = 0; col < img_width; col++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=COLS max=COLS
        #pragma HLS pipeline
        // clang-format on
        buf[row_ind][col] = _src_mat.read();
    }
    row_ind++;

Row_Loop:
    for (row = 2; row < img_height + 2; row++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=ROWS max=ROWS
        // clang-format on

        // modify the buffer indices to re use
        if (row_ind == 4) {
            tp1 = 0;
            tp2 = 1;
            mid = 2;
            bottom1 = 3;
            bottom2 = 4;
        } else if (row_ind == 0) {
            tp1 = 1;
            tp2 = 2;
            mid = 3;
            bottom1 = 4;
            bottom2 = 0;
        } else if (row_ind == 1) {
            tp1 = 2;
            tp2 = 3;
            mid = 4;
            bottom1 = 0;
            bottom2 = 1;
        } else if (row_ind == 2) {
            tp1 = 3;
            tp2 = 4;
            mid = 0;
            bottom1 = 1;
            bottom2 = 2;
        } else if (row_ind == 3) {
            tp1 = 4;
            tp2 = 0;
            mid = 1;
            bottom1 = 2;
            bottom2 = 3;
        }

        for (int i = 0; i < 5; i++) {
// clang-format off
            #pragma HLS UNROLL
            // clang-format on
            for (int j = 0; j < 4; j++) {
// clang-format off
                #pragma HLS UNROLL
                // clang-format on
                src_buf[i][j] = 0;
            }
        }

        xFProcessCensusTransform5x5<ROWS, COLS, DEPTH_SRC, DEPTH_DST, NPC, WORDWIDTH_SRC, WORDWIDTH_DST>(
            _src_mat, _dst_mat, buf, src_buf, censusVal, img_width, img_height, row_ind, tp1, tp2, mid, bottom1,
            bottom2, row);

        for (int i = 0; i < 5; i++) {
// clang-format off
            #pragma HLS UNROLL
            // clang-format on
            src_buf[i][4] = 0;
        }
// clang-format off
        #pragma HLS ALLOCATION function instances=xFComputeTransform5x5<DEPTH_SRC, DEPTH_DST> limit=1
        // clang-format on
        censusVal = xFComputeTransform5x5<DEPTH_SRC, DEPTH_DST>(src_buf);
        _dst_mat.write(censusVal);

        for (ap_uint<4> i = 0; i < 5; i++) {
            for (ap_uint<4> j = 0; j < 4; j++) {
// clang-format off
                #pragma HLS unroll
                // clang-format on
                src_buf[i][j] = src_buf[i][j + 1];
            }
        }
        for (int i = 0; i < 5; i++) {
// clang-format off
            #pragma HLS UNROLL
            // clang-format on
            src_buf[i][4] = 0;
        }

        censusVal = xFComputeTransform5x5<DEPTH_SRC, DEPTH_DST>(src_buf);
        _dst_mat.write(censusVal);

        row_ind++;

        if (row_ind == 5) {
            row_ind = 0;
        }
    } // Row_Loop
} // end of xFCensus5x5

template <int SIZE>
class xFMinSAD {
   public:
    template <typename T, typename T_idx>
    static void find(T a[SIZE], T_idx& loc, T& val) {
// clang-format off
        #pragma HLS INLINE
        #pragma HLS array_partition variable=a complete dim=0
        // clang-format on

        T a1[SIZE / 2];
        T a2[SIZE - SIZE / 2];

        for (int i = 0; i < SIZE / 2; i++) {
// clang-format off
            #pragma HLS UNROLL
            // clang-format on
            a1[i] = a[i];
        }
        for (int i = 0; i < SIZE - SIZE / 2; i++) {
// clang-format off
            #pragma HLS UNROLL
            // clang-format on
            a2[i] = a[i + SIZE / 2];
        }

        T_idx l1, l2;
        T v1, v2;
        xFMinSAD<SIZE / 2>::find(a1, l1, v1);
        xFMinSAD<SIZE - SIZE / 2>::find(a2, l2, v2);

        if (v2 < v1) {
            val = v2;
            loc = l2 + SIZE / 2;
        } else {
            val = v1;
            loc = l1;
        }
    }
};

template <>
class xFMinSAD<1> {
   public:
    template <typename T, typename T_idx>
    static void find(T a[1], T_idx& loc, T& val) {
// clang-format off
        #pragma HLS INLINE
        // clang-format on

        loc = 0;
        val = a[0];
    }
};

template <>
class xFMinSAD<2> {
   public:
    template <typename T, typename T_idx>
    static void find(T a[2], T_idx& loc, T& val) {
// clang-format off
        #pragma HLS INLINE
        #pragma HLS array_partition variable=a complete dim=0
        // clang-format on

        T_idx l1 = 0, l2 = 1;
        T v1 = a[0], v2 = a[1];
        if (v2 < v1) {
            val = v2;
            loc = l2;
        } else {
            val = v1;
            loc = l1;
        }
    }
};

/**
 * xFCensusTransformKernel : This function calls the transform operations depending upon the
 * window size. This acts as a wrapper function.
 */
template <int ROWS, int COLS, int DEPTH_SRC, int DEPTH_DST, int NPC, int WORDWIDTH_SRC, int WORDWIDTH_DST>
void xFCensusTransformKernel(hls::stream<XF_SNAME(WORDWIDTH_SRC)>& _src,
                             hls::stream<XF_SNAME(WORDWIDTH_DST)>& _dst,
                             uint8_t _window_size,
                             uint8_t _border_type,
                             uint16_t img_height,
                             uint16_t img_width) {
#ifndef _SYNTHESIS_
    assert(((_window_size == XF_FILTER_3X3) || (_window_size == XF_FILTER_5X5)) &&
           ("Filter width must be either 3 or 5"));
    assert(_border_type == XF_BORDER_CONSTANT && "Only XF_BORDER_CONSTANT is supported");
    assert(((img_height <= ROWS) && (img_width <= COLS)) && "ROWS and COLS should be greater than input image");
    assert((NPC == XF_NPPC1) && ("NPC must be XF_NPPC1"));
#endif
    xFCensus5x5<ROWS, COLS, DEPTH_SRC, DEPTH_DST, NPC, WORDWIDTH_SRC, WORDWIDTH_DST>(_src, _dst, img_height, img_width);
} // end of wrapper function

template <int NDISP, int PU, int ROWS, int COLS>
void xFSGBMcomputecost(hls::stream<ap_uint<24> >& _src_census24_l,
                       hls::stream<ap_uint<24> >& _src_census24_r,
                       hls::stream<ap_uint<8> > _cost[PU],
                       int height,
                       int width) {
// clang-format off
    #pragma HLS INLINE OFF
    #pragma HLS ARRAY_PARTITION variable=_cost complete dim=1 // TODO
    // clang-format on

    ap_uint<24> l_val;
    ap_uint<24> r_val;
    ap_uint<24> r_buff[NDISP];
// clang-format off
    #pragma HLS ARRAY_PARTITION variable=r_buff complete dim=1
// clang-format on

loop_height:
    for (int r = 0; r < height; r++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=1 max=ROWS
    // clang-format on

    loop_sweep:
        for (int i = 0; i < NDISP; i++) {
// clang-format off
            #pragma HLS UNROLL
            // clang-format on
            r_buff[i] = 0;
        }

    loop_width:
        for (int c = 0; c < width; c++) {
// clang-format off
            #pragma HLS LOOP_TRIPCOUNT min=1 max=COLS
            // clang-format on
            if (PU == NDISP) {
// clang-format off
                #pragma HLS PIPELINE II=1
                // clang-format on
            }

        loop_sweep_inside:
            for (int i = 0; i < NDISP / PU; i++) {
// clang-format off
                #pragma HLS PIPELINE II=1
                #pragma HLS loop_flatten
                // clang-format on

                if (i == 0) {
                    l_val = _src_census24_l.read();
                    r_val = _src_census24_r.read();

                // shift the buffer left
                loop_shift:
                    for (int i = NDISP - 1; i > 0; i--) r_buff[i] = r_buff[i - 1];
                    // insert the new value at the end
                    r_buff[0] = r_val;
                }

            loop_parallel_unit:
                for (int j = 0; j < PU; j++) {
// clang-format off
                    #pragma HLS UNROLL
                    // clang-format on
                    ap_uint<24> xor_val = l_val ^ r_buff[i * PU + j];
                    uint8_t sum = 0;

                loop_hamming_sum:
                    for (int k = 0; k < 24; k++) {
// clang-format off
                        #pragma HLS LOOP_TRIPCOUNT min=1 max=24
                        // clang-format on

                        uint8_t c = (uint8_t)(xor_val & 0x1);
                        sum += xor_val.range(k, k);
                    }
                    _cost[j].write((ap_uint8_t)sum);
                }
            }
        }
    }
}

static uint8_t min_of_4(uint8_t a, uint8_t b, uint8_t c, uint8_t d) {
// clang-format off
    #pragma HLS INLINE
    // clang-format on
    uint8_t res, res1, res2;
    res1 = a < b ? a : b;
    res2 = c < d ? c : d;
    res = res1 < res2 ? res1 : res2;
    return res;
}

static uint8_t fn_reg(uint8_t value) {
    //#pragma HLS inline off
    //#pragma HLS interface register port=return
    return value;
}

template <typename T>
static T fn_reg_scalar(T scalar) {
// clang-format off
    #pragma HLS inline //off
    // clang-format on
    //#pragma HLS interface register port=return
    return scalar;
}

template <int NDISP, int PU, int R, int ROWS, int COLS>
void xFSGBMoptimization(hls::stream<ap_uint<8> > _cost[PU],
                        hls::stream<ap_uint<16> > _agg_cost[PU],
                        int height,
                        int width,
                        uint8_t p1,
                        uint8_t p2) {
    // array for the Lr data
    //	int CYC_PART = PU+1;  // PU+1 or PU+2, TODO need to check and fix this

    // array to store the Lr computed values for the Lr computation of neighboring pixels, we don't need for 180 degree
    // so R-1. This array consumes BRAMs first dimension completely partitioned, while the second dimension of the Array
    // was partitioned in a cyclic manner, which will allow us to access
    //     PU consecutive array indexes to be accessed in parallel
    uint8_t Lr[R - 1][NDISP][COLS];
// clang-format off
    #pragma HLS RESOURCE variable=Lr core=RAM_T2P_BRAM
    #pragma HLS ARRAY_PARTITION variable=Lr complete dim=1
    #pragma HLS ARRAY_PARTITION variable=Lr complete dim=2
    // clang-format on
    //	#pragma HLS ARRAY_PARTITION variable=Lr cyclic factor=CYC_PART dim=2

    uint8_t Lr_r1[NDISP];
// clang-format off
    #pragma HLS ARRAY_PARTITION variable=Lr_r1 complete dim=1
    // clang-format on
    uint8_t Lr_r1_tmp[PU];
// clang-format off
    #pragma HLS ARRAY_PARTITION variable=Lr_r1_tmp complete dim=1
    // clang-format on

    // array to store r0 data for the computation of next pixel in the raster scan manner, so one pixel's Lr data is
    // sufficient
    uint8_t Lr_r0[NDISP];
// clang-format off
    #pragma HLS ARRAY_PARTITION variable=Lr_r0 complete dim=0
    // clang-format on

    // temporary array which reads in the data from Lr array. This array will be processed
    // this array works as a queue, after initialization, reads the data from BRAM to the last index while the first
    // index is off loaded.
    uint8_t tmp_store_Lr[R][PU + 2];
// clang-format off
    #pragma HLS ARRAY_PARTITION variable=tmp_store_Lr complete dim=0
    // clang-format on

    // array to hold the first four min values, first dimension R is for directions, second one to hold first four min
    // values. This array consumes BRAMs holds for all direction except 180 degree
    uint8_t Lr_min[R - 1][COLS];
// clang-format off
    #pragma HLS ARRAY_PARTITION variable=Lr_min complete dim=1
    // clang-format on
    // array to temporarily hold the data for r1 direction
    uint8_t r1_min;
    // array to hold the first four min values for the first direction, 180 degree
    uint8_t r0_min;

    // temporary array which reads in the data from Lr_min array. This array will be using in the processing block. The
    // min arrays will always be in a sorted
    uint8_t tmp_Lr_min[R];
// clang-format off
    #pragma HLS ARRAY_PARTITION variable=tmp_Lr_min complete dim=1
    // clang-format on
    // suffix post to temporarily store the min values before writing to the BRAM.
    // Post buffers are used for computation of minimum on the run, and the data are finally dumped into Lr_min array
    uint8_t tmp_Lr_min_post[R];
// clang-format off
    #pragma HLS ARRAY_PARTITION variable=tmp_Lr_min_post complete dim=1
    // clang-format on
    // dim 1 is for directions, and dim 2 for parallel units. This is an intermediary array which hold the computed Lr
    // for min computation
    uint8_t store_lr_for_min[R][PU];
// clang-format off
    #pragma HLS ARRAY_PARTITION variable=store_lr_for_min complete dim=0
    // clang-format on
    for (int i = 0; i < R - 1; i++) {
        for (int j = 0; j < NDISP; j++) {
            for (int k = 0; k < COLS; k++) {
                Lr[i][j][k] = 0;
            }
        }
    }
    for (int j = 0; j < NDISP; j++) {
        Lr_r0[j] = 0;
    }
    for (int i = 0; i < R - 1; i++) {
        for (int k = 0; k < COLS; k++) {
            Lr_min[i][k] = 0;
        }
    }
    tmp_Lr_min_post[0] = 0;

loop_row:
    for (int ro = 0; ro < height; ro++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=1 max=ROWS
    // clang-format on

    loop_col:
        for (int co = 0; co < width; co++) {
// clang-format off
            #pragma HLS LOOP_TRIPCOUNT min=1 max=COLS
            // clang-format on
            if (PU == NDISP) {
// clang-format off
                #pragma HLS PIPELINE II=2
                // clang-format on
            }
            // process loop
            uint8_t min_d0, min_cost0; // vs
        disp_loop:
            for (int d = 0; d < NDISP / PU; d++) {
// clang-format off
                #pragma HLS PIPELINE II=2
                #pragma HLS DEPENDENCE variable=Lr array intra false
                //#pragma HLS DEPENDENCE variable=Lr array inter false
                #pragma HLS DEPENDENCE variable=Lr_min array inter false
                #pragma HLS LOOP_FLATTEN
                // clang-format on

                if (d == 0) {
                    /////// Initialization of the process array from the BRAMs ///////
                    for (int r = 0; r < R; r++) // previous disparity for d=0 is initialized with zero
                    {
// clang-format off
                        #pragma HLS UNROLL
                        // clang-format on
                        tmp_store_Lr[r][0] = 0;
                    }

                    for (int pu = 0; pu < PU; pu++) {
// clang-format off
                        #pragma HLS UNROLL
                        // clang-format on
                        tmp_store_Lr[0][pu + 1] = Lr_r0[pu];
                        for (int r = 1; r < R; r++) {
// clang-format off
                            #pragma HLS UNROLL
                            // clang-format on
                            tmp_store_Lr[r][pu + 1] = Lr[r - 1][pu][co + r - 2];
                        }
                    }

                    // border disparity case, when PU is same as number of disparity
                    // if parallel units are same as the maximum disparity then no next disparity for the final
                    // disparity computation
                    if (PU < NDISP)
                        tmp_store_Lr[0][PU + 1] = Lr_r0[PU];
                    else
                        tmp_store_Lr[0][PU + 1] = 0;
                    for (int r = 1; r < R; r++) {
// clang-format off
                        #pragma HLS UNROLL
                        // clang-format on
                        if (PU < NDISP)
                            tmp_store_Lr[r][PU + 1] = Lr[r - 1][PU][co + r - 2];
                        else
                            tmp_store_Lr[r][PU + 1] = 0;
                    }

                    // Copy Lr min values from the BRAM to temporary array which is used for processing
                    // vs tmp_Lr_min[0] = r0_min;
                    tmp_Lr_min[0] = tmp_Lr_min_post[0];
                    for (int r = 1; r < R; r++) {
// clang-format off
                        #pragma HLS UNROLL
                        // clang-format on
                        tmp_Lr_min[r] = Lr_min[r - 1][co + r - 2];
                    }

                    // initialize the post buffer with max values, helps in comparisons while sorting
                    for (int r = 0; r < R; r++) {
// clang-format off
                        #pragma HLS UNROLL
                        // clang-format on
                        tmp_Lr_min_post[r] = MAX_UCHAR;
                    }
                } else {
                    for (int r = 0; r < R; r++) {
// clang-format off
                        #pragma HLS UNROLL
                        // clang-format on
                        tmp_store_Lr[r][0] = tmp_store_Lr[r][PU];
                        tmp_store_Lr[r][1] = tmp_store_Lr[r][PU + 1];
                    }
                    for (int pu = 1; pu < PU; pu++) {
// clang-format off
                        #pragma HLS UNROLL
                        // clang-format on
                        uint16_t disp_idx = (d * PU + pu);
                        tmp_store_Lr[0][pu + 1] = Lr_r0[disp_idx];
                        for (int r = 1; r < R; r++) {
// clang-format off
                            #pragma HLS UNROLL
                            // clang-format on
                            tmp_store_Lr[r][pu + 1] = Lr[r - 1][disp_idx][co + r - 2];
                        }
                    }

                    uint16_t disp_idx = d * PU + PU;
                    if (disp_idx < NDISP)
                        tmp_store_Lr[0][PU + 1] = Lr_r0[disp_idx];
                    else
                        tmp_store_Lr[0][PU + 1] = 0;
                    for (int r = 1; r < R; r++) {
// clang-format off
                        #pragma HLS UNROLL
                        // clang-format on
                        if (disp_idx < NDISP)
                            tmp_store_Lr[r][PU + 1] = Lr[r - 1][disp_idx][co + r - 2];
                        else
                            tmp_store_Lr[r][PU + 1] = 0;
                    }
                }

            loop_pu:
                for (int pu = 0; pu < PU; pu++) {
// clang-format off
                    #pragma HLS UNROLL
                    // clang-format on
                    uint8_t cpd = (uint8_t)_cost[pu].read();
                    uint16_t agg_val = 0;

                loop_directions:
                    for (int r = 0; r < R; r++) {
// clang-format off
                        #pragma HLS UNROLL
                        // clang-format on
                        // mink minimum of all disparity, mini minimum of all disparity except d-1, d, d+1
                        uint8_t lr_dp, lr_d, lr_dn, lr_mink = MAX_UCHAR;
                        lr_dp = tmp_store_Lr[r][pu];
                        lr_d = tmp_store_Lr[r][pu + 1];
                        lr_dn = tmp_store_Lr[r][pu + 2];
                        lr_mink = tmp_Lr_min[r];

                        // vs
                        uint8_t p1reg = fn_reg_scalar<uint8_t>(p1);

                        // border disparity cases for storing the lr values
                        int disp_idx = d * PU + pu;

                        // border case with respect to disparity
                        if (disp_idx == 0) lr_dp = MAX_UCHAR - p1reg;
                        if (disp_idx >= (NDISP - 1)) lr_dn = MAX_UCHAR - p1reg;

                        uint8_t tmini, tminv;
                        uint8_t tmp_arr[4];
// clang-format off
                        #pragma HLS ARRAY_PARTITION variable=tmp_arr complete dim=1
                        // clang-format on
                        tmp_arr[0] = lr_d;
                        tmp_arr[1] = lr_dp + p1;
                        tmp_arr[2] = lr_dn + p1;
                        uint8_t p2reg;
                        if ((r == 0) && (co == 0)) {
                            p2reg = 0;
                        } else {
                            p2reg = p2;
                        }
                        tmp_arr[3] = lr_mink + p2reg;
                        xFMinSAD<4>::find(tmp_arr, tmini, tminv);

                        // process block
                        uint8_t lr_tmp;
// clang-format off
                        #pragma HLS RESOURCE variable=lr_tmp core=AddSub_DSP
                        // clang-format on
                        lr_tmp = cpd - (uint8_t)lr_mink;

                        uint8_t lr;
// clang-format off
                        #pragma HLS RESOURCE variable=lr_tmp core=AddSub_DSP
                        // clang-format on
                        lr = lr_tmp + tminv;

                        // row or col border case
                        if (((r == 1) && (co == 0)) || (((r == 1) || (r == 2) || (r == 3)) && (ro == 0)) ||
                            ((r == 3) && (co == width - 1)))
                            lr = cpd;

                        // assignment
                        if (r == 0)
                            Lr_r0[disp_idx] = lr;
                        else if (r == 1)
                            Lr_r1_tmp[pu] = lr;
                        else
                            Lr[r - 1][disp_idx][co] = lr;

                        store_lr_for_min[r][pu] = lr;
                        agg_val += lr;
                    }
                    _agg_cost[pu].write((ap_uint16_t)agg_val);
                }

                //				uint8_t store_lr_for_min_reg[R][PU];
                //#pragma HLS array_partition variable=store_lr_for_min complete dim=0
                //				for (int p=0; p<PU; p++) {
                //					for (int r=0; r<R; r++) {
                //						store_lr_for_min_reg[r][p] =
                // fn_reg(store_lr_for_min[r][p]);
                //					}
                //				}

                // compute min value for all sets of disparities
                xFMinSAD<PU>::find(store_lr_for_min[0], min_d0, min_cost0);
                if (min_cost0 < tmp_Lr_min_post[0]) tmp_Lr_min_post[0] = min_cost0;

                for (int r = 1; r < R; r++) {
// clang-format off
                    #pragma HLS UNROLL
                    // clang-format on
                    uint8_t min_d, min_cost;
                    xFMinSAD<PU>::find(store_lr_for_min[r], min_d, min_cost);
                    if (min_cost < tmp_Lr_min_post[r]) tmp_Lr_min_post[r] = min_cost;
                }

                // updating the previous
                for (int pu = 0; pu < PU; pu++) {
// clang-format off
                    #pragma HLS UNROLL
                    // clang-format on
                    int disp_idx = d * PU + pu;
                    if (co >= 1) {
                        Lr[0][disp_idx][co - 1] = Lr_r1[disp_idx];
                    }
                    Lr_r1[disp_idx] = Lr_r1_tmp[pu];
                }

                if (d >= (NDISP / PU -
                          1)) // when its the last set of disparities update the min arrays from the min post arrays
                {
                    // for the last pixel in the col update the min values
                    if (co > 0) {
                        Lr_min[0][co - 1] = r1_min;
                    }

                    // vs r0_min = tmp_Lr_min_post[0];

                    r1_min = tmp_Lr_min_post[1];
                    for (int r = 2; r < R; r++) {
// clang-format off
                        #pragma HLS UNROLL
                        // clang-format on
                        Lr_min[r - 1][co] = tmp_Lr_min_post[r];
                    }
                }
                // update the tmp_store array for the next set of disparity computation, so not necessary for the last
                // set
            }
        }
    }
}

template <int NDISP, int PU, int ROWS, int COLS>
void xfSGBMcomputedisparity(hls::stream<ap_uint<16> > _agg_cost[PU],
                            hls::stream<ap_uint<8> >& _dst,
                            int height,
                            int width) {
// clang-format off
    #pragma HLS INLINE OFF
    // clang-format on

    const int TOTAL_ITER = NDISP / PU;

    for (int r = 0; r < height; r++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=1 max=ROWS
        // clang-format on
        for (int c = 0; c < width; c++) {
// clang-format off
            #pragma HLS LOOP_TRIPCOUNT min=1 max=COLS
            // clang-format on
            if (PU == NDISP) {
// clang-format off
                #pragma HLS PIPELINE II=1
                // clang-format on
            }

            ap_uint<8> lmin_d;
            ap_uint<16> lmin_cost;
            ap_uint<16> min_cost = 32768;
            ap_uint<8> min_disp;

            for (int i = 0; i < TOTAL_ITER; i++) {
// clang-format off
                #pragma HLS PIPELINE II=1
                #pragma HLS LOOP_FLATTEN
                // clang-format on

                ap_uint<16> tmp[PU];
                for (int j = 0; j < PU; j++) {
                    tmp[j] = _agg_cost[j].read();
                }
                xFMinSAD<PU>::find(tmp, lmin_d, lmin_cost);
                if (lmin_cost < min_cost) {
                    min_disp = i * PU + lmin_d;
                    min_cost = lmin_cost;
                }
            }
            _dst.write(min_disp);
        }
    }
}

template <int BORDER_TYPE, int WINDOW_SIZE, int NDISP, int PU, int R, int SRC_T, int DST_T, int ROWS, int COLS, int NPC>
void SemiGlobalBM(xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _src_mat_l,
                  xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _src_mat_r,
                  xf::cv::Mat<DST_T, ROWS, COLS, NPC>& _dst_mat,
                  uint8_t p1,
                  uint8_t p2) {
#ifndef _SYNTHESIS_
    assert((SRC_T == XF_8UC1) && " WORDWIDTH_SRC must be XF_8UC1 ");
    assert((DST_T == XF_8UC1) && " WORDWIDTH_DST must be XF_8UC1 ");
    assert((NPC == XF_NPPC1) && " NPC must be XF_NPPC1 ");
    assert((WINDOW_SIZE == 5) && " WSIZE must be set to '5' ");
    assert(((NDISP > 1) && (NDISP <= 256)) && " NDISP must be greater than '1' and less than or equal to '256' ");
    assert((NDISP >= PU) && " NDISP must not be lesser than PU (parallel units)");
    assert((((NDISP / PU) * PU) == NDISP) && " NDISP/PU must be a non-fractional number ");
    assert(((R == 2) || (R == 3) || (R == 4)) && "Number of directions R must be '2', '3' or '4' ");
    assert((p1 < p2) && "p1 must be always less than p2");
    assert((p2 <= 100) && "Maximum value of p2 must be 100 ");
#endif

    hls::stream<XF_TNAME(SRC_T, NPC)> _src_l;
    hls::stream<XF_TNAME(SRC_T, NPC)> _src_r;

    hls::stream<ap_uint<32> > _src_census_l;
    hls::stream<ap_uint<32> > _src_census_r;

    hls::stream<ap_uint<24> > _src_census24_l;
    hls::stream<ap_uint<24> > _src_census24_r;

    hls::stream<ap_uint<8> > _cost[PU];

    hls::stream<ap_uint<16> > _agg_cost[PU];

    hls::stream<XF_TNAME(DST_T, NPC)> _dst;

// clang-format off
    #pragma HLS INLINE OFF
    #pragma HLS DATAFLOW
    // clang-format on

    int height = _src_mat_l.rows;
    int width = _src_mat_l.cols;
    int dheight = _dst_mat.rows;
    int dwidth = _dst_mat.cols;

    // Reading data from Mat to stream
    for (int i = 0; i < height; i++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=1 max=ROWS
        // clang-format on
        for (int j = 0; j < width; j++) {
// clang-format off
            #pragma HLS LOOP_TRIPCOUNT min=1 max=COLS
            #pragma HLS LOOP_FLATTEN off
            #pragma HLS PIPELINE
            // clang-format on
            _src_l.write(_src_mat_l.read(i * width + j));
            _src_r.write(_src_mat_r.read(i * width + j));
        }
    }

    xFCensusTransformKernel<ROWS, COLS, XF_DEPTH(SRC_T, NPC), XF_DEPTH(XF_32UC1, NPC), NPC, XF_WORDWIDTH(SRC_T, NPC),
                            XF_WORDWIDTH(XF_32UC1, NPC)>(_src_l, _src_census_l, WINDOW_SIZE, BORDER_TYPE, height,
                                                         width);
    xFCensusTransformKernel<ROWS, COLS, XF_DEPTH(SRC_T, NPC), XF_DEPTH(XF_32UC1, NPC), NPC, XF_WORDWIDTH(SRC_T, NPC),
                            XF_WORDWIDTH(XF_32UC1, NPC)>(_src_r, _src_census_r, WINDOW_SIZE, BORDER_TYPE, height,
                                                         width);

    for (int i = 0; i < height; i++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=1 max=ROWS
        // clang-format on
        for (int j = 0; j < width; j++) {
// clang-format off
            #pragma HLS LOOP_TRIPCOUNT min=1 max=COLS
            #pragma HLS LOOP_FLATTEN off
            #pragma HLS PIPELINE
            // clang-format on
            _src_census24_l.write((ap_uint<24>)_src_census_l.read());
            _src_census24_r.write((ap_uint<24>)_src_census_r.read());
        }
    }

    xFSGBMcomputecost<NDISP, PU, ROWS, COLS>(_src_census24_l, _src_census24_r, _cost, height, width);

    xFSGBMoptimization<NDISP, PU, R, ROWS, COLS>(_cost, _agg_cost, height, width, p1, p2);

    xfSGBMcomputedisparity<NDISP, PU, ROWS, COLS>(_agg_cost, _dst, height, width);

    // write back from stream to Mat
    for (int i = 0; i < dheight; i++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=1 max=ROWS
        // clang-format on
        for (int j = 0; j < dwidth; j++) {
// clang-format off
            #pragma HLS LOOP_TRIPCOUNT min=1 max=COLS
            #pragma HLS LOOP_FLATTEN off
            #pragma HLS PIPELINE
            // clang-format on
            _dst_mat.write(i * dwidth + j, _dst.read());
        }
    }
}
} // namespace cv
} // namespace xf
#endif
