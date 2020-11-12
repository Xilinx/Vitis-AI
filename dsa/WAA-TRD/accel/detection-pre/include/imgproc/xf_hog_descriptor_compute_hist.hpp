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

#ifndef _XF_HOG_DESCRIPTOR_COMPUTE_HIST_HPP_
#define _XF_HOG_DESCRIPTOR_COMPUTE_HIST_HPP_

#ifndef __cplusplus
#error C++ is needed to include this header
#endif

#include "imgproc/xf_hog_descriptor_utility.hpp"

/*************************************************************************
 *					Macros for the histogram function
 *************************************************************************/
#define XF_HOG_1_BY_20 3276 // considering the Bin stride as 20
//			in Q0.16 format

#define XF_HOG_PI 23040 // 180 in Q8.7 format

// functional macro to scale the angle to 180
#define scale_to_PI(a) \
    if (a > XF_HOG_PI) a = a - XF_HOG_PI

/*************************************************************************/

/***************************************************************************
 *    		                 xFDHOGbilinearNO
 ***************************************************************************
 * functionality: utility function for the compute histogram kernel.
 *
 * 		-> performs bilinear interpolation on the magnitude depending upon
 * 		   the phase value and the bin position where the pixel falls under
 *
 *		-> p: input phase value (Q9.7)
 *
 *		-> m: input magnitude value (Q9.7)
 *
 *		-> bin_center: bin center array which acts as a LUT to fetch the
 *				corresponding bin centers
 *
 *		-> bin: temporary array to hold the histogram
 *
 ***************************************************************************/
template <int DEPTH, typename hist_type, int NOB, int BIN_STRIDE_FIX>
void xFDHOGbilinearNO(XF_PTNAME(DEPTH) p, XF_PTNAME(DEPTH) m, uint16_t* bin_center, hist_type* bin) {
// clang-format off
    #pragma HLS INLINE OFF
    // clang-format on

    // finding the bin index with respect to the phase
    uint32_t tmp_bin_idx = p * XF_HOG_1_BY_20; // Q9.7xQ0.16 -> Q9.23
    uchar_t bin_idx = tmp_bin_idx >> 23;
    uchar_t index_1, index_2;

    // magnitude factor
    //	uint32_t mag_factor = (uint16_t)XF_1_BY_20 * m;  // Q9.16 format
    uint32_t mag_factor = m * XF_HOG_1_BY_20; // Q9.7xQ0.16 -> Q9.23
    uint16_t frac = 0;

    // interpolate with the previous cell, when 'p' is less than bin_center
    if (p < bin_center[bin_idx]) {
        /* when the bin is the first bin and the 'p' is
                less than the corresponding bin_center value then
                interpolate with the last cell (in a circular fashion)  */
        if (bin_idx == 0) {
            frac = (p - (bin_center[bin_idx] - BIN_STRIDE_FIX));
            index_1 = bin_idx;
            index_2 = (NOB - 1);
        } else {
            frac = (p - bin_center[bin_idx - 1]);
            index_1 = bin_idx;
            index_2 = (bin_idx - 1);
        }
    }

    // interpolate with the next cell, when 'p' is greater than bin_center
    else {
        /* when the bin is the last bin and the 'p' is
                greater than the corresponding bin_center value then
                interpolate with the first cell (in a circular fashion)  */
        if (bin_idx == (NOB - 1)) {
            frac = ((bin_center[bin_idx] + BIN_STRIDE_FIX) - p);
            index_1 = bin_idx;
            index_2 = 0;
        } else {
            frac = (bin_center[bin_idx + 1] - p);
            index_1 = bin_idx;
            index_2 = (bin_idx + 1);
        }
    }

    // interpolate to the bin according to the fraction values  // frac -> Q9.7 format
    hist_type part_1 =
        (hist_type)(((ap_uint<40>)frac * (ap_uint<40>)mag_factor) >> 22); // Q9.7xQ9.23 -> Q18.30>>22 -> Q18.8 format
    uint32_t m_shifted = (uint32_t)m << 1;                                // converting 'm' to Q9.8 format
    hist_type part_2 = (hist_type)(m_shifted - part_1);                   // Q9.8 format

    // accumulate the data to bin array  [ bin array in Q9.8 format ]
    bin[index_1] += part_1;
    bin[index_2] += part_2;
}

/***************************************************************************************
 *    		                		 xFDHOGcomputeHistNO
 ***************************************************************************************
 * functionality: utility function for the compute histogram kernel.
 *
 * 		-> performs binning of the interpolated values into the histogram
 * 		   array
 *
 *		-> bin : array containing the bin values that need to be accumulated
 *			into the HA array
 *
 *		-> HA : the histogram array that need to be accumulated
 *
 *		-> ssv : sum of squared value (square all the 9 bins and accumulate then).
 *				This value acts as a temporary result for normalization factor.
 *
 **************************************************************************************/
template <int ROWS,
          int COLS,
          int DEPTH,
          int NPC,
          int WORDWIDTH,
          int CELL_HEIGHT,
          int CELL_WIDTH,
          int NOHC,
          int TC,
          int WIN_STRIDE,
          int BIN_STRIDE,
          typename hist_type,
          int NOB,
          typename ssv_type>
void xFDHOGcomputeHistNO(hls::stream<XF_SNAME(WORDWIDTH)>& _phase_strm,
                         hls::stream<XF_SNAME(WORDWIDTH)>& _mag_strm,
                         hist_type HA[][NOB],
                         ssv_type* ssv,
                         uint16_t* bin_center,
                         uint16_t nohc) {
    // read the input data from the streams to the local variables
    XF_SNAME(WORDWIDTH) phase_data, mag_data;
    XF_PTNAME(DEPTH) p, m;

    uint16_t proc_loop = XF_WORDDEPTH(WORDWIDTH), frac;
    uchar_t step = XF_PIXELDEPTH(DEPTH), bin_idx = 0;
    ap_uint<16> i, j, r;
    ap_uint<8> k;

    // NPC copied of the histogram array
    hist_type bin[NOB];
// clang-format off
    #pragma HLS ARRAY_PARTITION variable=bin complete dim=1
// clang-format on

// initializing the histogram array with zero
loop_i_init_zero:
    for (i = 0; i < nohc; i++) {
// clang-format off
        #pragma HLS PIPELINE
    // clang-format on

    loop_j_init_zero:
        for (j = 0; j < NOB; j++) {
// clang-format off
            #pragma HLS unroll
            // clang-format on
            HA[i][j] = 0;
        }
    }

cell_height_loop:
    for (i = 0; i < CELL_HEIGHT; i++) {
    no_of_horz_cell_loop:
        for (r = 0; r < nohc; r++) {
// clang-format off
            #pragma HLS LOOP_TRIPCOUNT min=NOHC max=NOHC
            #pragma HLS PIPELINE
        // clang-format on

        init_bin_loop_k:
            for (k = 0; k < NOB; k++) {
// clang-format off
                #pragma HLS unroll
                // clang-format on
                bin[k] = 0;
            }

        img_col_loop:
            for (j = 0; j < CELL_WIDTH; j++) {
                // reading data from the stream
                phase_data = _phase_strm.read();
                mag_data = _mag_strm.read();

                p = phase_data.range((step - 1), 0);
                m = mag_data.range((step - 1), 0);

                // scale the angle to 180 degree (if it is beyond 180)
                scale_to_PI(p);

                // accumulating to the temporary histogram array 'bin'
                xFDHOGbilinearNO<DEPTH, hist_type, NOB, (BIN_STRIDE << 7)>(p, m, bin_center, bin);
            }

        hist_BRAM_updation_loop:
            for (k = 0; k < NOB; k++) {
                HA[r][k] += bin[k]; // HA array in Q15.8 format
            }

            // square computation on temporary ssv val
            ssv_type tmp = 0;

        tmp_ssv_computation:
            for (k = 0; k < NOB; k++) // adder tree inferred
            {
// clang-format off
                #pragma HLS unroll
                // clang-format on
                tmp += (HA[r][k] * HA[r][k]);
            }
            ssv[r] = tmp; // Q29.16 format <45> bits
        }
    }
}

/***************************************************************************
 *    		                 xFDHOGbilinearRO
 ***************************************************************************
 * functionality: utility function for the compute histogram kernel.
 *
 * 		-> performs bilinear interpolation on the magnitude depending upon
 * 		   the phase value and the bin position where the pixel falls under
 *
 *		-> index : these variables indicates the bins on which the
 *			particular pixel falls under
 *
 *		-> part : these variables contains the splitted up magnitude data
 *
 ***************************************************************************/
template <int BIN_STRIDE, int DEPTH, typename hist_type, int NOB>
void xFDHOGbilinearRO(XF_PTNAME(DEPTH) p,
                      XF_PTNAME(DEPTH) m,
                      uint16_t* bin_center,
                      uchar_t* index_1,
                      uchar_t* index_2,
                      hist_type* part_1,
                      hist_type* part_2) {
// clang-format off
    #pragma HLS INLINE OFF
    // clang-format on

    // finding the bin index with respect to the phase
    uint32_t tmp_bin_idx = p * (uint16_t)XF_HOG_1_BY_20;
    uchar_t bin_idx = tmp_bin_idx >> 16;

    // magnitude factor
    uint32_t mag_factor = (uint16_t)XF_HOG_1_BY_20 * m;
    uint16_t frac = 0;

    // interpolate with the previous cell, when 'p' is less than bin_center
    if (p < bin_center[bin_idx]) {
        /* when the bin is the first bin and the 'p' is
                less than the corresponding bin_center value then
                interpolate with the last cell (in a circular fashion)  */
        if (bin_idx == 0) {
            frac = (p - (bin_center[bin_idx] - BIN_STRIDE));
            *index_1 = bin_idx;
            *index_2 = (NOB - 1);
        } else {
            frac = (p - bin_center[bin_idx - 1]);
            *index_1 = bin_idx;
            *index_2 = (bin_idx - 1);
        }
    }

    // interpolate with the next cell, when 'p' is greater than bin_center
    else {
        /* when the bin is the last bin and the 'p' is
                greater than the corresponding bin_center value then
                interpolate with the first cell (in a circular fashion)  */
        if (bin_idx == (NOB - 1)) {
            frac = ((bin_center[bin_idx] + BIN_STRIDE) - p);
            *index_1 = bin_idx;
            *index_2 = 0;
        } else {
            frac = (bin_center[bin_idx + 1] - p);
            *index_1 = bin_idx;
            *index_2 = (bin_idx + 1);
        }
    }

    // interpolate to the bin according to the fraction values
    *part_1 = (hist_type)((frac * mag_factor) >> 8); // Q9.8 format
    uint32_t m_shifted = (uint32_t)m << 8;           // converting 'm' to Q9.8 format
    *part_2 = (hist_type)(m_shifted - *part_1);      // Q9.8 format
}

/***************************************************************************
 *    		                 xFDHOGBinRO
 ***************************************************************************
 * functionality: utility function for the compute histogram kernel to
 * 			perform the histogram accumulation.
 *
 *		-> bin: temporary bin that must be accumulated to HA array
 *
 *		-> HA : histogram array
 *
 *		-> j : indicates the cell index
 *
 ***************************************************************************/
template <int NPC, int NOB, typename hist_type, int NOHC>
void xFDHOGBinRO(hist_type bin[][NOB], hist_type HA[][NOB], uint16_t j) {
// clang-format off
    #pragma HLS INLINE
// clang-format on

// add all the 8 copies of the temporary array into single copy
accumulate_to_bin_0_k:
    for (uchar_t k = 0; k < NOB; k++) {
// clang-format off
        #pragma HLS UNROLL
    // clang-format on

    accumulate_to_bin_0_l:
        for (uchar_t l = 1; l < (1 << XF_BITSHIFT(NPC)); l++) {
            bin[0][k] += bin[l][k];
        }
    }

// accumulating the temporary histogram to HA array
accumulate_HA:
    for (uchar_t k = 0; k < NOB; k++) {
// clang-format off
        #pragma HLS PIPELINE
        // clang-format on

        HA[j][k] += bin[0][k];
    }
}

/***************************************************************************
 *    		               xFDHOGcomputeHistRO
 ***************************************************************************
 * functionality: kernel function for computing the histogram
 *
 * 		-> compute the histogram with the input phase and the magnitude
 * 				values. The output of this function is the computed
 * 				histogram array.
 *
 *		-> _phase_strm: input phase mat object (stream)
 *		   _mag_strm : input magnitude mat object (stream)
 *
 *		-> HA : histogram array (output)
 *
 *		-> ssv : sum of squared value (output)
 *
 *    note: This code is disabled for now, and when further develpoment for
 *    higher parallelism is introduced, it will be develped from this code
 ***************************************************************************/
/*template <int ROWS, int COLS, int DEPTH, int NPC, int WORDWIDTH, int CELL_HEIGHT,
int CELL_WIDTH, int NOHC, int TC, int WIN_STRIDE, int BIN_STRIDE, typename hist_type,
int NOB, typename ssv_type>
void xFDHOGcomputeHistRO(
                auviz::Mat<ROWS,COLS,DEPTH,NPC,WORDWIDTH>& _phase_strm,
                auviz::Mat<ROWS,COLS,DEPTH,NPC,WORDWIDTH>& _mag_strm,
                hist_type HA[][NOB], ssv_type* ssv, uint16_t* bin_center)
{
        // read the input data from the streams to the local variables
        XF_SNAME(WORDWIDTH) phase_data, mag_data;
        XF_PTNAME(DEPTH) p, m;

        // NPC copied of the histogram array
        hist_type bin[XF_NPIXPERCYCLE(NPC)][NOB], part_1, part_2;
// clang-format off
#pragma HLS ARRAY_PARTITION variable=bin complete dim=0
// clang-format on

        uint16_t proc_loop = XF_WORDDEPTH(WORDWIDTH);
        uchar_t step = XF_PIXELDEPTH(DEPTH), npc_idx, index_1, index_2;

        // initializing the histogram array with zero
        loop_i_init_zero:
        for(uint16_t i = 0; i < NOHC; i++)
        {
// clang-format off
#pragma HLS pipeline
// clang-format on

                loop_j_init_zero:
                for(uint16_t j = 0; j < NOB; j++)
                {
// clang-format off
#pragma HLS unroll
// clang-format on
                        HA[i][j] = 0;
                }
        }

        cell_height_loop:
        for(uint16_t i = 0; i < CELL_HEIGHT; i++)
        {
                img_col_loop:
                for(uint16_t j = 0; j < _phase_strm.cols; j++)
                {
// clang-format off
#pragma HLS LOOP_TRIPCOUNT min=TC max=TC
#pragma HLS pipeline
// clang-format on

                        // reading data from the stream
                        phase_data = _phase_strm.read();
                        mag_data = _mag_strm.read();
                        npc_idx = 0;

                        init_bin_loop_k:
                        for(uchar_t k = 0; k < NOB; k++)
                        {
// clang-format off
#pragma HLS unroll
// clang-format on
                                init_bin_loop_l:
                                for(uchar_t l = 0; l < (1<<XF_BITSHIFT(NPC)); l++)
                                {
                                        bin[l][k] = 0;
                                }
                        }

                        proc_loop:
                        for(uint16_t k = 0; k < proc_loop; k+=step)
                        {
// clang-format off
#pragma HLS unroll
// clang-format on
                                p = phase_data.range(k + (step-1), k);
                                m = mag_data.range(k + (step-1), k);

                                // scale the angle to 180 degree if it is beyond 180
                                scale_to_PI(p);

                                // perform bilinear interpolation
                                xFDHOGbilinearRO<BIN_STRIDE,DEPTH,hist_type,NOB>(p,m,bin_center,
                                                &index_1,&index_2,&part_1,&part_2);

                                bin[npc_idx][index_1] = part_1;
                                bin[npc_idx][index_2] = part_2;
                                npc_idx++;
                        }
                        xFDHOGBinRO<NPC,NOB,hist_type,NOHC>(bin,HA,j); // HA array will be of format Q15.8

                        ssv_type tmp_ssv[NOB];
// clang-format off
#pragma HLS ARRAY_PARTITION variable=tmp_ssv complete dim=1
// clang-format on

                        // square computation on temporary ssv val
                        for(uchar_t k = 0; k < NOB; k++)
                        {
// clang-format off
#pragma HLS unroll
// clang-format on
                                tmp_ssv[k] = (HA[j][k] * HA[j][k]);
                        }

                        // adder tree for accumulating the array
                        ssv_type var1, var2, var3, var4;
                        var1 = tmp_ssv[0] + tmp_ssv[1];
                        var2 = tmp_ssv[2] + tmp_ssv[3];
                        var3 = tmp_ssv[4] + tmp_ssv[5];
                        var4 = tmp_ssv[6] + tmp_ssv[7];

                        var1 = var1 + var2;
                        var3 = var3 + var4;

                        var1 = var1 + var3;
                        ssv[j] = var1 + tmp_ssv[8];   // Q29.16 format <45> bits
                }
        }
}*/

/*******************************************************************************************
 *    		                 			xFDHOGcomputeHist
 *******************************************************************************************
 *
 * functionality: this function acts as a wrapper function for the compute histogram kernel
 * 				function. Computed the histogram and the sum of squared value (SSV)
 *
 *******************************************************************************************/
template <int ROWS,
          int COLS,
          int DEPTH,
          int NPC,
          int WORDWIDTH,
          int CELL_HEIGHT,
          int CELL_WIDTH,
          int NOHC,
          int TC,
          int WIN_STRIDE,
          int BIN_STRIDE,
          int NOB,
          typename hist_type,
          typename ssv_type>
void xFDHOGcomputeHist(hls::stream<XF_SNAME(WORDWIDTH)>& _phase_strm,
                       hls::stream<XF_SNAME(WORDWIDTH)>& _mag_strm,
                       hist_type HA[][NOB],
                       ssv_type* ssv,
                       uint16_t* bin_center,
                       uint16_t nohc) {
    // NO mode
    if (NPC == XF_NPPC1) {
        xFDHOGcomputeHistNO<ROWS, COLS, DEPTH, NPC, WORDWIDTH, CELL_HEIGHT, CELL_WIDTH, NOHC,
                            (COLS >> XF_BITSHIFT(NPC)), WIN_STRIDE, BIN_STRIDE>(_phase_strm, _mag_strm, HA, ssv,
                                                                                bin_center, nohc);
    }
}

#endif // _XF_HOG_DESCRIPTOR_COMPUTE_HIST_HPP_
