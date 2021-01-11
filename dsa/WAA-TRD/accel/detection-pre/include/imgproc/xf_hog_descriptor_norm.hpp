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

#ifndef _XF_HOG_DESCRIPTOR_NORM_HPP_
#define _XF_HOG_DESCRIPTOR_NORM_HPP_

#ifndef __cplusplus
#error C++ is needed to include this header
#endif

#define XF_CLIP_VAL 13108 //  Q0.2 in Q0.16 format
#define XF_MAX_VAL_16_BIT 65535
#define XF_3_P_6 921 //  Q2.8 format

/*******************************************************************************
 * 						 xFDHOGnormalizeKernel
 *******************************************************************************
 *  This function performs normalization and computes the partial
 *  renormalization factor.
 *
 *  HA arrays: contains binned data from the histogram computation function (I)
 *
 *  ssv arrays: contains sum of squared data from the histogram computation
 *  		function (I)
 *
 *  norm_block: holds the normalized data (O)
 *
 *  partial_rnf_sum: contains the partial ssv value for Re-normalization (O)
 *
 *  bj : index for horizontal blocks
 *
 *******************************************************************************/
template <typename ssv_type, typename tmp_nf_sq24_type>
void xFDHOGnormalizeKernel1(ssv_type ssv_1, ssv_type ssv_2, uint16_t bj, tmp_nf_sq24_type& tmp_nf_sq24) {
// clang-format off
    #pragma HLS INLINE OFF
    // clang-format on

    // temporary arrays to hold the ssv values of 4 cells
    ap_uint<51> tmp_nf_1, tmp_nf_2;

    tmp_nf_1 = ssv_1[bj] + ssv_2[bj];
    tmp_nf_2 = ssv_1[bj + 1] + ssv_2[bj + 1];
    ap_ufixed<50, 50, AP_TRN, AP_SAT> tmp_nf_sum;

    // contains the added up ssv values Q31.16 format
    tmp_nf_sum = (tmp_nf_1 + tmp_nf_2);

    // after square root the format is Q16.8
    tmp_nf_sq24 = xFSqrtHOG<24>(tmp_nf_sum) + XF_3_P_6;
}

template <typename tmp_nf_sq24_type, typename norm_fact_type>
void xFDHOGnormalizeKernelInv(tmp_nf_sq24_type tmp_nf_sq24, norm_fact_type& nf, char& n) {
    int m = 16;
    // after inverse the format will be Q(32-n).n
    nf = xFInverse24(tmp_nf_sq24, m, &n);
}

template <int NOHC,
          int NOHCPB,
          int NOVCPB,
          typename hist_type,
          int NOB,
          typename norm_block_type,
          typename fx_rnf_sq_type,
          typename norm_fact_type>
void xFDHOGnormalizeKernel2(hist_type HA_1[][NOB],
                            hist_type HA_2[][NOB],
                            norm_fact_type nf,
                            norm_block_type* norm_block,
                            fx_rnf_sq_type& fx_rnf_sq,
                            uint16_t bj,
                            char n) {
// clang-format off
    #pragma HLS INLINE OFF
    // clang-format on

    // HA_1 nad HA_2 in the Q15.8 format
    uint32_t tmp_clip_1, tmp_clip_2;

    // offsets to index the norm_array
    uchar_t offset_1[2], offset_2[2];
// clang-format off
    #pragma HLS ARRAY_PARTITION variable=offset_1 complete dim=1
    #pragma HLS ARRAY_PARTITION variable=offset_2 complete dim=1
    // clang-format on

    offset_1[0] = 0;
    offset_1[1] = NOB << 1;
    offset_2[0] = NOB;
    offset_2[1] = (NOB << 1) + NOB;

    /* keeping the clip value and norm factor in
                    the same format in the same format  */
    uint16_t fx_clip_val = XF_CLIP_VAL; // 0.2 taken in Q0.16 format
    ap_uint<16> i, j;

    ap_uint<33> rnf_sum = 0;
norm_loop:
    for (j = 0; j < NOHCPB; j++) {
    num_of_bins_loop:
        for (i = 0; i < NOB; i++) {
// clang-format off
            #pragma HLS LOOP_FLATTEN
            #pragma HLS PIPELINE
            // clang-format on

            // normalization I
            // // (Q15.8 x Q(32-n).n) >> n-8 -> Q16.16 format
            tmp_clip_1 = (((HA_1[bj + j][i]) * nf) >> (n - 8));
            tmp_clip_2 = (((HA_2[bj + j][i]) * nf) >> (n - 8));

            // clipping
            if (tmp_clip_1 > fx_clip_val) tmp_clip_1 = fx_clip_val;

            if (tmp_clip_2 > fx_clip_val) tmp_clip_2 = fx_clip_val;

            // norm_block format -> Q0.16
            norm_block[offset_1[j] + i] = tmp_clip_1;
            norm_block[offset_2[j] + i] = tmp_clip_2;

            // rnf_sum -> Q0.32
            rnf_sum += (norm_block[offset_1[j] + i] * norm_block[offset_1[j] + i]) +
                       (norm_block[offset_2[j] + i] * norm_block[offset_2[j] + i]);
        }
    }

    ap_ufixed<33, 33, AP_TRN, AP_SAT> fx_rnf_sum = rnf_sum;

    ap_uint17_t fx_rnf_sq_2 = (ap_uint17_t)(xFSqrtHOG<17>(fx_rnf_sum)); // Q1.16 format result
    fx_rnf_sq = fx_rnf_sq_2 >> 1;                                       // Q1.15 format (to take in 16 bits)
}

/*****************************************************************************
 * 						 xFDHOGReNormalizeKernel
 *****************************************************************************
 *  This function performs the renormalization operation
 *
 *  norm_block: normalized block data (I)
 *
 *  partial_rnf_sum: temporary variable for re-normalization (I)
 *
 *  _block_strm: output mat containing the normalized data (O)
 *
 *****************************************************************************/

template <int NOHC,
          int NOHCPB,
          int NOVCPB,
          int NOB,
          int ROWS,
          int COLS,
          int DEPTH,
          int NPC,
          int WORDWIDTH,
          typename norm_block_type,
          typename fx_rnf_sq_type>
void xFDHOGReNormalizeKernel(norm_block_type* norm_block,
                             fx_rnf_sq_type fx_rnf_sq,
                             hls::stream<XF_SNAME(WORDWIDTH)>& _block_strm) {
// clang-format off
    #pragma HLS INLINE OFF
    // clang-format on

    char n_rnf;
    uint32_t rnf = xf::cv::Inverse(fx_rnf_sq, 1, &n_rnf); // output in Q(32-n_rnf).n_rnf format

    XF_SNAME(WORDWIDTH) block_data;
    uchar_t step = XF_PIXELDEPTH(DEPTH);
    ap_uint<10> offset = 0;

renorm_loop2:
    for (uchar_t k = 0; k < (NOB * NOHCPB * NOVCPB); k++) {
// clang-format off
        #pragma HLS PIPELINE
        // clang-format on

        ap_uint32_t tmp_block_data = (norm_block[k] * rnf) >> n_rnf; // output in format Q0.16

        // to take care of the MSBs
        if (tmp_block_data > XF_MAX_VAL_16_BIT) tmp_block_data = XF_MAX_VAL_16_BIT;

        // packing the data to the output variable
        block_data.range(offset + (step - 1), offset) = tmp_block_data;
        offset += step;
    }

    _block_strm.write(block_data);
}

/*****************************************************************************
 * 				xFDHOGNormalize
 *****************************************************************************
 * This function acts as a wrapper function for normalize and renormalize
 * functions
 *
 * Inputs - HA arrays, ssv arrays and bi index
 *
 * Outputs - _block_strm (stream)
 *
 *****************************************************************************/
template <int ROWS,
          int COLS,
          int DEPTH,
          int NPC,
          int WORDWIDTH,
          int NOHC,
          int NOHCPB,
          int NOVCPB,
          int NOHW,
          int NOVW,
          int NODPB,
          int WIN_HEIGHT,
          int WIN_WIDTH,
          int CELL_HEIGHT,
          int CELL_WIDTH,
          int NOHB,
          int NOB,
          typename hist_type,
          typename ssv_type>
void xFDHOGNormalize(hist_type HA_1[][NOB],
                     hist_type HA_2[][NOB],
                     ssv_type* ssv_1,
                     ssv_type* ssv_2,
                     hls::stream<XF_SNAME(WORDWIDTH)>& _block_strm,
                     uint16_t bi,
                     uint16_t nohb,
                     uint16_t nohc) {
    // number of horizontal block index
    uint16_t bj = 0;

    ap_uint<26> tmp_nf_sq24_1[1], tmp_nf_sq24_2[1];
    uint32_t nf_1[1], nf_2[1];
    char n_1[1], n_2[1];
// clang-format off
    #pragma HLS RESOURCE variable=nf_1 core=RAM_1P_LUTRAM
    #pragma HLS RESOURCE variable=nf_2 core=RAM_1P_LUTRAM
    #pragma HLS RESOURCE variable=tmp_nf_sq24_1 core=RAM_1P_LUTRAM
    #pragma HLS RESOURCE variable=tmp_nf_sq24_2 core=RAM_1P_LUTRAM
    #pragma HLS RESOURCE variable=n_1 core=RAM_1P_LUTRAM
    #pragma HLS RESOURCE variable=n_2 core=RAM_1P_LUTRAM
    // clang-format on

    // taking each bin as 16-bit unsigned type
    uint16_t norm_block_1[NODPB], norm_block_2[NODPB];
// clang-format off
    #pragma HLS ARRAY_PARTITION variable=norm_block_1 complete dim=1
    #pragma HLS ARRAY_PARTITION variable=norm_block_2 complete dim=1
    // clang-format on

    // temporary variable for the renormalization (to hold the sum value)
    uint16_t fx_rnf_sq_1[1], fx_rnf_sq_2[1];

    bool flag = 1;
    xFDHOGnormalizeKernel1(ssv_1, ssv_2, bj, tmp_nf_sq24_1[0]);

    bj++;
    xFDHOGnormalizeKernel1(ssv_1, ssv_2, bj, tmp_nf_sq24_2[0]);
    xFDHOGnormalizeKernelInv(tmp_nf_sq24_1[0], nf_1[0], n_1[0]);

    bj++;
    xFDHOGnormalizeKernel1(ssv_1, ssv_2, bj, tmp_nf_sq24_1[0]);
    xFDHOGnormalizeKernelInv(tmp_nf_sq24_2[0], nf_2[0], n_2[0]);
    xFDHOGnormalizeKernel2<NOHC, NOHCPB, NOVCPB>(HA_1, HA_2, nf_1[0], norm_block_1, fx_rnf_sq_1[0], bj - 2, n_1[0]);

no_of_HB:
    for (bj = 3; bj < nohb; bj++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=NOHB max=NOHB
        // clang-format on

        if (flag) {
            xFDHOGnormalizeKernel1(ssv_1, ssv_2, bj, tmp_nf_sq24_2[0]);
            xFDHOGnormalizeKernelInv(tmp_nf_sq24_1[0], nf_1[0], n_1[0]);
            xFDHOGnormalizeKernel2<NOHC, NOHCPB, NOVCPB>(HA_1, HA_2, nf_2[0], norm_block_2, fx_rnf_sq_2[0], bj - 2,
                                                         n_2[0]);
            xFDHOGReNormalizeKernel<NOHC, NOHCPB, NOVCPB, NOB, ROWS, COLS, DEPTH, NPC, WORDWIDTH>(
                norm_block_1, fx_rnf_sq_1[0], _block_strm);
            flag = 0;
        }

        else {
            xFDHOGnormalizeKernel1(ssv_1, ssv_2, bj, tmp_nf_sq24_1[0]);
            xFDHOGnormalizeKernelInv(tmp_nf_sq24_2[0], nf_2[0], n_2[0]);
            xFDHOGnormalizeKernel2<NOHC, NOHCPB, NOVCPB>(HA_1, HA_2, nf_1[0], norm_block_1, fx_rnf_sq_1[0], bj - 2,
                                                         n_1[0]);
            xFDHOGReNormalizeKernel<NOHC, NOHCPB, NOVCPB, NOB, ROWS, COLS, DEPTH, NPC, WORDWIDTH>(
                norm_block_2, fx_rnf_sq_2[0], _block_strm);
            flag = 1;
        }
    }

    if (flag) {
        xFDHOGnormalizeKernelInv(tmp_nf_sq24_1[0], nf_1[0], n_1[0]);
        xFDHOGnormalizeKernel2<NOHC, NOHCPB, NOVCPB>(HA_1, HA_2, nf_2[0], norm_block_2, fx_rnf_sq_2[0], bj - 2, n_2[0]);
        xFDHOGReNormalizeKernel<NOHC, NOHCPB, NOVCPB, NOB, ROWS, COLS, DEPTH, NPC, WORDWIDTH>(
            norm_block_1, fx_rnf_sq_1[0], _block_strm);
        flag = 0;
        bj++;
    }

    else {
        xFDHOGnormalizeKernelInv(tmp_nf_sq24_2[0], nf_2[0], n_2[0]);
        xFDHOGnormalizeKernel2<NOHC, NOHCPB, NOVCPB>(HA_1, HA_2, nf_1[0], norm_block_1, fx_rnf_sq_1[0], bj - 2, n_1[0]);
        xFDHOGReNormalizeKernel<NOHC, NOHCPB, NOVCPB, NOB, ROWS, COLS, DEPTH, NPC, WORDWIDTH>(
            norm_block_2, fx_rnf_sq_2[0], _block_strm);
        flag = 1;
        bj++;
    }

    if (flag) {
        xFDHOGnormalizeKernel2<NOHC, NOHCPB, NOVCPB>(HA_1, HA_2, nf_2[0], norm_block_2, fx_rnf_sq_2[0], bj - 2, n_2[0]);
        xFDHOGReNormalizeKernel<NOHC, NOHCPB, NOVCPB, NOB, ROWS, COLS, DEPTH, NPC, WORDWIDTH>(
            norm_block_1, fx_rnf_sq_1[0], _block_strm);
        flag = 0;
    }

    else {
        xFDHOGnormalizeKernel2<NOHC, NOHCPB, NOVCPB>(HA_1, HA_2, nf_1[0], norm_block_1, fx_rnf_sq_1[0], bj - 2, n_1[0]);
        xFDHOGReNormalizeKernel<NOHC, NOHCPB, NOVCPB, NOB, ROWS, COLS, DEPTH, NPC, WORDWIDTH>(
            norm_block_2, fx_rnf_sq_2[0], _block_strm);
        flag = 1;
    }

    if (flag) {
        xFDHOGReNormalizeKernel<NOHC, NOHCPB, NOVCPB, NOB, ROWS, COLS, DEPTH, NPC, WORDWIDTH>(
            norm_block_1, fx_rnf_sq_1[0], _block_strm);
    } else {
        xFDHOGReNormalizeKernel<NOHC, NOHCPB, NOVCPB, NOB, ROWS, COLS, DEPTH, NPC, WORDWIDTH>(
            norm_block_2, fx_rnf_sq_2[0], _block_strm);
    }
}

#endif // _XF_HOG_DESCRIPTOR_NORM_HPP_
