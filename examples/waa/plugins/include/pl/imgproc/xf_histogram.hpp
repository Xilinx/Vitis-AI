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

#ifndef _XF_HISTOGRAM_HPP_
#define _XF_HISTOGRAM_HPP_

#ifndef __cplusplus
#error C++ is needed to include this header
#endif

#include "common/xf_common.hpp"
#include "hls_stream.h"
namespace xf {
namespace cv {

template <int SRC_T, int ROWS, int COLS, int DEPTH, int NPC, int WORDWIDTH, int SRC_TC, int PLANES>
void xFHistogramKernel(xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _src_mat,
                       uint32_t hist_array[PLANES][256],
                       uint16_t& imgheight,
                       uint16_t& imgwidth) {
    // Temporary array used while computing histogram
    uint32_t tmp_hist[(PLANES << XF_BITSHIFT(NPC))][256];
    uint32_t tmp_hist1[(PLANES << XF_BITSHIFT(NPC))][256];
// clang-format off
#pragma HLS ARRAY_PARTITION variable=tmp_hist complete dim=1
#pragma HLS ARRAY_PARTITION variable=tmp_hist1 complete dim=1
    // clang-format on
    XF_SNAME(WORDWIDTH) in_buf, in_buf1, temp_buf;

    bool flag = 0;

HIST_INITIALIZE_LOOP:
    for (ap_uint<10> i = 0; i < 256; i++) //
    {
// clang-format off
#pragma HLS PIPELINE
        // clang-format on
        for (ap_uint<5> j = 0; j < (1 << XF_BITSHIFT(NPC) * PLANES); j++) {
// clang-format off
#pragma HLS LOOP_TRIPCOUNT min=256 max=256
            // clang-format on
            tmp_hist[j][i] = 0;
            tmp_hist1[j][i] = 0;
        }
    }

HISTOGRAM_ROW_LOOP:
    for (ap_uint<13> row = 0; row < imgheight; row++) {
// clang-format off
#pragma HLS LOOP_TRIPCOUNT min=ROWS max=ROWS
    // clang-format on
    HISTOGRAM_COL_LOOP:
        for (ap_uint<13> col = 0; col < (imgwidth); col = col + 2) {
// clang-format off
#pragma HLS PIPELINE II=2
#pragma HLS LOOP_FLATTEN OFF
#pragma HLS LOOP_TRIPCOUNT min=SRC_TC max=SRC_TC
            // clang-format on
            in_buf = _src_mat.read(row * (imgwidth) + col); //.data[row*(imgwidth) + col];

            if (col == (imgwidth - 1))
                in_buf1 = 0;
            else
                in_buf1 = _src_mat.read(row * (imgwidth) + col + 1); //.data[row*(imgwidth) + col+1];

        EXTRACT_UPDATE:
            for (ap_uint<9> i = 0, j = 0; i < ((8 << XF_BITSHIFT(NPC)) * PLANES); j++, i += 8) {
// clang-format off
#pragma HLS DEPENDENCE variable=tmp_hist array intra false
#pragma HLS DEPENDENCE variable=tmp_hist1 array intra false
#pragma HLS UNROLL
                // clang-format on

                ap_uint<8> val = 0, val1 = 0;
                val = in_buf.range(i + 7, i);
                val1 = in_buf1.range(i + 7, i);

                uint32_t tmpval = tmp_hist[j][val];
                uint32_t tmpval1 = tmp_hist1[j][val1];
                tmp_hist[j][val] = tmpval + 1;
                if (!(col == (imgwidth - 1))) tmp_hist1[j][val1] = tmpval1 + 1;
            }
        }
    }

    const int num_ch = XF_CHANNELS(SRC_T, NPC);

MERGE_HIST_LOOP:
    for (ap_uint<32> i = 0; i < 256; i++) {
// clang-format off
#pragma HLS pipeline
    // clang-format on

    MERGE_HIST_CH_UNROLL:
        for (ap_uint<5> ch = 0; ch < num_ch; ch++) {
// clang-format off
#pragma HLS UNROLL
            // clang-format on

            uint32_t value = 0;

        MERGE_HIST_NPPC_UNROLL:
            for (ap_uint<5> p = 0; p < XF_NPIXPERCYCLE(NPC); p++) {
// clang-format off
#pragma HLS UNROLL
                // clang-format on
                value += tmp_hist[p * num_ch + ch][i] + tmp_hist1[p * num_ch + ch][i];
            }

            hist_array[ch][i] = value;
        }
    }
}

template <int SRC_T, int ROWS, int COLS, int NPC = 1>
void calcHist(xf::cv::Mat<SRC_T, ROWS, COLS, NPC>& _src, uint32_t* histogram) {
#ifndef __SYNTHESIS__
    assert(((NPC == XF_NPPC1) || (NPC == XF_NPPC8)) && "NPC must be XF_NPPC1, XF_NPPC8 ");
    assert(((_src.rows <= ROWS) && (_src.cols <= COLS)) && "ROWS and COLS should be greater than input image");
#endif
// clang-format off
#pragma HLS INLINE OFF
    // clang-format on

    uint32_t hist_array[XF_CHANNELS(SRC_T, NPC)][256] = {0};
    uint16_t width = _src.cols >> (XF_BITSHIFT(NPC));
    uint16_t height = _src.rows;

    xFHistogramKernel<SRC_T, ROWS, COLS, XF_DEPTH(SRC_T, NPC), NPC, XF_WORDWIDTH(SRC_T, NPC),
                      ((COLS >> (XF_BITSHIFT(NPC))) >> 1), XF_CHANNELS(SRC_T, NPC)>(_src, hist_array, height, width);

    for (int i = 0; i < (XF_CHANNELS(SRC_T, NPC)); i++) {
        for (int j = 0; j < 256; j++) {
// clang-format off
#pragma HLS LOOP_TRIPCOUNT min=1 max=256
#pragma HLS PIPELINE
#pragma HLS LOOP_FLATTEN off
            // clang-format on
            histogram[(i * 256) + j] = hist_array[i][j];
        }
    }
}

} // namespace cv
} // namespace xf
#endif // _XF_HISTOGRAM_HPP_
