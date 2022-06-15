/*
 * Copyright 2020 Xilinx, Inc.
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

#ifndef _XF_QUANTIZATION_DITHERING_
#define _XF_QUANTIZATION_DITHERING_

#include "ap_int.h"
#include "common/xf_common.hpp"
#include "common/xf_utility.hpp"
#include "core/xf_math.h"
#include "hls_stream.h"
#include "math.h"

namespace xf {
namespace cv {

constexpr unsigned XF_LOG2(unsigned x) {
    return (x < 2) ? 0 : 1 + XF_LOG2(x >> 1);
}

template <int OUT_TYPE>
bool isPowerOfTwo(int n) {
    if (n == 0) return 0;
    while (n != 1) {
        if (n % 2 != 0) return 0;
        n = n / 2;
    }
    return 1;
}

template <int IN_TYPE, int OUT_TYPE, int ROWS, int COLS, int SCALE_FACTOR, int MAX_REPRESENTED_VALUE, int NPC>
void xf_QuatizationDithering(xf::cv::Mat<IN_TYPE, ROWS, COLS, NPC>& stream_in,
                             xf::cv::Mat<OUT_TYPE, ROWS, COLS, NPC>& stream_out) {
    enum {
        PLANES = XF_CHANNELS(IN_TYPE, NPC),

        PIXELWIDTH_IN = XF_PIXELWIDTH(IN_TYPE, NPC),
        BITDEPTH_IN = PIXELWIDTH_IN / PLANES,

        PIXELWIDTH_OUT = XF_PIXELWIDTH(OUT_TYPE, NPC),
        BITDEPTH_OUT = PIXELWIDTH_OUT / PLANES,

        QUANTIZATION_INTERVAL = MAX_REPRESENTED_VALUE / SCALE_FACTOR,

        LOG2_SCALE_FACTOR = XF_LOG2(SCALE_FACTOR),
        LOG2_QUANTIZATION_INTERVAL = XF_LOG2(QUANTIZATION_INTERVAL),
        LOG2_MAX_REPRESENTED_VALUE = XF_LOG2(MAX_REPRESENTED_VALUE),

        DEPTH_OFFSETBUFFER = (COLS + (NPC - 1)) / NPC
    };

#ifndef __SYNTHESIS__

    assert(((stream_in.rows <= ROWS) && (stream_in.cols <= COLS)) &&
           "ROWS and COLS should be greater than input image");
    assert(((stream_out.rows <= ROWS) && (stream_out.cols <= COLS)) &&
           "ROWS and COLS should be greater than output image");
    assert(((NPC == XF_NPPC1) || (NPC == XF_NPPC2)) && "The NPC must be XF_NPPC1 or XF_NPPC2");
    assert(((IN_TYPE == XF_8UC1) || (IN_TYPE == XF_8UC3) || (IN_TYPE == XF_10UC1) || (IN_TYPE == XF_10UC3) ||
            (IN_TYPE == XF_12UC1) || (IN_TYPE == XF_12UC3) || (IN_TYPE == XF_16UC1) || (IN_TYPE == XF_16UC3)) &&
           "The IN_TYPE must be XF_8UC1 or XF_8UC3 or XF_10UC1 or XF_10UC3 or "
           "XF_12UC1 or XF_12UC3 or XF_16UC1 or XF_16UC3");

    assert(((OUT_TYPE == XF_8UC1) || (OUT_TYPE == XF_8UC3) || (OUT_TYPE == XF_10UC1) || (OUT_TYPE == XF_10UC3) ||
            (OUT_TYPE == XF_12UC1) || (OUT_TYPE == XF_12UC3) || (OUT_TYPE == XF_16UC1) || (OUT_TYPE == XF_16UC3)) &&
           "The OUT_TYPE must be XF_8UC1 or XF_8UC3 or XF_10UC1 or XF_10UC3 or "
           "XF_12UC1 or XF_12UC3 or XF_16UC1 or XF_16UC3");

    bool scale_power_of_2 = isPowerOfTwo<OUT_TYPE>(SCALE_FACTOR);
    assert((scale_power_of_2 == 1) && "The SCALE_FACTOR must be power of two");

    assert((SCALE_FACTOR <= (1 << BITDEPTH_OUT)) &&
           "The SCALE_FACTOR must be "
           "less than or equal to "
           "2^(output pixel bit width)");
    assert((MAX_REPRESENTED_VALUE == (1 << BITDEPTH_IN)) &&
           "The MAX_REPRESENTED_VALUE must be 2^(input pixel bit width)");
    assert((SCALE_FACTOR <= MAX_REPRESENTED_VALUE) &&
           "The SCALE_FACTOR must be less than or equal to MAX_REPRESENTED_VALUE");

#endif

    unsigned short height = stream_in.rows;
    unsigned short width = stream_in.cols;

    unsigned short imgInput_ncpr = (width + (NPC - 1)) >> XF_BITSHIFT(NPC);

    short in_col_loop_bound = imgInput_ncpr + 1;

    //## offset buffer
    ap_int<BITDEPTH_IN> offset_buffer[PLANES][NPC][DEPTH_OFFSETBUFFER];
// clang-format off
#pragma HLS ARRAY_PARTITION variable = offset_buffer complete dim = 1
#pragma HLS ARRAY_PARTITION variable = offset_buffer complete dim = 2
    // clang-format on

    for (int col_index = 0; col_index < imgInput_ncpr; col_index++) {
// clang-format off
#pragma HLS LOOP_TRIPCOUNT min = COLS / NPC max = COLS / NPC
#pragma HLS PIPELINE II = 1
        // clang-format on
        for (int npc_index = 0; npc_index < NPC; npc_index++) {
// clang-format off
#pragma HLS unroll
            // clang-format on
            for (int channel_index = 0; channel_index < PLANES; channel_index++) {
// clang-format off
#pragma HLS unroll
                // clang-format on
                offset_buffer[channel_index][npc_index][col_index] = 0;
            } // channel_index
        }     // npc_index
    }         // col_index

    ap_int<BITDEPTH_IN> offset_NPC[PLANES][NPC];
// clang-format off
#pragma HLS ARRAY_PARTITION variable = offset_NPC complete dim = 0
    // clang-format on
    ap_int<BITDEPTH_IN> offset_prev_NPC[PLANES][NPC];
// clang-format off
#pragma HLS ARRAY_PARTITION variable = offset_prev_NPC complete dim = 0
    // clang-format on

    int read_index = 0;
    int write_index = 0;

    ap_int<BITDEPTH_IN> q_err_1st[PLANES][NPC];
    ap_int<BITDEPTH_IN> q_err_2nd[PLANES][NPC];
    ap_int<BITDEPTH_IN> q_err_3rd[PLANES][NPC];
// clang-format off
#pragma HLS ARRAY_PARTITION variable = q_err_1st complete dim = 0
#pragma HLS ARRAY_PARTITION variable = q_err_2nd complete dim = 0
#pragma HLS ARRAY_PARTITION variable = q_err_3rd complete dim = 0
    // clang-format on

    // initialize at the beginning for every row
    for (int channel_index = 0; channel_index < PLANES; channel_index++) {
// clang-format off
#pragma HLS unroll
        // clang-format on
        for (int pix_num = 0; pix_num < NPC; pix_num++) {
// clang-format off
#pragma HLS unroll
            // clang-format on
            q_err_1st[channel_index][pix_num] = 0;
            q_err_2nd[channel_index][pix_num] = 0;
            q_err_3rd[channel_index][pix_num] = 0;
        } // npc
    }     // channel_index

LOOP_ROW:
    for (short row_index = 0; row_index < height; row_index++) {
// clang-format off
#pragma HLS LOOP_TRIPCOUNT min = ROWS max = ROWS
    // clang-format on
    LOOP_COL:
        for (short col_index = 0; col_index < in_col_loop_bound; col_index++) {
// clang-format off
#pragma HLS LOOP_TRIPCOUNT min = COLS / NPC max = COLS / NPC
#pragma HLS PIPELINE II = 1
#pragma HLS DEPENDENCE variable = offset_buffer inter false
            // clang-format on

            XF_TNAME(IN_TYPE, NPC) read_word;
            XF_TNAME(OUT_TYPE, NPC) write_word;
            if (col_index < imgInput_ncpr) read_word = stream_in.read(read_index++);

            ap_uint<BITDEPTH_IN> read_word_extract[PLANES][NPC];
// clang-format off
#pragma HLS ARRAY_PARTITION variable = read_word_extract complete dim = 0
            // clang-format on
            for (int pixel = 0, bit1 = 0, bit1_out = 0; pixel < NPC;
                 pixel++, bit1 += (PLANES * BITDEPTH_IN), bit1_out += (PLANES * BITDEPTH_OUT)) {
// clang-format off
#pragma HLS unroll
                // clang-format on
                for (int channel = 0, bit2 = 0, bit2_out = 0; channel < PLANES;
                     channel++, bit2 += BITDEPTH_IN, bit2_out += BITDEPTH_OUT) {
// clang-format off
#pragma HLS unroll
                    // clang-format on
                    ap_uint<BITDEPTH_IN> in_pixel = read_word.range(bit1 + (bit2 + BITDEPTH_IN - 1), bit1 + bit2);

                    ap_int<BITDEPTH_IN + 2> q_2nd_err_scale7 = q_err_2nd[channel][pixel] * 7;
                    ap_int<BITDEPTH_IN - 2> q_2nd_err_scale7by16 = q_2nd_err_scale7.range(BITDEPTH_IN + 1, 4);

                    ap_int<BITDEPTH_IN + 2> quatizer_in =
                        (ap_int<BITDEPTH_IN + 2>)offset_buffer[channel][pixel][col_index] +
                        (ap_int<BITDEPTH_IN + 2>)in_pixel + q_2nd_err_scale7by16;

                    ap_int<BITDEPTH_IN + 2> round_out =
                        (ap_int<LOG2_SCALE_FACTOR + 2>)quatizer_in.range(BITDEPTH_IN + 1, LOG2_QUANTIZATION_INTERVAL) +
                        quatizer_in[LOG2_QUANTIZATION_INTERVAL - 1];
                    ap_int<LOG2_QUANTIZATION_INTERVAL + 1> q_err_3rd_local;
                    q_err_3rd_local.range(LOG2_QUANTIZATION_INTERVAL - 1, 0) =
                        quatizer_in.range(LOG2_QUANTIZATION_INTERVAL - 1, 0);
                    q_err_3rd_local[LOG2_QUANTIZATION_INTERVAL] = quatizer_in[LOG2_QUANTIZATION_INTERVAL - 1];

                    ap_int<BITDEPTH_IN + 4> sum_tmp =
                        q_err_1st[channel][pixel] + q_err_2nd[channel][pixel] * 5 + q_err_3rd_local * 3;
                    offset_NPC[channel][pixel] = sum_tmp.range(BITDEPTH_IN + 3, 4);

                    if (col_index != 0) offset_buffer[channel][pixel][col_index - 1] = offset_prev_NPC[channel][pixel];

                    if (col_index == in_col_loop_bound - 1 && pixel == NPC - 1) {
                        q_err_1st[channel][pixel] = 0;
                        q_err_2nd[channel][pixel] = 0;
                    } else {
                        if (pixel != NPC - 1) {
                            q_err_1st[channel][pixel + 1] = q_err_2nd[channel][pixel];
                            q_err_2nd[channel][pixel + 1] = q_err_3rd_local;
                        } else {
                            q_err_1st[channel][0] = q_err_2nd[channel][pixel];
                            q_err_2nd[channel][0] = q_err_3rd_local;
                        }
                    }

                    ap_uint<BITDEPTH_OUT> out_tmp;
                    if ((col_index == in_col_loop_bound - 1) || (round_out[LOG2_SCALE_FACTOR + 1] == 1)) {
                        out_tmp = 0;
                    } else {
                        if (round_out[LOG2_SCALE_FACTOR] == 0) {
                            out_tmp = (ap_uint<BITDEPTH_OUT>)round_out.range(BITDEPTH_OUT - 1, 0);
                        } else {
                            out_tmp = SCALE_FACTOR - 1;
                        }
                    }

                    write_word.range(bit1_out + (bit2_out + BITDEPTH_OUT - 1), bit1_out + bit2_out) = out_tmp;

                    offset_prev_NPC[channel][pixel] = offset_NPC[channel][pixel];
                }
            }

            if (col_index < imgInput_ncpr) stream_out.write(write_index++, write_word);

        } // LOOP_COL
    }     // LOOP_ROW
}

} // namespace cv
} // namespace xf

#endif //_XF_QUANTIZATION_DITHERING_
