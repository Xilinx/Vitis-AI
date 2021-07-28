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

#ifndef _XF_PYR_UP_
#define _XF_PYR_UP_

#include "ap_int.h"
#include "hls_stream.h"
#include "imgproc/xf_pyr_up_gaussian_blur.hpp"
#include "common/xf_common.hpp"

namespace xf {
namespace cv {

template <unsigned int ROWS, unsigned int COLS, unsigned int NPC, unsigned int DEPTH, int PLANES>
void xFpyrUpKernel(xf::cv::Mat<DEPTH, ROWS, COLS, NPC>& _src,
                   xf::cv::Mat<DEPTH, 2 * ROWS, 2 * COLS, NPC>& _dst,
                   unsigned short in_rows,
                   unsigned short in_cols) {
// clang-format off
    #pragma HLS INLINE OFF
    #pragma HLS DATAFLOW
    // clang-format on
    hls::stream<XF_TNAME(DEPTH, NPC)> _filter_in;
    hls::stream<XF_TNAME(DEPTH, NPC)> _filter_out;

    unsigned short output_height = in_rows << 1;
    unsigned short output_width = in_cols << 1;
    int read_pointer = 0, write_pointer = 0;
    for (int i = 0; i < output_height; i++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=1 max=ROWS
        // clang-format on
        for (int j = 0; j < output_width; j++) {
// clang-format off
            #pragma HLS LOOP_TRIPCOUNT min=1 max=COLS
            #pragma HLS PIPELINE II=1
            #pragma HLS LOOP_FLATTEN OFF
            // clang-format on
            XF_TNAME(DEPTH, NPC) read_input;
            if (i % 2 == 0 && j % 2 == 0) {
                read_input = _src.read(read_pointer); //*(in_image + read_pointer);
                read_pointer++;
            } else
                read_input = 0;
            _filter_in.write(read_input);
        }
    }
    xFPyrUpGaussianBlur<2 * ROWS, 2 * COLS, DEPTH, NPC, 0, 0, 5, 25, PLANES>(
        _filter_in, _filter_out, 5, XF_BORDER_DEFAULT, output_height, output_width);

    for (int i = 0; i < output_height; i++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=1 max=ROWS
        // clang-format on
        for (int j = 0; j < output_width; j++) {
// clang-format off
            #pragma HLS LOOP_TRIPCOUNT min=1 max=COLS
            #pragma HLS PIPELINE II=1
            #pragma HLS LOOP_FLATTEN OFF
            // clang-format on
            //*(out_image + write_pointer) = _filter_out.read();
            _dst.write(write_pointer, (_filter_out.read()));
            write_pointer++;
        }
    }

    return;
}

template <int TYPE, int ROWS, int COLS, int NPC = 1>
void pyrUp(xf::cv::Mat<TYPE, ROWS, COLS, NPC>& _src, xf::cv::Mat<TYPE, 2 * ROWS, 2 * COLS, NPC>& _dst) {
// clang-format off
    #pragma HLS INLINE OFF
    // clang-format on
    unsigned short input_height = _src.rows;
    unsigned short input_width = _src.cols;

    xFpyrUpKernel<ROWS, COLS, NPC, TYPE, XF_CHANNELS(TYPE, NPC)>(_src, _dst, input_height, input_width);

    return;
}
} // namespace cv
} // namespace xf
#endif
