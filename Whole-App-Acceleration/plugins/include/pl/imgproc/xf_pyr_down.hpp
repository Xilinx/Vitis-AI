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

#ifndef _XF_PYR_DOWN_
#define _XF_PYR_DOWN_

#include "hls_stream.h"
#include "ap_int.h"
#include "common/xf_common.hpp"
#include "imgproc/xf_pyr_down_gaussian_blur.hpp"

namespace xf {
namespace cv {

template <unsigned int ROWS, unsigned int COLS, unsigned int TYPE, unsigned int NPC, int PLANES, bool USE_URAM>
void xFpyrDownKernel(xf::cv::Mat<TYPE, ROWS, COLS, NPC>& _src,
                     xf::cv::Mat<TYPE, ROWS, COLS, NPC>& _dst,
                     unsigned short in_rows,
                     unsigned short in_cols) {
// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on
    hls::stream<XF_TNAME(TYPE, NPC)> _filter_in;
    hls::stream<XF_TNAME(TYPE, NPC)> _filter_out;
    unsigned int read_pointer = 0;
    for (int i = 0; i < in_rows; i++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=1 max=ROWS
        // clang-format on
        for (int j = 0; j < in_cols; j++) {
// clang-format off
            #pragma HLS LOOP_TRIPCOUNT min=1 max=COLS
            #pragma HLS PIPELINE II=1
            // clang-format on
            _filter_in.write(_src.read(read_pointer));
            read_pointer++;
        }
    }
    xFPyrDownGaussianBlur<ROWS, COLS, TYPE, NPC, XF_WORDWIDTH(TYPE, NPC), 0, 5, 25, PLANES>(
        _filter_in, _filter_out, 5, XF_BORDER_CONSTANT, in_rows, in_cols);

    unsigned int write_ptr = 0;
    for (int i = 0; i < in_rows; i++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=1 max=ROWS
        // clang-format on
        for (int j = 0; j < in_cols; j++) {
// clang-format off
            #pragma HLS LOOP_TRIPCOUNT min=1 max=COLS
            #pragma HLS PIPELINE II=1
            // clang-format on
            XF_TNAME(TYPE, NPC) read_fil_out = _filter_out.read();
            if (i % 2 == 0 && j % 2 == 0) {
                _dst.write(write_ptr, read_fil_out);
                write_ptr++;
            }
        }
    }
    return;
}

template <int TYPE, int ROWS, int COLS, int NPC, bool USE_URAM = false>
void pyrDown(xf::cv::Mat<TYPE, ROWS, COLS, NPC>& _src, xf::cv::Mat<TYPE, ROWS, COLS, NPC>& _dst) {
// clang-format off
    #pragma HLS INLINE OFF
    // clang-format on
    unsigned short input_height = _src.rows;
    unsigned short input_width = _src.cols;
    xFpyrDownKernel<ROWS, COLS, TYPE, NPC, XF_CHANNELS(TYPE, NPC), USE_URAM>(_src, _dst, input_height, input_width);
    return;
}
} // namespace cv
} // namespace xf
#endif
