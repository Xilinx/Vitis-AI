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

#ifndef __XF_CORNER_DENSE_TO_SPARSE_HPP__
#define __XF_CORNER_DENSE_TO_SPARSE_HPP__
#include "ap_int.h"
#include "hls_stream.h"
#include "common/xf_common.hpp"
namespace xf {
namespace cv {
template <unsigned int MAXCORNERSNO, unsigned int TYPE, unsigned int ROWS, unsigned int COLS, unsigned int NPC>
void cornersImgToList(xf::cv::Mat<TYPE, ROWS, COLS, NPC>& _src,
                      unsigned int list[MAXCORNERSNO],
                      unsigned int* ncorners) {
    int cornerCount = 0;
    for (unsigned short i = 0; i < _src.rows; i++) {
        for (unsigned short j = 0; j < _src.cols; j++) {
// clang-format off
            #pragma HLS PIPELINE
            // clang-format on
            ap_uint<8> tempValue = _src.read(i * _src.cols + j);
            if (tempValue == 255 && cornerCount < MAXCORNERSNO) // value is 255 if there's a corner
            {
                ap_uint<32> point;
                point.range(31, 16) = i;
                point.range(15, 0) = j;

                list[cornerCount] = (unsigned int)point;
                cornerCount++;
            }
        }
    }
    *ncorners = cornerCount;
    for (int i = cornerCount; i < MAXCORNERSNO; i++) {
// clang-format off
        #pragma HLS PIPELINE
        // clang-format on
        list[i] = 0;
    }
}
} // namespace cv
} // namespace xf
#endif
