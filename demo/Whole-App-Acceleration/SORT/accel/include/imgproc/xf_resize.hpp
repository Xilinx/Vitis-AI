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

#ifndef _XF_RESIZE_
#define _XF_RESIZE_

#include "xf_resize_headers.h"

/**
 * Image resizing function.
 */
namespace xf {
namespace cv {

template <int INTERPOLATION_TYPE,
          int TYPE,
          int SRC_ROWS,
          int SRC_COLS,
          int DST_ROWS,
          int DST_COLS,
          int NPC,
          int MAX_DOWN_SCALE>
void resize(xf::cv::Mat<TYPE, SRC_ROWS, SRC_COLS, NPC>& _src, xf::cv::Mat<TYPE, DST_ROWS, DST_COLS, NPC>& _dst) {
// clang-format off
    #pragma HLS INLINE OFF
    // clang-format on

    assert(((INTERPOLATION_TYPE == XF_INTERPOLATION_NN) || (INTERPOLATION_TYPE == XF_INTERPOLATION_BILINEAR) ||
            (INTERPOLATION_TYPE == XF_INTERPOLATION_AREA)) &&
           "Incorrect parameters interpolation type");
    assert(((_src.rows <= SRC_ROWS) && (_src.cols <= SRC_COLS)) &&
           "SRC_ROWS and SRC_COLS should be greater than input image");
    assert(((_dst.rows <= DST_ROWS) && (_dst.cols <= DST_COLS)) &&
           "DST_ROWS and DST_COLS should be greater than output image");

    if (INTERPOLATION_TYPE == XF_INTERPOLATION_AREA) {
        assert((((_src.rows < _dst.rows) && (_src.cols < _dst.cols)) ||
                ((_src.rows >= _dst.rows) && (_src.cols >= _dst.cols))) &&
               " For Area mode, Image can be upscaled or downscaled simultaneously across height & width. But it can't "
               "be upscaled across height & downscaled across width and vice versa.  For example: input image-128x128 "
               "& output image-150x80 is not supported");

        if ((_src.rows < _dst.rows) && (_src.cols < _dst.cols)) {
            xFResizeAreaUpScale<SRC_ROWS, SRC_COLS, XF_CHANNELS(TYPE, NPC), TYPE, NPC, XF_WORDWIDTH(TYPE, NPC),
                                DST_ROWS, DST_COLS, (SRC_COLS >> XF_BITSHIFT(NPC)), (DST_COLS >> XF_BITSHIFT(NPC))>(
                _src, _dst);
        } else if ((_src.rows >= _dst.rows) && (_src.cols >= _dst.cols)) {
            xFResizeAreaDownScale<SRC_ROWS, SRC_COLS, XF_CHANNELS(TYPE, NPC), TYPE, NPC, XF_WORDWIDTH(TYPE, NPC),
                                  DST_ROWS, DST_COLS, (SRC_COLS >> XF_BITSHIFT(NPC)), (DST_COLS >> XF_BITSHIFT(NPC))>(
                _src, _dst);
        }

        return;
    } else {
        resizeNNBilinear<TYPE, SRC_ROWS, SRC_COLS, NPC, DST_ROWS, DST_COLS, INTERPOLATION_TYPE, MAX_DOWN_SCALE>(_src,
                                                                                                                _dst);
    }
}
} // namespace cv
} // namespace xf
#endif
