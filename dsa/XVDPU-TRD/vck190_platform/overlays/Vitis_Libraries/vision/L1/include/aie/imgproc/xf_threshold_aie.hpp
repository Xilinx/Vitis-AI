/*
 * Copyright 2021 Xilinx, Inc.
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

#include <adf.h>
#include <aie_api/aie.hpp>
#include <common/xf_aie_hw_utils.hpp>

#ifndef _AIE_THRESHOLD_H_
#define _AIE_THRESHOLD_H_

namespace xf {
namespace cv {
namespace aie {

/**
 * ----------------------------------------------------------------------------
 * HLI Threshold
 * ----------------------------------------------------------------------------
*/
template <typename T, int N>
__attribute__((noinline)) void threshold(
    T* img_in, T* img_out, const T& img_width, const T& img_height, const T& thresh_val, const T& max_val) {
    ::aie::vector<T, N> constants;
    ::aie::vector<T, N> data_out;
    ::aie::mask<N> temp_val;
    constants[0] = 0;          // updating constant zero_val value
    constants[1] = thresh_val; // updating constant threshold value
    constants[2] = max_val;    // updating constant max_val value

    for (int j = 0; j < (img_height * img_width); j += N) // 16x samples per loop
        chess_prepare_for_pipelining chess_loop_range(14, ) {
            ::aie::vector<T, N> data_buf1 = ::aie::load_v(img_in); // in:00++15|_________|_________|_________
            img_in += N;
            switch (THRESH_TYPE) {
                case XF_THRESHOLD_TYPE_TRUNC:
                    data_out = ::aie::min(constants[1], data_buf1);
                    break;
                case XF_THRESHOLD_TYPE_BINARY:
                    temp_val = ::aie::lt(constants[1], data_buf1);
                    data_out = ::aie::select(constants[0], constants[2], temp_val);
                    break;
                case XF_THRESHOLD_TYPE_BINARY_INV:
                    temp_val = ::aie::lt(constants[1], data_buf1);
                    data_out = ::aie::select(constants[2], constants[0], temp_val);
                    break;
                case XF_THRESHOLD_TYPE_TOZERO:
                    temp_val = ::aie::lt(constants[1], data_buf1);
                    data_out = ::aie::select(constants[0], data_buf1, temp_val);
                    break;
                case XF_THRESHOLD_TYPE_TOZERO_INV:
                    temp_val = ::aie::lt(constants[1], data_buf1);
                    data_out = ::aie::select(data_buf1, constants[0], temp_val);
                    break;

                default:
                    data_out = ::aie::min(constants[1], data_buf1);
            }
            ::aie::store_v(img_out, data_out);
            img_out += N;
        }
}

/**
 * ----------------------------------------------------------------------------
 * 16-bit Threshold
 * ----------------------------------------------------------------------------
*/

__attribute__((noinline)) void threshold_api(input_window_int16* restrict img_in,
                                             output_window_int16* restrict img_out,
                                             const int16& thresh_val,
                                             const int16& max_val) {
    int16_t* img_in_ptr = (int16_t*)img_in->ptr;
    int16_t* img_out_ptr = (int16_t*)img_out->ptr;

    const int16_t image_width = xfGetTileWidth(img_in_ptr);
    const int16_t image_height = xfGetTileHeight(img_in_ptr);

    xfCopyMetaData(img_in_ptr, img_out_ptr);

    int16_t* restrict ptr_in = (int16_t*)xfGetImgDataPtr(img_in_ptr);
    int16_t* restrict ptr_out = (int16_t*)xfGetImgDataPtr(img_out_ptr);

    threshold<int16_t, 32>(ptr_in, ptr_out, image_width, image_height, thresh_val, max_val);
}

} // aie
} // cv
} // xf

#endif
