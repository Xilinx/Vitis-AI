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

#ifndef _AIE_ZERO_H_
#define _AIE_ZERO_H_

namespace xf {
namespace cv {
namespace aie {

/**
 * ----------------------------------------------------------------------------
 * HLI zero
 * ----------------------------------------------------------------------------
*/
template <typename T, int N>
__attribute__((noinline)) void zero(const T* restrict img_in1,
                                    T* restrict img_out,
                                    const T& img_width,
                                    const T& img_height) {
    ::aie::vector<T, N> data_buf1;
    ::aie::vector<T, N> zero_val(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);

    for (int j = 0; j < (img_height * img_width); j += 16) // 32x samples per loop
        chess_prepare_for_pipelining chess_loop_range(16, )

        {
            data_buf1 = ::aie::load_v<N>(img_in1); //|00++15
            img_in1 += N;
            ::aie::store_v(img_out, zero_val); // Write compute pixel to output buffer
            img_out += N;
        }
}

/**
 * ----------------------------------------------------------------------------
 * 16-bit zero
 * ----------------------------------------------------------------------------
*/

__attribute__((noinline)) void zero_api(input_window_int16* restrict img_in, output_window_int16* restrict img_out) {
    int16_t* img_in_ptr = (int16_t*)img_in->ptr;
    int16_t* img_out_ptr = (int16_t*)img_out->ptr;

    const int16_t image_width = xfGetTileWidth(img_in_ptr);
    const int16_t image_height = xfGetTileHeight(img_in_ptr);

    xfCopyMetaData(img_in_ptr, img_out_ptr);

    int16_t* restrict ptr_in = (int16_t*)xfGetImgDataPtr(img_in_ptr);
    int16_t* restrict ptr_out = (int16_t*)xfGetImgDataPtr(img_out_ptr);

    zero<int16_t, 16>(ptr_in, ptr_out, image_width, image_height);
}

} // aie
} // cv
} // xf

#endif
