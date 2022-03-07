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

#ifndef _AIE_ABSDIFF_H_
#define _AIE_ABSDIFF_H_

#define PARALLEL_FACTOR_16b 32

namespace xf {
namespace cv {
namespace aie {

/**
 * ----------------------------------------------------------------------------
 * HLI absdiff
 * ----------------------------------------------------------------------------
*/
template <typename T, int N>
__attribute__((noinline)) void absdiff(
    const T* restrict img_in1, const T* restrict img_in2, T* restrict img_out, int image_width, int image_height) {
    auto it1 = ::aie::begin_vector<N>(img_in1);
    auto it2 = ::aie::begin_vector<N>(img_in2);
    auto out = ::aie::begin_vector<N>(img_out);

    while (it1 != img_in1 + image_height * image_width) // 32x samples per loop
        chess_prepare_for_pipelining chess_loop_range(14, ) { *out++ = ::aie::abs(::aie::sub(*it1++, *it2++)); }
}

/**
 * ----------------------------------------------------------------------------
 * 16-bit absdiff
 * ----------------------------------------------------------------------------
*/
void absdiff_api(input_window_int16* img_in1, input_window_int16* img_in2, output_window_int16* img_out) {
    int16* restrict img_in_ptr = (int16*)img_in1->ptr;
    int16* restrict img_in_ptr1 = (int16*)img_in2->ptr;
    int16* restrict img_out_ptr = (int16*)img_out->ptr;

    const int16_t img_width = xfGetTileWidth(img_in_ptr);
    const int16_t img_height = xfGetTileHeight(img_in_ptr);

    xfCopyMetaData(img_in_ptr, img_out_ptr);
    xfUnsignedSaturation(img_out_ptr);

    int16_t* restrict ptr0 = (int16_t*)xfGetImgDataPtr(img_in_ptr);
    int16_t* restrict ptr1 = (int16_t*)xfGetImgDataPtr(img_in_ptr1);
    int16_t* restrict ptr_out = (int16_t*)xfGetImgDataPtr(img_out_ptr);

    absdiff<int16_t, PARALLEL_FACTOR_16b>(ptr0, ptr1, ptr_out, img_width, img_height);
}

} // aie
} // cv
} // xf
#endif
