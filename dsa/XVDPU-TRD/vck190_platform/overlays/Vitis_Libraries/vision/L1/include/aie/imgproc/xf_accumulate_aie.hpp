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
#include <common/xf_aie_utils.hpp>

#ifndef _AIE_ACCUMULATEWEIGHTED_H_
#define _AIE_ACCUMULATEWEIGHTED_H_

#define PARALLEL_FACTOR_16b 32
//#define PARALLEL_FACTOR_32b 16

namespace xf {
namespace cv {
namespace aie {

/**
 * ----------------------------------------------------------------------------
 * HLI accumulate
 * ----------------------------------------------------------------------------
*/
template <typename T, int N>
__attribute__((noinline)) void accumulate(const T* restrict img_in1,
                                          const T* restrict img_in2,
                                          T* restrict img_out,
                                          const T& img_width,
                                          const T& img_height) {
    for (int i = 0; i < (img_height * img_width); i += N) // 32x samples per loop
        chess_prepare_for_pipelining chess_loop_range(4, )

        {
            ::aie::vector<T, N> vec = ::aie::add(::aie::load_v<N>(img_in1 + i), ::aie::load_v<N>(img_in2 + i));
            ::aie::store_v(img_out + i, vec);
        }
}

/**
 * ----------------------------------------------------------------------------
 * 16-bit Accumulate
 * ----------------------------------------------------------------------------
 */
__attribute__((noinline)) void accumulate_api(input_window_int16* img_in1,
                                              input_window_int16* img_in2,
                                              output_window_int16* img_out) {
    int16* restrict img_in_ptr = (int16*)img_in1->ptr;
    int16* restrict img_in_ptr1 = (int16*)img_in2->ptr;
    int16* restrict img_out_ptr = (int16*)img_out->ptr;

    const int16_t img_width = xfcvGetTileWidth(img_in_ptr);
    const int16_t img_height = xfcvGetTileHeight(img_in_ptr);

    xfcvCopyMetaData(img_in_ptr, img_out_ptr);
    xfcvUnsignedSaturation(img_out_ptr);

    int16_t* restrict ptr0 = xfcvGetImgDataPtr(img_in_ptr);
    int16_t* restrict ptr1 = xfcvGetImgDataPtr(img_in_ptr1);
    int16_t* restrict ptr_out = xfcvGetImgDataPtr(img_out_ptr);

    accumulate<int16_t, PARALLEL_FACTOR_16b>(ptr0, ptr1, ptr_out, img_width, img_height);
}

} // aie
} // cv
} // xf
#endif
