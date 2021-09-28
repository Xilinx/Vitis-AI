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

#ifndef _AIE_MULTIPLICATION_H_
#define _AIE_MULTIPLICATION_H_

namespace xf {
namespace cv {
namespace aie {

#define SRS_SHIFT 15
/**
 * ----------------------------------------------------------------------------
 * HLI multiplication
 * ----------------------------------------------------------------------------
*/
template <typename T, int N>
__attribute__((noinline)) void multiplication(const T* restrict img_in1,
                                              const T* restrict img_in2,
                                              T* restrict img_out,
                                              const T& img_width,
                                              const T& img_height,
                                              const float& scale) {
    ::aie::vector<T, N> data_buf1;
    ::aie::vector<T, N> data_buf2;
    ::aie::vector<int32_t, N> temp_out;
    ::aie::accum<acc32, N> acc0;
    //    ::aie::vector<T, N> weight(1638,1638,1638,1638,1638,1638,1638,1638,1638,1638,1638,1638,1638,1638,1638,1638);

    T fix_scale = ::aie::to_fixed<float>(scale, SRS_SHIFT); //(float)scale*(1<<15);
    ::aie::vector<T, N> weight = ::aie::broadcast<T, N>(fix_scale);

    for (int j = 0; j < (img_height * img_width); j += 16) // 16x samples per loop
        chess_prepare_for_pipelining chess_loop_range(16, )

        {
            data_buf1 = ::aie::load_v<N>(img_in1); //|00++15
            img_in1 += N;

            data_buf2 = ::aie::load_v<N>(img_in2); //|00++15
            img_in2 += N;

            acc0 = ::aie::mul(data_buf1, data_buf2);

            temp_out = acc0.template to_vector<int32_t>(0);

            acc0 = ::aie::mul(temp_out, weight);

            ::aie::store_v(img_out, acc0.template to_vector<T>(SRS_SHIFT)); // Write compute pixel to output buffer
            img_out += N;
        }
}

/**
 * ----------------------------------------------------------------------------
 * 16-bit multiplication
 * ----------------------------------------------------------------------------
*/

__attribute__((noinline)) void multiplication_api(input_window_int16* restrict img_in,
                                                  input_window_int16* restrict img_in1,
                                                  output_window_int16* restrict img_out,
                                                  const float& scale) {
    int16_t* img_in_ptr = (int16_t*)img_in->ptr;
    int16_t* img_in_ptr1 = (int16_t*)img_in1->ptr;
    int16_t* img_out_ptr = (int16_t*)img_out->ptr;

    const int16_t image_width = xfcvGetTileWidth(img_in_ptr);
    const int16_t image_height = xfcvGetTileHeight(img_in_ptr);

    xfcvCopyMetaData(img_in_ptr, img_out_ptr);

    int16_t* restrict ptr_in = xfcvGetImgDataPtr(img_in_ptr);
    int16_t* restrict ptr_in1 = xfcvGetImgDataPtr(img_in_ptr1);
    int16_t* restrict ptr_out = xfcvGetImgDataPtr(img_out_ptr);

    multiplication<int16_t, 16>(ptr_in, ptr_in1, ptr_out, image_width, image_height, scale);
}

} // aie
} // cv
} // xf

#endif
