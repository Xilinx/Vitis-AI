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
#include <common/xf_aie_hw_utils.hpp>

#ifndef _AIE_PP_MEANSUB_H_
#define _AIE_PP_MEANSUB_H_

#define SHIFT_CNT 8
#include <aie_api/aie.hpp>

/**
 * ----------------------------------------------------------------------------
 * 16-bit Mean Subtraction which is a part of ML pre-process
 * ----------------------------------------------------------------------------
*/
namespace xf {
namespace cv {
namespace aie {

template <int N, typename T>
inline void pp_meansub(const T* restrict img_in,
                       T* restrict img_out,
                       const int16_t img_width,
                       const int16_t img_height,
                       const float alpha) {
    int16_t alpha_q1dot8 = float2fix(alpha, SHIFT_CNT); //(alpha * (1 << 8));

    ::aie::vector<T, N> data_buf, alpha_reg;
    ::aie::accum<acc32, N> alpha_acc;

    for (int i = 0; i < N; i++) {
        alpha_reg[i] = -alpha_q1dot8;
    }
    alpha_acc.from_vector(alpha_reg);

    ::aie::vector<T, N> out_reg, out_reg_shift, in_reg_shift;
    ::aie::accum<acc32, N> acc0;
    set_sat();
    for (int j = 0; j < (img_height * img_width); j += N) // 16 samples per loop
        chess_prepare_for_pipelining

        {
            data_buf = ::aie::load_v<N>(img_in); // in1:00++15|_________|_________|_________
            img_in += N;
            acc0 = ::aie::mac(alpha_acc, data_buf, (1 << SHIFT_CNT));
            out_reg_shift = acc0.template to_vector<int8_t>(SHIFT_CNT).unpack();

            ::aie::store_v(img_out, out_reg_shift); // Write compute pixel to output buffer
            img_out += N;
        }
    clr_sat();
}

__attribute__((noinline)) void pp_meansub_api(input_window_int16* img_in,
                                              output_window_int16* img_out,
                                              const float& alpha) {
    int16_t* img_in_ptr = (int16_t*)img_in->ptr;
    int16_t* img_out_ptr = (int16_t*)img_out->ptr;

    const int16_t img_width = xfGetTileWidth(img_in_ptr);
    const int16_t img_height = xfGetTileHeight(img_in_ptr);

    xfCopyMetaData(img_in_ptr, img_out_ptr);
    xfDefaultSaturation(img_out_ptr);

    int16_t* ptr0 = (int16_t*)xfGetImgDataPtr(img_in_ptr);
    int16_t* ptr_out = (int16_t*)xfGetImgDataPtr(img_out_ptr);

    pp_meansub<16, int16_t>(ptr0, ptr_out, img_width, img_height, alpha);
}

} // aie
} // cv
} // xf
#endif
