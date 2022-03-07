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

#ifndef _AIE_PP_ALL_OP_H_
#define _AIE_PP_ALL_OP_H_

#define SHIFT_CNT 4
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
__attribute__((noinline)) void pp_all_op(const T* __restrict img_in,
                                         T* __restrict img_out,
                                         const int16_t img_width,
                                         const int16_t img_height,
                                         const float alpha,
                                         const float beta,
                                         const float gamma) {
    // The expression to be implemented is (x - a) * b + c which is rearranged as (c - a * b) + x *b -> implemented
    // using a single mac

    // compute the constant expression and convert to fixed point
    const int16_t gamma_minus_alphabeta_q1dot8 = ::aie::to_fixed(::aie::msc(gamma, alpha, beta), SHIFT_CNT);
    const int16_t beta_q1dot8 = ::aie::to_fixed(beta, SHIFT_CNT);

    auto it_in = ::aie::cbegin_vector<N>(img_in);
    auto it_out = ::aie::begin_restrict_vector<N>(img_out);

    ::aie::accum<acc32, N> gamma_minus_alphabeta_acc;
    ::aie::vector<T, N> out_reg_shift;

    gamma_minus_alphabeta_acc.from_vector(::aie::broadcast<T, N>(gamma_minus_alphabeta_q1dot8));
    set_sat();
    for (int j = 0; j < (img_height * img_width); j += N) // 16 samples per loop
        chess_prepare_for_pipelining chess_loop_range(14, ) {
            ::aie::accum<acc32, N> acc0;

            acc0 = ::aie::mac(gamma_minus_alphabeta_acc, *it_in++, beta_q1dot8);

            out_reg_shift = acc0.template to_vector<int8_t>(SHIFT_CNT).unpack();

            ::aie::store_v(img_out, out_reg_shift); // Write compute pixel to output buffer
            img_out += N;
        }
    clr_sat();
}

__attribute__((noinline)) void pp_all_op_api(input_window_int16* img_in,
                                             output_window_int16* img_out,
                                             const float& alpha,
                                             const float& beta,
                                             const float& gamma) {
    int16_t* img_in_ptr = (int16_t*)img_in->ptr;
    int16_t* img_out_ptr = (int16_t*)img_out->ptr;

    const int16_t img_width = xfGetTileWidth(img_in_ptr);
    const int16_t img_height = xfGetTileHeight(img_in_ptr);

    xfCopyMetaData(img_in_ptr, img_out_ptr);
    xfDefaultSaturation(img_out_ptr);

    int16_t* ptr0 = (int16_t*)xfGetImgDataPtr(img_in_ptr);
    int16_t* ptr_out = (int16_t*)xfGetImgDataPtr(img_out_ptr);

    pp_all_op<16, int16_t>(ptr0, ptr_out, img_width, img_height, alpha, beta, gamma);
}

} // aie
} // cv
} // xf
#endif
