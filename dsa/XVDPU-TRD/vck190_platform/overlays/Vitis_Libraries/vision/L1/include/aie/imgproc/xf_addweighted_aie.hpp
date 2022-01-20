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

#ifndef _AIE_ADDWEIGHTED_H_
#define _AIE_ADDWEIGHTED_H_

#define SHIFT_CNT 10

/**
 * ----------------------------------------------------------------------------
 * 16-bit Accumulate weighted
 * ----------------------------------------------------------------------------
*/

namespace xf {
namespace cv {
namespace aie {

template <typename T, int N>
__attribute__((noinline)) void addweighted(const T* restrict src1,
                                           const T* restrict src2,
                                           T* restrict dst,
                                           const int16_t width,
                                           const int16_t height,
                                           const float& alpha,
                                           const float& beta,
                                           const float& gamma) {
    int16_t alpha_q1dot15 = float2fix(alpha, SHIFT_CNT); //(alpha * (1 << 15));
    int16_t beta_q1dot15 = float2fix(beta, SHIFT_CNT);   //(beta * (1 << 15));
    int16_t gamma_q1dot15 = float2fix(gamma, SHIFT_CNT); //(gamma * (1 << 15));

    ::aie::vector<T, N> coeff(alpha_q1dot15, beta_q1dot15);
    ::aie::vector<T, N> gamma_coeff;
    ::aie::accum<acc32, N> gamma_acc;

    for (int i = 0; i < N; i++) {
        gamma_coeff[i] = gamma_q1dot15;
    }
    gamma_acc.template from_vector(gamma_coeff, 0);

    for (int j = 0; j < width * height; j += N)             // 16 samples per loop
        chess_prepare_for_pipelining chess_loop_range(14, ) // loop_range(14) - loop : 1 cycle

        {
            ::aie::vector<T, N> data_buf1 = ::aie::load_v<16>(src1);
            src1 += N;
            ::aie::vector<T, N> data_buf2 = ::aie::load_v<16>(src2);
            src2 += N;
            ::aie::accum<acc32, N> acc = ::aie::accumulate<16>(
                gamma_acc, coeff, 0, data_buf1, data_buf2); // weight[0] * data_buf1 + weight[1] * data_buf2
            ::aie::store_v(dst, acc.template to_vector<T>(SHIFT_CNT));
            dst += N;
        }
}

void addweighted_api(input_window_int16* img_in1,
                     input_window_int16* img_in2,
                     output_window_int16* img_out,
                     const float& alpha,
                     const float& beta,
                     const float& gamma) {
    int16* ptr0 = (int16*)img_in1->ptr;
    int16* ptr1 = (int16*)img_in2->ptr;
    int16* ptr_out = (int16*)img_out->ptr;

    const int16_t img_width = xfGetTileWidth(ptr0);
    const int16_t img_height = xfGetTileHeight(ptr0);

    xfCopyMetaData(ptr0, ptr_out);
    xfUnsignedSaturation(ptr_out);

    int16* ptr_src1 = (int16*)xfGetImgDataPtr(ptr0);
    int16* ptr_src2 = (int16*)xfGetImgDataPtr(ptr1);
    int16* ptr_dst = (int16*)xfGetImgDataPtr(ptr_out);

    addweighted<int16_t, 16>(ptr_src1, ptr_src2, ptr_dst, img_width, img_height, alpha, beta, gamma);
    /*
    int16_t alpha_q1dot15 = float2fix(alpha, SHIFT_CNT); //(alpha * (1 << 15));
    int16_t beta_q1dot15 = float2fix(beta, SHIFT_CNT);   //(beta * (1 << 15));
    int16_t gamma_q1dot15 = float2fix(gamma, SHIFT_CNT); //(gamma * (1 << 15));

    v16int16 data_buf1;
    v16int16 data_buf2;
    v16int16 gamma_reg;
    v16int16 coeff_v16;
    v16acc48 gamma_out;
    v16acc48 acc;
    // loading accumulator with gama value
    for (int i = 0; i < 16; i++) {
        gamma_reg = upd_elem(gamma_reg, i, gamma_q1dot15);
    }
    gamma_out = ups(gamma_reg, 0);
    // loading alpha, beta into vec register
    coeff_v16 = upd_elem(coeff_v16, 0, alpha_q1dot15);
    coeff_v16 = upd_elem(coeff_v16, 1, beta_q1dot15);

    // process loop

    for (int j = 0; j < img_width * img_height; j += 16)    // 16 samples per loop
        chess_prepare_for_pipelining chess_loop_range(14, ) // loop_range(14) - loop : 1 cycle

        {
            data_buf1 = *(ptr_src1++);
            data_buf2 = *(ptr_src2++);

            acc = mac16(gamma_out, concat(data_buf1, data_buf2), 0, 0x73727170, 0x77767574, 0x3120, coeff_v16, 0, 0, 0,
                        1);

            *(ptr_dst++) = srs(acc, SHIFT_CNT);
        }
*/
}

} // aie
} // cv
} // xf

#endif
