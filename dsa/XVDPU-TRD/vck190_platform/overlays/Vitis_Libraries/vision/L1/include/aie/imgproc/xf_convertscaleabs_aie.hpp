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

#ifndef _AIE_CONVERTSCALEABS_H_
#define _AIE_CONVERTSCALEABS_H_

#define SHIFT_CNT 8
#include <aie_api/aie.hpp>

/**
 * ----------------------------------------------------------------------------
 * 16-bit convertscaleabs
 * ----------------------------------------------------------------------------
*/
namespace xf {
namespace cv {
namespace aie {

template <int N, typename T>
inline void convertscaleabs(const T* restrict img_in,
                            T* restrict img_out,
                            const int16_t img_width,
                            const int16_t img_height,
                            const float alpha,
                            const float beta) {
    int16_t alpha_q1dot15 = float2fix(alpha, SHIFT_CNT); //(alpha * (1 << 15));
    int16_t beta_q1dot15 = float2fix(beta, SHIFT_CNT);   //(beta * (1 << 15));

    ::aie::vector<T, N> data_buf1;
    ::aie::vector<T, N> alpha_reg;
    ::aie::vector<T, N> beta_reg;

    ::aie::accum<acc32, N> acc;
    ::aie::accum<acc32, N> beta_out;

    for (int i = 0; i < 16; i++) {
        beta_reg[i] = beta_q1dot15;
        alpha_reg[i] = alpha_q1dot15;
    }

    beta_out.from_vector(beta_reg);

    set_sat();
    for (int j = 0; j < (img_height * img_width); j += 16) // 32x samples per loop
        chess_prepare_for_pipelining chess_loop_range(14, )

        {
            data_buf1 = ::aie::load_v<N>(img_in); // in1:00++15|_________|_________|_________
            img_in += N;
            acc = ::aie::mac(beta_out, alpha_reg, data_buf1);
            ::aie::store_v(img_out, acc.template to_vector<uint8_t>(SHIFT_CNT).unpack());
            img_out += N;
        }
    clr_sat();
}

void convertscaleabs_api(input_window_int16* img_in,
                         output_window_int16* img_out,
                         const float& alpha,
                         const float& beta) {
    int16_t* img_in_ptr = (int16_t*)img_in->ptr;
    int16_t* img_out_ptr = (int16_t*)img_out->ptr;

    const int16_t img_width = xfGetTileWidth(img_in_ptr);
    const int16_t img_height = xfGetTileHeight(img_in_ptr);

    xfCopyMetaData(img_in_ptr, img_out_ptr);

    int16_t* ptr0 = (int16_t*)xfGetImgDataPtr(img_in_ptr);
    int16_t* ptr_out = (int16_t*)xfGetImgDataPtr(img_out_ptr);

    convertscaleabs<16, int16_t>(ptr0, ptr_out, img_width, img_height, alpha, beta);
}

} // aie
} // cv
} // xf
#endif
