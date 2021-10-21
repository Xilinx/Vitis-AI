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

#ifndef _AIE_GAINCONTROL_H_
#define _AIE_GAINCONTROL_H_

namespace xf {
namespace cv {
namespace aie {
template <typename T, int N>
inline auto compute_gain_vector_even(const int16_t& gain) {
    ::aie::vector<T, N> coeff;
    for (int i = 0; i < (N / 2); i++) chess_flatten_loop {
            coeff[2 * i] = gain;
            coeff[2 * i + 1] = (1 << 7);
        }
    return coeff;
}

template <typename T, int N>
inline auto compute_gain_vector_odd(const int16_t& gain) {
    ::aie::vector<T, N> coeff;
    for (int i = 0; i < (N / 2); i++) chess_flatten_loop {
            coeff[2 * i] = (1 << 7);
            coeff[2 * i + 1] = gain;
        }
    return coeff;
}

template <typename T, int N, int code>
class ComputeGainVector {};

template <typename T, int N>
class ComputeGainVector<T, N, 0> {
   public:
    // code == 0 : RG
    static inline void compute_gain_kernel_coeff(const int16_t& rgain,
                                                 const int16_t& bgain,
                                                 ::aie::vector<T, N>& coeff0,
                                                 ::aie::vector<T, N>& coeff1) {
        coeff0 = compute_gain_vector_even<T, N>(rgain);
        coeff1 = compute_gain_vector_odd<T, N>(bgain);
    }
};

template <typename T, int N>
class ComputeGainVector<T, N, 1> {
   public:
    // code == 1 : GR
    static inline void compute_gain_kernel_coeff(const int16_t& rgain,
                                                 const int16_t& bgain,
                                                 ::aie::vector<T, N>& coeff0,
                                                 ::aie::vector<T, N>& coeff1) {
        coeff0 = compute_gain_vector_odd<T, N>(rgain);
        coeff1 = compute_gain_vector_even<T, N>(bgain);
    }
};

template <typename T, int N>
class ComputeGainVector<T, N, 2> {
   public:
    // code == 2 : BG
    static inline void compute_gain_kernel_coeff(const int16_t& rgain,
                                                 const int16_t& bgain,
                                                 ::aie::vector<T, N>& coeff0,
                                                 ::aie::vector<T, N>& coeff1) {
        coeff0 = compute_gain_vector_even<T, N>> (bgain);
        coeff1 = compute_gain_vector_odd<T, N>(rgain);
    }
};

template <typename T, int N>
class ComputeGainVector<T, N, 3> {
   public:
    // code == 3 : GB
    static inline void compute_gain_kernel_coeff(const int16_t& rgain,
                                                 const int16_t& bgain,
                                                 ::aie::vector<T, N>& coeff0,
                                                 ::aie::vector<T, N>& coeff1) {
        coeff0 = compute_gain_vector_odd<T, N>(bgain);
        coeff1 = compute_gain_vector_even<T, N>(rgain);
    }
};

template <typename T, int N, int code>
inline void gaincontrol(const T* restrict img_in,
                        T* restrict img_out,
                        int image_width,
                        int image_height,
                        const ::aie::vector<T, N>& coeff0,
                        const ::aie::vector<T, N>& coeff1) {
    auto it = ::aie::begin_vector<N>(img_in);
    auto out = ::aie::begin_vector<N>(img_out);

    for (int i = 0; i < image_height / 2; i++) chess_prepare_for_pipelining chess_loop_range(16, ) {
            for (int j = 0; j < image_width; j += N) // even rows
                chess_prepare_for_pipelining chess_loop_range(16, ) {
                    *out++ = ::aie::mul(coeff0, *it++).template to_vector<T>(7);
                }
            for (int j = 0; j < image_width; j += N) // odd rows
                chess_prepare_for_pipelining chess_loop_range(16, ) {
                    *out++ = ::aie::mul(coeff1, *it++).template to_vector<T>(7);
                }
        }
}

template <int code>
void gaincontrol_api(input_window_int16* img_in,
                     output_window_int16* img_out,
                     const int16_t& rgain,
                     const int16_t& bgain) {
    int16_t* img_in_ptr = (int16_t*)img_in->ptr;
    int16_t* img_out_ptr = (int16_t*)img_out->ptr;

    const int16_t img_width = xfcvGetTileWidth(img_in_ptr);
    const int16_t img_height = xfcvGetTileHeight(img_in_ptr);

    xfcvCopyMetaData(img_in_ptr, img_out_ptr);
    xfcvUnsignedSaturation(img_out_ptr);

    int16_t* in_ptr = (int16_t*)xfcvGetImgDataPtr(img_in_ptr);
    int16_t* out_ptr = (int16_t*)xfcvGetImgDataPtr(img_out_ptr);

    ::aie::vector<int16_t, 16> coeff0;
    ::aie::vector<int16_t, 16> coeff1;
    ComputeGainVector<int16_t, 16, code>::compute_gain_kernel_coeff(rgain, bgain, coeff0, coeff1);
    gaincontrol<int16_t, 16, code>(in_ptr, out_ptr, img_width, img_height, coeff0, coeff1);
}

} // aie
} // cv
} // xf
#endif
