/*
 * Copyright 2019 Xilinx, Inc.
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

// File Name :  hls_ssr_fft_complex_exp_table.hpp
#ifndef HLS_SSR_FFT_COMPLEX_EXP_TALBE_H_
#define HLS_SSR_FFT_COMPLEX_EXP_TALBE_H_

//#define HLS_SSR_FFT_COMPLEX_EXP_TALBE_PRINT_DEBUG_MESSAGES
//#define HLS_SSR_FFT_DEBUG
/*
 =========================================================================================
 -_-                                                                                   -_-
 -_-                                                                                   -_-
 -_-                                                                                   -_-
 -_-                                                                                   -_-
 -_-                                                                                   -_-
 -_-                                                                                   -_-
 -_-  initComplexExpTable Specialization is defined here, in general this function     -_-
 -_-  is used for initialization of complex exp table, with given length,the template  -_-
 -_-  parameter is named t_R that captures the length of table because for ssr fft    -_-
 -_-  the complex exponential table has a length equal to radix of fft being           -_-
 -_-  calculated.                                                                      -_-
 -_-                                                                                   -_-
 -_-                                                                                   -_-
 -_-                                                                                   -_-
 -_-                                                                                   -_-
 -_-                                                                                   -_-
 ========================================================================================
 */

#ifdef HLS_SSR_FFT_COMPLEX_EXP_TALBE_PRINT_DEBUG_MESSAGES
#ifndef __SYNTHESIS__
#include <iostream>
#endif
#endif
#include <math.h>

#include "vitis_fft/hls_ssr_fft_twiddle_table_traits.hpp"

#include <ap_fixed.h>

namespace xf {
namespace dsp {
namespace fft {

template <int t_R,
          transform_direction_enum transform_direction,
          butterfly_rnd_mode_enum butterfly_rnd_mode,
          typename T_dtype>
struct ComplexExpTable {};
template <int t_R,
          transform_direction_enum transform_direction,
          butterfly_rnd_mode_enum butterfly_rnd_mode,
          typename T_dtype>
struct ComplexExpTable<t_R, transform_direction, butterfly_rnd_mode, std::complex<T_dtype> > {
    static void initComplexExpTable(std::complex<T_dtype> p_table[]) {
#pragma HLS INLINE off
        typedef typename TwiddleTypeCastingTraits<std::complex<T_dtype> >::T_roundingBasedCastType T_roundedCastType;
        static const int INIT_LEN = t_R > 16 ? t_R : 16;
        for (int i = 0; i < INIT_LEN; i++) {
            double real = cos((2 * i * M_PI) / t_R);

            double imag;
            if (transform_direction == REVERSE_TRANSFORM)
                imag = sin((2 * i * M_PI) / t_R);
            else if (transform_direction == FORWARD_TRANSFORM)
                imag = -sin((2 * i * M_PI) / t_R);

            p_table[i] = T_roundedCastType(real, imag);
        }
    }
};

} // end namespace fft
} // end namespace dsp
} // end namespace xf

#endif // HLS_SSR_FFT_COMPLEX_EXP_TALBE_H_
