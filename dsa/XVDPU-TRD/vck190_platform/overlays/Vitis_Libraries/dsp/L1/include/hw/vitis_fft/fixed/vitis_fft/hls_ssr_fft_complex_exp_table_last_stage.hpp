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

// File Name : hls_ssr_fft_ComplexExpTableLastStage.h
#ifndef HLS_SSR_FFT_COMPLEX_EXP_TALBE_LAST_STAGE_H_
#define HLS_SSR_FFT_COMPLEX_EXP_TALBE_LAST_STAGE_H_
#include <ap_fixed.h>

//#define HLS_SSR_FFT_COMPLEX_EXP_TALBE_LAST_STAGE_PRINT_DEBUG_MESSAGES
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
 -_-  is used for initialization of complex exp table, with given length, this spec    -_-
 -_-  is written to deal with the case when actual table length and the length used   -_-
 -_-  in calculating the phase factor is different. HLS have some synthesis issues     -_-
 -_-  for mapping -sin/cose tables to ROM memory for length smaller then 4, so this     -_-
 -_-  spec is created to deal with this case. All tables are kept of length 4 or       -_-
 -_-  larger to deal with this issue while calling the function original length is     -_-
 -_-  passed but internally the minimum table length is used.                           -_-
 -_-                                                                                   -_-
 -_-                                                                                   -_-
 -_-                                                                                   -_-
 -_-                                                                                   -_-
 -_-                                                                                   -_-
 -_-                                                                                   -_-
 -_-                                                                                   -_-
 -_-                                                                                   -_-
 ========================================================================================
 */
#ifdef HLS_SSR_FFT_COMPLEX_EXP_TALBE_LAST_STAGE_PRINT_DEBUG_MESSAGES

#ifndef __SYNTHESIS__
#include <iostream>
#endif

#endif

#include "vitis_fft/hls_ssr_fft_twiddle_table_traits.hpp"

namespace xf {
namespace dsp {
namespace fft {

template <int t_R,
          transform_direction_enum transform_direction,
          butterfly_rnd_mode_enum butterfly_rnd_mode,
          int t_realLength,
          typename T_dtype>
struct ComplexExpTableLastStage {};
// spec-1

template <int t_R,
          transform_direction_enum transform_direction,
          butterfly_rnd_mode_enum butterfly_rnd_mode,
          int t_realLength,
          typename T_dtype>

struct ComplexExpTableLastStage<t_R, transform_direction, butterfly_rnd_mode, t_realLength, std::complex<T_dtype> > {
    static void initComplexExpTable(std::complex<T_dtype> p_table[]) {
        typedef typename TwiddleTypeCastingTraits<std::complex<T_dtype> >::T_roundingBasedCastType T_roundedCastType;

#pragma HLS INLINE off
        static const int EXTENDED_LEN = t_R > 16 ? t_R : 16;
        for (int i = 0; i < EXTENDED_LEN; i++) {
            double real = cos((2 * i * M_PI) / t_realLength);
            double imag;
            if (transform_direction == REVERSE_TRANSFORM)
                imag = sin((2 * i * M_PI) / t_realLength);
            else if (transform_direction == FORWARD_TRANSFORM)
                imag = -sin((2 * i * M_PI) / t_realLength);
            p_table[i] = T_roundedCastType(real, imag);
        }
    }
};
} // end namespace fft
} // end namespace dsp
} // end namespace xf

#endif // HLS_SSR_FFT_COMPLEX_EXP_TALBE_LAST_STAGE_H_
