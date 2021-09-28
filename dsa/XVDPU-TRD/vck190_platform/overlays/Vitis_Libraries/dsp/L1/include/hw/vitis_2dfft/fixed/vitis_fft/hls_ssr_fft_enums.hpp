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

// File Name : hls_ssr_fft_enums.hpp
#ifndef HLS_SSR_FFT_ENUMS_H_
#define HLS_SSR_FFT_ENUMS_H_

/*
 =========================================================================================
 -_-                                                                                   -_-
 -_-                                                                                   -_-
 -_-                                                                                   -_-
 -_-                                                                                   -_-
 -_-                                                                                   -_-
 -_-                                                                                   -_-
 -_-

 scaling_mode_enum is defined to enumerate all of the possible scaling
 modes. 3 different scaling modes are supported.
 1.  SSR_FFT_NO_SCALING        : in this mode no scaling is done
 2.  SSR_FFT_SCALE             : in this mode scaling done in all stages
 3.  SSR_FFT_GROW_TO_MAX_WIDTH : Grow to max width and saturate

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

#include <ap_fixed.h>
//#include <complex>

#include "vitis_fft/fft_complex.hpp"
#define HLS_SSR_FFT_DEFAULT_INSTANCE_ID 999999

namespace xf {
namespace dsp {
namespace fft {
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

/*===================================================================================
 * Enumeration Defines Saturation Modes for SSR FFT Datapath, Specifically related to
 * Butterfly computations. Butterfly computation are essentially Sum of product
 * operation and terminate at an adder tree resulting in bit growth. Different options
 * are available to deal with this bit growth.
 * 1- Do not scale the output and allow progressive bit growth in all FFT stages
 * 2- Do not scale the output and allow progressive bit growth but only until the bit-
 * width reaches 27 bits which is the max number of bit on ultrascale DSP48 slices for
 * multiplier input, so that multiplication operation can fit on these DSPs.
 * 3-Scale the outputs after summation so that there is no effective bit-growth
 * ===================================================================================*/
enum scaling_mode_enum { SSR_FFT_NO_SCALING, SSR_FFT_GROW_TO_MAX_WIDTH, SSR_FFT_SCALE };
enum fft_output_order_enum { SSR_FFT_NATURAL, SSR_FFT_DIGIT_REVERSED_TRANSPOSED };
enum transform_direction_enum { FORWARD_TRANSFORM, REVERSE_TRANSFORM };
enum butterfly_rnd_mode_enum { TRN, CONVERGENT_RND };
enum DataTypeEnum { COMPLEX, REAL, IMAG };
struct ssr_fft_default_params {
    // Scaling Mode Selection

    static const int N = 1024;
    static const int R = 4;
    static const scaling_mode_enum scaling_mode = SSR_FFT_NO_SCALING;
    static const fft_output_order_enum output_data_order = SSR_FFT_NATURAL; // SSR_FFT_DIGIT_REVERSED_TRANSPOSED;
    // Twiddle and Complex Exponential Tables : Effectively sin/cos storage resolution
    static const int twiddle_table_word_length = 18;
    static const int twiddle_table_intger_part_length = 2; // 2 bits are selected to store +1/-1 correctly
    static const int default_t_instanceID = HLS_SSR_FFT_DEFAULT_INSTANCE_ID;
    static const transform_direction_enum transform_direction = FORWARD_TRANSFORM;
    static const butterfly_rnd_mode_enum butterfly_rnd_mode = TRN;
};
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
} // end namespace fft
} // end namespace dsp
} // end namespace xf

#endif // HLS_SSR_FFT_ENUMS_H_
