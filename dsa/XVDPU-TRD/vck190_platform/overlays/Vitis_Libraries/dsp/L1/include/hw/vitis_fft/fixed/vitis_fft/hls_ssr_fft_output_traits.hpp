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

// File Name : hls_ssr_fft_output_traits.hpp
#ifndef HLS_SSR_FFT_OUTPUT_TRAITS_H_
#define HLS_SSR_FFT_OUTPUT_TRAITS_H_

#include <ap_fixed.h>
#include <complex>

#include "vitis_fft/hls_ssr_fft_utilities.hpp"
#include "vitis_fft/hls_ssr_fft_enums.hpp"
#include "vitis_fft/hls_ssr_fft_dsp48.hpp"
#include "vitis_fft/hls_ssr_fft_input_traits.hpp"
#include "vitis_fft/fft_complex.hpp"

/*
 =========================================================================================
 -_- -_-
 -_- -_-
 -_- -_-
 -_- -_-
 -_- -_-
 -_- -_-
 -_-
 FFTOutputTraits struct is defined in this file. It defines trait
 for SSR FFT output. Given the input type, scaling mode and t_L and t_R
 it can be used to find :
 1- inner type used for std::complex output
 2- complex type which will be used by output std::complex<ap_fixed<>>

 -_- -_-
 -_- -_-
 -_- -_-
 -_- -_-
 -_- -_-
 -_- -_-
 -_- -_-
 -_- -_-
 ========================================================================================
 */

namespace xf {
namespace dsp {
namespace fft {

template <int t_L,
          int t_R,
          scaling_mode_enum t_scalingMode,
          transform_direction_enum transform_direction,
          butterfly_rnd_mode_enum butterfly_rnd_mode,
          typename T_in>
struct FFTOutputTraits {};

//==============================================Float Type========================================
template <int t_L,
          int t_R,
          scaling_mode_enum t_scalingMode,
          transform_direction_enum transform_direction,
          butterfly_rnd_mode_enum butterfly_rnd_mode>
struct FFTOutputTraits<t_L, t_R, t_scalingMode, transform_direction, butterfly_rnd_mode, std::complex<float> > {
    typedef std::complex<float> T_FFTOutType;
    typedef float T_innerFFTOutType;

    static const int type = -1;
};
//==============================================Double Type========================================
template <int t_L,
          int t_R,
          scaling_mode_enum t_scalingMode,
          transform_direction_enum transform_direction,
          butterfly_rnd_mode_enum butterfly_rnd_mode>
struct FFTOutputTraits<t_L, t_R, t_scalingMode, transform_direction, butterfly_rnd_mode, std::complex<double> > {
    typedef std::complex<double> T_FFTOutType;
    typedef double T_innerFFTOutType;

    static const int type = 0;
};

//==============================================ap_fixed Type:: generic========================================
template <int t_L,
          int t_R,
          scaling_mode_enum t_scalingMode,
          transform_direction_enum transform_direction,
          butterfly_rnd_mode_enum butterfly_rnd_mode,
          int t_inputSizeBits,
          int t_integerPartBits>
struct FFTOutputTraits<t_L,
                       t_R,
                       t_scalingMode,
                       transform_direction,
                       butterfly_rnd_mode,
                       std::complex<ap_fixed<t_inputSizeBits, t_integerPartBits> > > {
    static const int type = 1;
};
template <int t_L,
          int t_R,
          scaling_mode_enum t_scalingMode,
          transform_direction_enum transform_direction,
          butterfly_rnd_mode_enum butterfly_rnd_mode,
          int t_inputSizeBits,
          int t_integerPartBits,
          ap_q_mode t_inputQuantizationMode,
          ap_o_mode t_inputOverflowMode,
          int t_inNumOfSatBits>
struct FFTOutputTraits<t_L,
                       t_R,
                       t_scalingMode,
                       transform_direction,
                       butterfly_rnd_mode,
                       std::complex<ap_fixed<t_inputSizeBits,
                                             t_integerPartBits,
                                             t_inputQuantizationMode,
                                             t_inputOverflowMode,
                                             t_inNumOfSatBits> > > {
    static const int type = 2;
};

//==============================================ap_fixed Type && SSR_FFT_NO_SCALING==================================
template <int t_L,
          int t_R,
          transform_direction_enum transform_direction,
          butterfly_rnd_mode_enum butterfly_rnd_mode,
          int t_inputSizeBits,
          int t_integerPartBits>
struct FFTOutputTraits<t_L,
                       t_R,
                       SSR_FFT_NO_SCALING,
                       transform_direction,
                       butterfly_rnd_mode,
                       std::complex<ap_fixed<t_inputSizeBits, t_integerPartBits> > > {
    static const int tp_log2R = ssrFFTLog2<t_R>::val;
    static const int OUTPUT_WL = t_inputSizeBits + (ssrFFTLog2<t_L>::val) + 1;
    static const int OUTPUT_IL_FFT = t_integerPartBits + (ssrFFTLog2<t_L>::val) + 1;
    static const int IFFT_SCALING_BITS = (ssrFFTLog2<t_L>::val);
    static const int OUTPUT_IL =
        (transform_direction == FORWARD_TRANSFORM) ? OUTPUT_IL_FFT : OUTPUT_IL_FFT - IFFT_SCALING_BITS;

    typedef std::complex<ap_fixed<OUTPUT_WL, OUTPUT_IL, AP_TRN, AP_WRAP, 0> > T_FFTOutType;
    typedef ap_fixed<OUTPUT_WL, OUTPUT_IL, AP_TRN, AP_WRAP, 0> T_innerFFTOutType;
    static const int type = 3;
};

template <int t_L,
          int t_R,
          transform_direction_enum transform_direction,
          butterfly_rnd_mode_enum butterfly_rnd_mode,
          int t_inputSizeBits,
          int t_integerPartBits,
          ap_q_mode t_inputQuantizationMode,
          ap_o_mode t_inputOverflowMode,
          int t_inNumOfSatBits>
struct FFTOutputTraits<t_L,
                       t_R,
                       SSR_FFT_NO_SCALING,
                       transform_direction,
                       butterfly_rnd_mode,
                       std::complex<ap_fixed<t_inputSizeBits,
                                             t_integerPartBits,
                                             t_inputQuantizationMode,
                                             t_inputOverflowMode,
                                             t_inNumOfSatBits> > > {
    static const int tp_log2R = ssrFFTLog2<t_R>::val;

    static const int OUTPUT_WL = t_inputSizeBits + (ssrFFTLog2<t_L>::val) + 1;
    static const int OUTPUT_IL_FFT = t_integerPartBits + (ssrFFTLog2<t_L>::val) + 1;
    static const int IFFT_SCALING_BITS = (ssrFFTLog2<t_L>::val);
    static const int OUTPUT_IL =
        (transform_direction == FORWARD_TRANSFORM) ? OUTPUT_IL_FFT : OUTPUT_IL_FFT - IFFT_SCALING_BITS;
    typedef std::complex<
        ap_fixed<OUTPUT_WL, OUTPUT_IL, t_inputQuantizationMode, t_inputOverflowMode, t_inNumOfSatBits> >
        T_FFTOutType;
    typedef ap_fixed<OUTPUT_WL, OUTPUT_IL, t_inputQuantizationMode, t_inputOverflowMode, t_inNumOfSatBits>
        T_innerFFTOutType;

    static const int type = 4;
};
//==============================================ap_fixed Type && SSR_FFT_SCALE=======================================

template <int t_L,
          int t_R,
          transform_direction_enum transform_direction,
          butterfly_rnd_mode_enum butterfly_rnd_mode,
          int t_inputSizeBits,
          int t_integerPartBits>
struct FFTOutputTraits<t_L,
                       t_R,
                       SSR_FFT_SCALE,
                       transform_direction,
                       butterfly_rnd_mode,
                       std::complex<ap_fixed<t_inputSizeBits, t_integerPartBits> > > {
    static const int tp_log2R = ssrFFTLog2<t_R>::val;

    static const int OUTPUT_WL = t_inputSizeBits;
    static const int OUTPUT_IL_FFT = t_integerPartBits + ((ssrFFTLog2<t_L>::val)) + 1;
    static const int IFFT_SCALING_BITS = (ssrFFTLog2<t_L>::val);
    static const int OUTPUT_IL =
        (transform_direction == FORWARD_TRANSFORM) ? OUTPUT_IL_FFT : OUTPUT_IL_FFT - IFFT_SCALING_BITS;

    typedef std::complex<ap_fixed<t_inputSizeBits, OUTPUT_IL, AP_TRN, AP_WRAP, 0> > T_FFTOutType;
    typedef ap_fixed<t_inputSizeBits, OUTPUT_IL, AP_TRN, AP_WRAP, 0> T_innerFFTOutType;

    static const int type = 5;
};
template <int t_L,
          int t_R,
          transform_direction_enum transform_direction,
          butterfly_rnd_mode_enum butterfly_rnd_mode,
          int t_inputSizeBits,
          int t_integerPartBits,
          ap_q_mode t_inputQuantizationMode,
          ap_o_mode t_inputOverflowMode,
          int t_inNumOfSatBits>
struct FFTOutputTraits<t_L,
                       t_R,
                       SSR_FFT_SCALE,
                       transform_direction,
                       butterfly_rnd_mode,
                       std::complex<ap_fixed<t_inputSizeBits,
                                             t_integerPartBits,
                                             t_inputQuantizationMode,
                                             t_inputOverflowMode,
                                             t_inNumOfSatBits> > > {
    static const int tp_log2R = ssrFFTLog2<t_R>::val;

    static const int OUTPUT_WL = t_inputSizeBits;
    static const int OUTPUT_IL_FFT = t_integerPartBits + ((ssrFFTLog2<t_L>::val)) + 1;
    static const int IFFT_SCALING_BITS = (ssrFFTLog2<t_L>::val);
    static const int OUTPUT_IL =
        (transform_direction == FORWARD_TRANSFORM) ? OUTPUT_IL_FFT : OUTPUT_IL_FFT - IFFT_SCALING_BITS;
    typedef std::complex<
        ap_fixed<OUTPUT_WL, OUTPUT_IL, t_inputQuantizationMode, t_inputOverflowMode, t_inNumOfSatBits> >
        T_FFTOutType;
    typedef ap_fixed<OUTPUT_WL, OUTPUT_IL, t_inputQuantizationMode, t_inputOverflowMode, t_inNumOfSatBits>
        T_innerFFTOutType;

    static const int type = 6;
};
//===============================================SSR_FFT_GROW_TO_MAX_WIDTH===========================================
template <int t_L,
          int t_R,
          transform_direction_enum transform_direction,
          butterfly_rnd_mode_enum butterfly_rnd_mode,
          int t_inputSizeBits,
          int t_integerPartBits>
struct FFTOutputTraits<t_L,
                       t_R,
                       SSR_FFT_GROW_TO_MAX_WIDTH,
                       transform_direction,
                       butterfly_rnd_mode,
                       std::complex<ap_fixed<t_inputSizeBits, t_integerPartBits> > > {
    static const int MAX_POSSIBLE_BIT_GROWTH = ssrFFTLog2<t_L>::val + 1;

    static const int tp_log2R = ssrFFTLog2<t_R>::val;
    static const int EXPECTED_OUTPUT_BIT_WIDTH = (t_inputSizeBits + (MAX_POSSIBLE_BIT_GROWTH));
    static const int EXPECTED_OUTPUT_INTEGER_WIDTH = (t_integerPartBits + (MAX_POSSIBLE_BIT_GROWTH));
    static const int MAX_ALLOWED_BIT_WIDTH_MARGIN = DSP48_OP2_BIT_WIDTH - t_inputSizeBits;

    // If the expected grown output has bit width larger then max allowed saturate it.
    static const int O_WL =
        (EXPECTED_OUTPUT_BIT_WIDTH <= DSP48_OP2_BIT_WIDTH) ? EXPECTED_OUTPUT_BIT_WIDTH : DSP48_OP2_BIT_WIDTH;
    // If the expected  grown integer part width is larger then max allowed saturate it with any margin if left.
    static const int O_IL_FFT =
        (EXPECTED_OUTPUT_INTEGER_WIDTH <= DSP48_OP2_BIT_WIDTH) ? EXPECTED_OUTPUT_INTEGER_WIDTH : DSP48_OP2_BIT_WIDTH;

    static const int IFFT_SCALING_BITS = (ssrFFTLog2<t_L>::val);
    static const int O_IL = (transform_direction == FORWARD_TRANSFORM) ? O_IL_FFT : O_IL_FFT - IFFT_SCALING_BITS;

    typedef std::complex<ap_fixed<O_WL, O_IL, AP_TRN, AP_WRAP, 0> > T_FFTOutType;
    typedef ap_fixed<O_WL, O_IL, AP_TRN, AP_WRAP, 0> T_innerFFTOutType;

    static const int type = 7;
};

template <int t_L,
          int t_R,
          transform_direction_enum transform_direction,
          butterfly_rnd_mode_enum butterfly_rnd_mode,
          int t_inputSizeBits,
          int t_integerPartBits,
          ap_q_mode t_inputQuantizationMode,
          ap_o_mode t_inputOverflowMode,
          int t_inNumOfSatBits>
struct FFTOutputTraits<t_L,
                       t_R,
                       SSR_FFT_GROW_TO_MAX_WIDTH,
                       transform_direction,
                       butterfly_rnd_mode,
                       std::complex<ap_fixed<t_inputSizeBits,
                                             t_integerPartBits,
                                             t_inputQuantizationMode,
                                             t_inputOverflowMode,
                                             t_inNumOfSatBits> > > {
    static const int MAX_POSSIBLE_BIT_GROWTH = ssrFFTLog2<t_L>::val + 1;
    static const int tp_log2R = ssrFFTLog2<t_R>::val;
    static const int EXPECTED_OUTPUT_BIT_WIDTH = (t_inputSizeBits + (MAX_POSSIBLE_BIT_GROWTH));
    static const int EXPECTED_OUTPUT_INTEGER_WIDTH = (t_integerPartBits + (MAX_POSSIBLE_BIT_GROWTH));
    static const int MAX_ALLOWED_BIT_WIDTH_MARGIN = DSP48_OP2_BIT_WIDTH - t_inputSizeBits;

    static const int O_WL =
        (EXPECTED_OUTPUT_BIT_WIDTH <= DSP48_OP2_BIT_WIDTH) ? EXPECTED_OUTPUT_BIT_WIDTH : DSP48_OP2_BIT_WIDTH;
    static const int O_IL_FFT =
        (EXPECTED_OUTPUT_INTEGER_WIDTH <= DSP48_OP2_BIT_WIDTH) ? EXPECTED_OUTPUT_INTEGER_WIDTH : DSP48_OP2_BIT_WIDTH;

    static const int IFFT_SCALING_BITS = (ssrFFTLog2<t_L>::val);
    static const int O_IL = (transform_direction == FORWARD_TRANSFORM) ? O_IL_FFT : O_IL_FFT - IFFT_SCALING_BITS;
    typedef std::complex<ap_fixed<O_WL, O_IL, t_inputQuantizationMode, t_inputOverflowMode, t_inNumOfSatBits> >
        T_FFTOutType;
    typedef ap_fixed<O_WL, O_IL, t_inputQuantizationMode, t_inputOverflowMode, t_inNumOfSatBits> T_innerFFTOutType;

    static const int type = 8;
};

/*
 * =======Struct to calculate Scaled output type for IFFT==================
 * ========================================================================
 */
template <int t_L, transform_direction_enum transform_direction, typename T_finalStageOutputType>
struct FFTScaledOutput {};

//======================== Specialize for float complex type===============
template <int t_L, transform_direction_enum transform_direction>
struct FFTScaledOutput<t_L, transform_direction, std::complex<float> > {
    typedef std::complex<float> T_scaledFFTOutputType;
};

//======================== Specialize for double complex type===============
template <int t_L, transform_direction_enum transform_direction>
struct FFTScaledOutput<t_L, transform_direction, std::complex<double> > {
    typedef std::complex<double> T_scaledFFTOutputType;
};
//======================== Specialize for float complex type===============
template <int t_L, transform_direction_enum transform_direction>
struct FFTScaledOutput<t_L, transform_direction, std::complex<int> > {
    typedef std::complex<int> T_scaledFFTOutputType;
};

//==================== Specialize for ap_fixed complex type================
// Case :when bit width and integer part bits are known
template <int t_L, transform_direction_enum transform_direction, int t_inputSizeBits, int t_integerPartBits>
struct FFTScaledOutput<t_L, transform_direction, std::complex<ap_fixed<t_inputSizeBits, t_integerPartBits> > > {
    static const int BITS_TO_SCALE = (transform_direction == REVERSE_TRANSFORM) ? (ssrFFTLog2<t_L>::val) : 0;
    typedef std::complex<ap_fixed<t_inputSizeBits, t_integerPartBits - BITS_TO_SCALE, AP_TRN, AP_WRAP, 0> >
        T_scaledFFTOutputType;
};

//==================== Specialize for ap_fixed complex type================
// Case :when bit width and integer part bits are know
template <int t_L,
          transform_direction_enum transform_direction,
          int t_inputSizeBits,
          int t_integerPartBits,
          ap_q_mode t_inputQuantizationMode,
          ap_o_mode t_inputOverflowMode,
          int t_inNumOfSatBits>
struct FFTScaledOutput<t_L,
                       transform_direction,
                       std::complex<ap_fixed<t_inputSizeBits,
                                             t_integerPartBits,
                                             t_inputQuantizationMode,
                                             t_inputOverflowMode,
                                             t_inNumOfSatBits> > > {
    static const int BITS_TO_SCALE = (transform_direction == REVERSE_TRANSFORM) ? (ssrFFTLog2<t_L>::val) : 0;
    typedef std::complex<ap_fixed<t_inputSizeBits,
                                  t_integerPartBits - BITS_TO_SCALE,
                                  t_inputQuantizationMode,
                                  t_inputOverflowMode,
                                  t_inNumOfSatBits> >
        T_scaledFFTOutputType;
};

template <typename fft_params, typename T_fft_input_type>
struct ssr_fft_output_type {
    //=======================================================================
    // Casted Type is essentially the same type as input expect that quantization
    // mode and round mode gets added with default values.
    typedef typename FFTInputTraits<T_fft_input_type>::T_castedType casted_type;
    //=======================================================================
    typedef typename FFTOutputTraits<fft_params::N,
                                     fft_params::R,
                                     fft_params::scaling_mode,
                                     fft_params::transform_direction,
                                     fft_params::butterfly_rnd_mode,
                                     casted_type>::T_FFTOutType t_ssr_fft_out;
};

} // end namespace fft
} // end namespace dsp
} // end namespace xf

//======================================================================================================================
#endif // HLS_SSR_FFT_OUTPUT_TRAITS_H_
