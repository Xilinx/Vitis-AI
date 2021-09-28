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

// File Name : hls_ssr_fft_butterfly_traits.hpp
#ifndef HLS_SSR_FFT_BUTTERFLY_TRAITS_H_
#define HLS_SSR_FFT_BUTTERFLY_TRAITS_H_

#include <ap_fixed.h>
//#include <complex>
#include "vitis_fft/hls_ssr_fft_utilities.hpp"
#include "vitis_fft/hls_ssr_fft_enums.hpp"
#include "vitis_fft/hls_ssr_fft_dsp48.hpp"

#include "vitis_fft/fft_complex.hpp"
//#include "DEBUG_CONSTANTS.hpp"

/*
 =========================================================================================
 -_-                                                                                   -_-
 -_-                                                                                   -_-
 -_-                                                                                   -_-
 -_-                                                                                   -_-
 -_-                                                                                   -_-
 -_-                                                                                   -_-
 -_-   Given the input type for accumulator the butterfly traits defines the output    -_-
 -_-   type to be used for storing the output of accumulator based on the input type   -_-
 -_-   and the scaling mode selected for accumulation. Three different scaling modes   -_-
 -_-   are defined:                                                                    -_-
 -_-               SSR_FFT_NO_SCALING : No scaling is done to prevent any overflows    -_-
 -_- 								   bit growth is allowed every addition -_-
 -_-               SSR_FFT_SCALE      : Scale the output by one bit ever binary        -_-
 -_-                                    addition                                       -_-
 -_-        SSR_FFT_GROW_TO_MAX_WIDTH : Initially the bit growth allowed until max     -_-
 -_-                                    width reached and then scaling used after that -_-
 -_-   Note : m_butterflyType is defined for debug purposes only                             -_-
 -_-                                                                                   -_-
 -_-                                                                                   -_-
 -_-                                                                                   -_-
 -_-                                                                                   -_-
 -_-                                                                                   -_-
 -_-                                                                                   -_-
 ========================================================================================
 */

// Traits for Butterfly input and output

#define HLS_SSR_FFT_ENABLE_ADDER_TREE_SCALING

namespace xf {
namespace dsp {
namespace fft {

#if 1
template <bool isFirtStage, typename T_in>
struct ButterflyTraitsDefault {
    // typedef T_in T_butterflyAccumType;
    // typedef T_in T_butterflyComplexRotatedType;
};

template <bool isFirtStage, scaling_mode_enum t_scalingMode, typename T_in>
struct ButterflyTraits : public ButterflyTraitsDefault<isFirtStage, T_in> {
    static const int m_butterflyType = 0;
};
//=====================================================================================================
// Integer Type Butterfly Traits
//=====================================================================================================

template <bool isFirtStage, scaling_mode_enum t_scalingMode>
struct ButterflyTraits<isFirtStage, t_scalingMode, std::complex<int> >
    : public ButterflyTraitsDefault<isFirtStage, std::complex<int> > {
    typedef std::complex<int> T_butterflyAccumType;
    typedef std::complex<int> T_butterflyComplexRotatedType;
    static const int m_butterflyType = 10;
};
//=====================================================================================================

//=====================================================================================================
// double Type Butterfly Traits
//=====================================================================================================
template <bool isFirtStage, scaling_mode_enum t_scalingMode>
struct ButterflyTraits<isFirtStage, t_scalingMode, std::complex<double> >
    : public ButterflyTraitsDefault<isFirtStage, std::complex<double> > {
    typedef std::complex<double> T_butterflyAccumType;
    typedef std::complex<double> T_butterflyComplexRotatedType;
    static const int m_butterflyType = 20;
};
//=====================================================================================================

//=====================================================================================================
// float Type Butterfly Traits
//=====================================================================================================
template <bool isFirtStage, scaling_mode_enum t_scalingMode>
struct ButterflyTraits<isFirtStage, t_scalingMode, std::complex<float> >
    : public ButterflyTraitsDefault<isFirtStage, std::complex<float> > {
    typedef std::complex<float> T_butterflyAccumType;
    typedef std::complex<float> T_butterflyComplexRotatedType;
    static const int m_butterflyType = 30;
};
//=====================================================================================================

//=====================================================================================================
// ap_fixed Type Butterfly Traits : Generic scaling mode
//=====================================================================================================
template <bool isFirtStage, scaling_mode_enum t_scalingMode, int t_inputSizeBits, int t_integerPartBits>
struct ButterflyTraits<isFirtStage, t_scalingMode, std::complex<ap_fixed<t_inputSizeBits, t_integerPartBits> > > {
    typedef std::complex<ap_fixed<t_inputSizeBits + 1, t_integerPartBits + 1, AP_TRN, AP_WRAP, 0> >
        T_butterflyAccumType;
    static const int COMPLEX_ROTATED_BIT_GROWTH = isFirtStage ? 1 : 0;
    typedef std::complex<
        ap_fixed<t_inputSizeBits + 1, t_integerPartBits + COMPLEX_ROTATED_BIT_GROWTH, AP_TRN, AP_WRAP, 0> >
        T_butterflyComplexRotatedType;
    static const int m_butterflyType = 1;
};
//=====================================================================================================

//=====================================================================================================
// ap_fixed Type Butterfly Traits : SSR_FFT_NO_SCALING
//=====================================================================================================
template <bool isFirtStage, int t_inputSizeBits, int t_integerPartBits>
struct ButterflyTraits<isFirtStage, SSR_FFT_NO_SCALING, std::complex<ap_fixed<t_inputSizeBits, t_integerPartBits> > > {
    typedef std::complex<ap_fixed<t_inputSizeBits + 1, t_integerPartBits + 1, AP_TRN, AP_WRAP, 0> >
        T_butterflyAccumType;
    static const int COMPLEX_ROTATED_BIT_GROWTH = isFirtStage ? 1 : 0;
    typedef std::complex<
        ap_fixed<t_inputSizeBits + 1, t_integerPartBits + COMPLEX_ROTATED_BIT_GROWTH, AP_TRN, AP_WRAP, 0> >
        T_butterflyComplexRotatedType;
    static const int m_butterflyType = 2;
};
//=====================================================================================================

//=====================================================================================================
// ap_fixed Type Butterfly Traits : SSR_FFT_SCALE
//=====================================================================================================
template <bool isFirtStage, int t_inputSizeBits, int t_integerPartBits>
struct ButterflyTraits<isFirtStage, SSR_FFT_SCALE, std::complex<ap_fixed<t_inputSizeBits, t_integerPartBits> > > {
#ifdef HLS_SSR_FFT_ENABLE_ADDER_TREE_SCALING
    static const int BIT_GROWTH_WL = 0;
    static const int BIT_GROWTH_IL = 1;
    static const int OUTPUT_WL = ((t_inputSizeBits + BIT_GROWTH_WL) <= DSP48_OP2_BIT_WIDTH)
                                     ? (t_inputSizeBits + BIT_GROWTH_WL)
                                     : DSP48_OP2_BIT_WIDTH;
    static const int OUTPUT_IL =
        ((t_integerPartBits + BIT_GROWTH_IL) <= OUTPUT_WL) ? (t_integerPartBits + BIT_GROWTH_IL) : OUTPUT_WL;

#else
    static const int BIT_GROWTH_WL = 1;
    static const int BIT_GROWTH_IL = 1;
    static const int OUTPUT_WL = (t_inputSizeBits + BIT_GROWTH_WL);
    static const int OUTPUT_IL = (t_integerPartBits + BIT_GROWTH_IL);
#endif
    typedef std::complex<ap_fixed<OUTPUT_WL, OUTPUT_IL, AP_TRN, AP_WRAP, 0> > T_butterflyAccumType;
    static const int COMPLEX_ROTATED_BIT_GROWTH = isFirtStage ? 1 : 0;
    typedef std::complex<ap_fixed<t_inputSizeBits, t_integerPartBits + COMPLEX_ROTATED_BIT_GROWTH, AP_TRN, AP_WRAP, 0> >
        T_butterflyComplexRotatedType;
    static const int m_butterflyType = 3;
};
//=====================================================================================================

//=====================================================================================================
// ap_fixed Type Butterfly Traits : SSR_FFT_GROW_TO_MAX_WIDTH
//=====================================================================================================
template <bool isFirtStage, int t_inputSizeBits, int t_integerPartBits>
struct ButterflyTraits<isFirtStage,
                       SSR_FFT_GROW_TO_MAX_WIDTH,
                       std::complex<ap_fixed<t_inputSizeBits, t_integerPartBits> > > {
// typedef ap_fixed<t_inputSizeBits,t_integerPartBits,AP_TRN,AP_WRAP,0> TTR_bflyProductType;
#ifdef HLS_SSR_FFT_ENABLE_ADDER_TREE_SCALING
    static const int BIT_GROWTH_WL = 1;
    static const int BIT_GROWTH_IL = 1;
    static const int OUTPUT_WL = ((t_inputSizeBits + BIT_GROWTH_WL) <= DSP48_OP2_BIT_WIDTH)
                                     ? (t_inputSizeBits + BIT_GROWTH_WL)
                                     : DSP48_OP2_BIT_WIDTH;
    static const int OUTPUT_IL = ((t_integerPartBits + BIT_GROWTH_IL) <= DSP48_OP2_BIT_WIDTH)
                                     ? (t_integerPartBits + BIT_GROWTH_IL)
                                     : DSP48_OP2_BIT_WIDTH;
#else
    static const int BIT_GROWTH_WL = 1;
    static const int BIT_GROWTH_IL = 1;
    static const int OUTPUT_WL = (t_inputSizeBits + BIT_GROWTH_WL);
    static const int OUTPUT_IL = (t_integerPartBits + BIT_GROWTH_IL);
#endif
    typedef std::complex<ap_fixed<OUTPUT_WL, OUTPUT_IL, AP_TRN, AP_WRAP, 0> > T_butterflyAccumType;

    static const int COMPLEX_ROTATED_BIT_GROWTH = isFirtStage ? 1 : 0;
    static const int GRWON_ROT_WL = COMPLEX_ROTATED_BIT_GROWTH + t_inputSizeBits;
    static const int ROL_WL = (GRWON_ROT_WL <= DSP48_OP2_BIT_WIDTH) ? GRWON_ROT_WL : t_inputSizeBits;

    static const int GRWON_ROT_IL = COMPLEX_ROTATED_BIT_GROWTH + t_integerPartBits;
    static const int ROL_IL = (GRWON_ROT_IL <= DSP48_OP2_BIT_WIDTH) ? GRWON_ROT_IL : t_integerPartBits;

    typedef std::complex<ap_fixed<ROL_WL, ROL_IL, AP_TRN, AP_WRAP, 0> > T_butterflyComplexRotatedType;
    static const int m_butterflyType = 4;
};
//=====================================================================================================

//=====================================================================================================
// ap_fixed Type Butterfly Traits : Generic with quantization and rounding mode input
//=====================================================================================================

template <bool isFirtStage,
          scaling_mode_enum t_scalingMode,
          int t_inputSizeBits,
          int t_integerPartBits,
          ap_q_mode Q,
          ap_o_mode O,
          int sat_bits>
struct ButterflyTraits<isFirtStage,
                       t_scalingMode,
                       std::complex<ap_fixed<t_inputSizeBits, t_integerPartBits, Q, O, sat_bits> > > {
    typedef std::complex<ap_fixed<t_inputSizeBits + 1, t_integerPartBits + 1, Q, O, sat_bits> > T_butterflyAccumType;
    static const int m_butterflyType = 5;
};
//=====================================================================================================

//=====================================================================================================
// ap_fixed Type Butterfly Traits : SSR_FFT_NO_SCALING with quantization and rounding mode input
//=====================================================================================================
template <bool isFirtStage, int t_inputSizeBits, int t_integerPartBits, ap_q_mode Q, ap_o_mode O, int sat_bits>
struct ButterflyTraits<isFirtStage,
                       SSR_FFT_NO_SCALING,
                       std::complex<ap_fixed<t_inputSizeBits, t_integerPartBits, Q, O, sat_bits> > > {
    static const int COMPLEX_ROTATED_BIT_GROWTH = isFirtStage ? 1 : 0;
    typedef std::complex<ap_fixed<t_inputSizeBits + 1, t_integerPartBits + 1, AP_TRN, AP_WRAP, 0> >
        T_butterflyAccumType;
    typedef std::complex<ap_fixed<t_inputSizeBits + COMPLEX_ROTATED_BIT_GROWTH,
                                  t_integerPartBits + COMPLEX_ROTATED_BIT_GROWTH,
                                  AP_TRN,
                                  AP_WRAP,
                                  0> >
        T_butterflyComplexRotatedType;

    static const int m_butterflyType = 6;
};
//=====================================================================================================

//=====================================================================================================
// ap_fixed Type Butterfly Traits : SSR_FFT_SCALE with quantization and rounding mode input
//=====================================================================================================

template <bool isFirtStage, int t_inputSizeBits, int t_integerPartBits, ap_q_mode Q, ap_o_mode O, int sat_bits>
struct ButterflyTraits<isFirtStage,
                       SSR_FFT_SCALE,
                       std::complex<ap_fixed<t_inputSizeBits, t_integerPartBits, Q, O, sat_bits> > > {
#ifdef HLS_SSR_FFT_ENABLE_ADDER_TREE_SCALING
    static const int BIT_GROWTH_WL = 0;
    static const int BIT_GROWTH_IL = 1;
    static const int OUTPUT_WL = ((t_inputSizeBits + BIT_GROWTH_WL) <= DSP48_OP2_BIT_WIDTH)
                                     ? (t_inputSizeBits + BIT_GROWTH_WL)
                                     : DSP48_OP2_BIT_WIDTH;
    static const int OUTPUT_IL =
        ((t_integerPartBits + BIT_GROWTH_IL) <= OUTPUT_WL) ? (t_integerPartBits + BIT_GROWTH_IL) : OUTPUT_WL;
#else
    static const int BIT_GROWTH_WL = 1;
    static const int BIT_GROWTH_IL = 1;
    static const int OUTPUT_WL = (t_inputSizeBits + BIT_GROWTH_WL);
    static const int OUTPUT_IL = (t_integerPartBits + BIT_GROWTH_IL);
#endif
    typedef std::complex<ap_fixed<OUTPUT_WL, OUTPUT_IL, Q, O, sat_bits> > T_butterflyAccumType;
    static const int COMPLEX_ROTATED_BIT_GROWTH = isFirtStage ? 1 : 0;
    typedef std::complex<ap_fixed<t_inputSizeBits, t_integerPartBits + COMPLEX_ROTATED_BIT_GROWTH, AP_TRN, AP_WRAP, 0> >
        T_butterflyComplexRotatedType;
    static const int m_butterflyType = 7;
};
//=====================================================================================================

//=====================================================================================================
// ap_fixed Type Butterfly Traits : SSR_FFT_GROW_TO_MAX_WIDTH with quantization and rounding mode input
//=====================================================================================================
template <bool isFirtStage, int t_inputSizeBits, int t_integerPartBits, ap_q_mode Q, ap_o_mode O, int sat_bits>
struct ButterflyTraits<isFirtStage,
                       SSR_FFT_GROW_TO_MAX_WIDTH,
                       std::complex<ap_fixed<t_inputSizeBits, t_integerPartBits, Q, O, sat_bits> > > {
#ifdef HLS_SSR_FFT_ENABLE_ADDER_TREE_SCALING
    static const int BIT_GROWTH_WL = 1;
    static const int BIT_GROWTH_IL = 1;
    static const int OUTPUT_WL = ((t_inputSizeBits + BIT_GROWTH_WL) <= DSP48_OP2_BIT_WIDTH)
                                     ? (t_inputSizeBits + BIT_GROWTH_WL)
                                     : DSP48_OP2_BIT_WIDTH;
    static const int OUTPUT_IL = ((t_integerPartBits + BIT_GROWTH_IL) <= DSP48_OP2_BIT_WIDTH)
                                     ? (t_inputSizeBits + BIT_GROWTH_IL)
                                     : DSP48_OP2_BIT_WIDTH;
#else
    static const int OUTPUT_WL = (t_inputSizeBits + 1);
    static const int OUTPUT_IL = (t_integerPartBits + 1);
#endif
    typedef std::complex<ap_fixed<OUTPUT_WL, OUTPUT_IL, Q, O, sat_bits> > T_butterflyAccumType;
    static const int COMPLEX_ROTATED_BIT_GROWTH = isFirtStage ? 1 : 0;
    static const int GRWON_ROT_WL = COMPLEX_ROTATED_BIT_GROWTH + t_inputSizeBits;
    static const int ROL_WL = (GRWON_ROT_WL <= DSP48_OP2_BIT_WIDTH) ? GRWON_ROT_WL : t_inputSizeBits;

    static const int GRWON_ROT_IL = COMPLEX_ROTATED_BIT_GROWTH + t_integerPartBits;
    static const int ROL_IL = (GRWON_ROT_IL <= DSP48_OP2_BIT_WIDTH) ? GRWON_ROT_IL : t_integerPartBits;

    typedef std::complex<ap_fixed<ROL_WL, ROL_IL, AP_TRN, AP_WRAP, 0> > T_butterflyComplexRotatedType;
    static const int m_butterflyType = 8;
};
//=====================================================================================================

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#endif
} // end namespace fft
} // end namespace dsp
} // end namespace xf

#endif // HLS_SSR_FFT_BUTTERFLY_TRAITS_H_
