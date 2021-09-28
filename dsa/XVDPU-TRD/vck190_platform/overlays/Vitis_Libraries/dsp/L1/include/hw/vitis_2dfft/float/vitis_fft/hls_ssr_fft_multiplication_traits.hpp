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

// File Name : hls_ssr_fft_multiplication_traits.hpp
#ifndef _HLS_SSR_FFT_MULTIPLICATION_TRAITS_H_
#define _HLS_SSR_FFT_MULTIPLICATION_TRAITS_H_

namespace xf {
namespace dsp {
namespace fft {

template <typename T_op1TwiddleType, typename T_op2SampleType>
struct FFTMultiplicationTraits {};

/*
 * ==========================================================================
 * ssr fft input traits for type : complex <float>
 * ==========================================================================
 */
template <>
struct FFTMultiplicationTraits<complex_wrapper<float>, complex_wrapper<float> > {
    typedef complex_wrapper<float> T_productOpType;
    typedef float T_op1RealType;
    typedef float T_op1ImagType;
    typedef float T_op2RealType;
    typedef float T_op2ImagType;
    typedef float T_accumOutType;
    typedef float T_productOutType;
};
/*
 * ==========================================================================
 * ssr fft input traits for type : complex <double>
 * ==========================================================================
 */
template <>
struct FFTMultiplicationTraits<complex_wrapper<double>, complex_wrapper<double> > {
    typedef complex_wrapper<double> T_productOpType;
    typedef double T_op1RealType;
    typedef double T_op1ImagType;
    typedef double T_op2RealType;
    typedef double T_op2ImagType;
    typedef double T_accumOutType;
    typedef double T_productOutType;
};

/* ==========================================================================
 * ssr fft input traits for type : complex <ap_fixed>
 * ==========================================================================
 */
template <int t_op1_WL, int t_op1_IL, int t_op2_WL, int t_op2_IL>
struct FFTMultiplicationTraits<complex_wrapper<ap_fixed<t_op1_WL, t_op1_IL> >,
                               complex_wrapper<ap_fixed<t_op2_WL, t_op2_IL> > > {
    static const int product_IL = (t_op1_IL > t_op2_IL) ? t_op1_IL : t_op2_IL;

    static const int op1_fraction_length = t_op1_WL - t_op1_IL;
    static const int op2_fraction_length = t_op2_WL - t_op2_IL;
    static const int product_fraction_lenth =
        (op1_fraction_length > op2_fraction_length) ? op1_fraction_length : op2_fraction_length;
    static const int product_WL = product_IL + product_fraction_lenth;
    typedef complex_wrapper<ap_fixed<product_WL, product_IL, AP_TRN, AP_WRAP, 0> > T_productOpType;

    /// Get inner types of the given comples types ::::
    typedef ap_fixed<t_op1_WL, t_op1_IL> T_op1RealType;
    typedef ap_fixed<t_op1_WL, t_op1_IL> T_op1ImagType;
    typedef ap_fixed<t_op2_WL, t_op2_IL> T_op2RealType;
    typedef ap_fixed<t_op2_WL, t_op2_IL> T_op2ImagType;

    // typedef complex_wrapper<double> T_accumOutType;
    // typedef complex_wrapper<double> T_productOutType;
};

/* ==========================================================================
 * ssr fft input traits for type : complex <ap_fixed>
 * ==========================================================================
 */
template <int t_op1_WL,
          int t_op1_IL,
          ap_q_mode t_op1QuantizationMode,
          ap_o_mode t_op1OverflowMode,
          int t_op1NumSatBits, // type-1
          int t_op2_WL,
          int t_op2_IL // type2
          >
struct FFTMultiplicationTraits<
    complex_wrapper<ap_fixed<t_op1_WL, t_op1_IL, t_op1QuantizationMode, t_op1OverflowMode, t_op1NumSatBits> >,
    complex_wrapper<ap_fixed<t_op2_WL, t_op2_IL> > > {
    static const int product_IL = (t_op1_IL > t_op2_IL) ? t_op1_IL : t_op2_IL;

    static const int op1_fraction_length = t_op1_WL - t_op1_IL;
    static const int op2_fraction_length = t_op2_WL - t_op2_IL;
    static const int product_fraction_lenth =
        (op1_fraction_length > op2_fraction_length) ? op1_fraction_length : op2_fraction_length;
    static const int product_WL = product_IL + product_fraction_lenth;
    typedef complex_wrapper<
        ap_fixed<product_WL, product_IL, t_op1QuantizationMode, t_op1OverflowMode, t_op1NumSatBits> >
        T_productOpType;
    /// Get inner types of the given comples types ::::
    typedef ap_fixed<t_op1_WL, t_op1_IL, t_op1QuantizationMode, t_op1OverflowMode, t_op1NumSatBits> T_op1RealType;
    typedef ap_fixed<t_op1_WL, t_op1_IL, t_op1QuantizationMode, t_op1OverflowMode, t_op1NumSatBits> T_op1ImagType;
    typedef ap_fixed<t_op2_WL, t_op2_IL> T_op2RealType;
    typedef ap_fixed<t_op2_WL, t_op2_IL> T_op2ImagType;
};

/* ==========================================================================
 * ssr fft input traits for type : complex <ap_fixed>
 * ==========================================================================
 */
template <int t_op1_WL,
          int t_op1_IL, // type-1
          int t_op2_WL,
          int t_op2_IL,
          ap_q_mode t_op2QuantizationMode,
          ap_o_mode t_op2OverflowMode,
          int t_op2NumSatBits // type2
          >
struct FFTMultiplicationTraits<
    complex_wrapper<ap_fixed<t_op1_WL, t_op1_IL> >,
    complex_wrapper<ap_fixed<t_op2_WL, t_op2_IL, t_op2QuantizationMode, t_op2OverflowMode, t_op2NumSatBits> > > {
    static const int product_IL = (t_op1_IL > t_op2_IL) ? t_op1_IL : t_op2_IL;

    static const int op1_fraction_length = t_op1_WL - t_op1_IL;
    static const int op2_fraction_length = t_op2_WL - t_op2_IL;
    static const int product_fraction_lenth =
        (op1_fraction_length > op2_fraction_length) ? op1_fraction_length : op2_fraction_length;
    static const int product_WL = product_IL + product_fraction_lenth;
    typedef complex_wrapper<
        ap_fixed<product_WL, product_IL, t_op2QuantizationMode, t_op2OverflowMode, t_op2NumSatBits> >
        T_productOpType;
    /// Get inner types of the given comples types ::::
    typedef ap_fixed<t_op1_WL, t_op1_IL> T_op1RealType;
    typedef ap_fixed<t_op1_WL, t_op1_IL> T_op1ImagType;
    typedef ap_fixed<t_op2_WL, t_op2_IL, t_op2QuantizationMode, t_op2OverflowMode, t_op2NumSatBits> T_op2RealType;
    typedef ap_fixed<t_op2_WL, t_op2_IL, t_op2QuantizationMode, t_op2OverflowMode, t_op2NumSatBits> T_op2ImagType;
};

/* ==========================================================================
 * ssr fft input traits for type : complex <ap_fixed>
 * ==========================================================================
 */
template <int t_op1_WL,
          int t_op1_IL,
          ap_q_mode t_op1QuantizationMode,
          ap_o_mode t_op1OverflowMode,
          int t_op1NumSatBits, // type-1
          int t_op2_WL,
          int t_op2_IL,
          ap_q_mode t_op2QuantizationMode,
          ap_o_mode t_op2OverflowMode,
          int t_op2NumSatBits // type2
          >
struct FFTMultiplicationTraits<
    complex_wrapper<ap_fixed<t_op1_WL, t_op1_IL, t_op1QuantizationMode, t_op1OverflowMode, t_op1NumSatBits> >,
    complex_wrapper<ap_fixed<t_op2_WL, t_op2_IL, t_op2QuantizationMode, t_op2OverflowMode, t_op2NumSatBits> > > {
    static const int product_IL = (t_op1_IL > t_op2_IL) ? t_op1_IL : t_op2_IL;

    static const int op1_fraction_length = t_op1_WL - t_op1_IL;
    static const int op2_fraction_length = t_op2_WL - t_op2_IL;
    static const int product_fraction_lenth =
        (op1_fraction_length > op2_fraction_length) ? op1_fraction_length : op2_fraction_length;
    static const int product_WL = product_IL + product_fraction_lenth;
    typedef complex_wrapper<
        ap_fixed<product_WL, product_IL, t_op2QuantizationMode, t_op2OverflowMode, t_op2NumSatBits> >
        T_productOpType;
    /// Get inner types of the given comples types ::::
    typedef ap_fixed<t_op1_WL, t_op1_IL, t_op1QuantizationMode, t_op1OverflowMode, t_op1NumSatBits> T_op1RealType;
    typedef ap_fixed<t_op1_WL, t_op1_IL, t_op1QuantizationMode, t_op1OverflowMode, t_op1NumSatBits> T_op1ImagType;
    typedef ap_fixed<t_op2_WL, t_op2_IL, t_op2QuantizationMode, t_op2OverflowMode, t_op2NumSatBits> T_op2RealType;
    typedef ap_fixed<t_op2_WL, t_op2_IL, t_op2QuantizationMode, t_op2OverflowMode, t_op2NumSatBits> T_op2ImagType;
};

} // end namespace fft
} // end namespace dsp
} // end namespace xf

#endif //_HLS_SSR_FFT_MULTIPLICATION_TRAITS_H_
