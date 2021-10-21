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

// File Name : hls_ssr_fft_input_traits.hpp
#ifndef HLS_SSR_FFT_INPUT_TRAITS_H_
#define HLS_SSR_FFT_INPUT_TRAITS_H_
#include <complex>
#include <ap_fixed.h>

#include "vitis_fft/hls_ssr_fft_dsp48.hpp"
#include "vitis_fft/fft_complex.hpp"
namespace xf {
namespace dsp {
namespace fft {

template <typename T_in>
struct FFTInputTraits {};
/*
 * ==========================================================================
 * ssr fft input traits for type : complex <float>
 * ==========================================================================
 */
template <>
struct FFTInputTraits<std::complex<float> > {
    typedef std::complex<float> T_castedType;
};

/*
 * ==========================================================================
 * ssr fft input traits for type : complex <double>
 * ==========================================================================
 */
template <>
struct FFTInputTraits<std::complex<double> > {
    typedef std::complex<double> T_castedType;
};
/*
 * ==========================================================================
 * ssr fft input traits for type : complex <ap_fixed>
 * ==========================================================================
 */
template <int tp_WL, int tp_IL>
struct FFTInputTraits<std::complex<ap_fixed<tp_WL, tp_IL> > > {
    typedef std::complex<ap_fixed<tp_WL, tp_IL, AP_TRN, AP_WRAP, 0> > T_castedType;
};

template <int tp_WL, int tp_IL, ap_q_mode t_in_q_mode, ap_o_mode t_inOverflowMode, int t_inNumOfSatBits>
struct FFTInputTraits<std::complex<ap_fixed<tp_WL, tp_IL, t_in_q_mode, t_inOverflowMode, t_inNumOfSatBits> > > {
    typedef std::complex<ap_fixed<tp_WL, tp_IL, t_in_q_mode, t_inOverflowMode, t_inNumOfSatBits> > T_castedType;
};

/*
 * ==========================================================================
 * ssr fft input traits for type : complex <ap_ufixed>
 * ==========================================================================
 */
template <int tp_WL, int tp_IL>
struct FFTInputTraits<std::complex<ap_ufixed<tp_WL, tp_IL> > > {
    static const int EXPECTED_WL = tp_WL + 1;
    static const int EXPECTED_IL = tp_IL + 1;
    static const int WL = (EXPECTED_WL < DSP48_OP2_BIT_WIDTH) ? EXPECTED_WL : DSP48_OP2_BIT_WIDTH;
    static const int IL = (EXPECTED_IL < DSP48_OP2_BIT_WIDTH) ? EXPECTED_IL : DSP48_OP2_BIT_WIDTH;

    typedef std::complex<ap_fixed<WL, IL, AP_TRN, AP_WRAP, 0> > T_castedType;
};

template <int tp_WL, int tp_IL, ap_q_mode t_in_q_mode, ap_o_mode t_inOverflowMode, int t_inNumOfSatBits>
struct FFTInputTraits<std::complex<ap_ufixed<tp_WL, tp_IL, t_in_q_mode, t_inOverflowMode, t_inNumOfSatBits> > > {
    static const int EXPECTED_WL = tp_WL + 1;
    static const int EXPECTED_IL = tp_IL + 1;
    static const int WL = (EXPECTED_WL < DSP48_OP2_BIT_WIDTH) ? EXPECTED_WL : DSP48_OP2_BIT_WIDTH;
    static const int IL = (EXPECTED_IL < DSP48_OP2_BIT_WIDTH) ? EXPECTED_IL : DSP48_OP2_BIT_WIDTH;

    typedef std::complex<ap_fixed<WL, IL, t_in_q_mode, t_inOverflowMode, t_inNumOfSatBits> > T_castedType;
};

} // end namespace fft
} // end namespace dsp
} // end namespace xf

#endif // HLS_SSR_FFT_INPUT_TRAITS_H_
