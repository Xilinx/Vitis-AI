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
#ifndef _HLS_SSR_FFT_UTILITY_TRAITS_H_
#define _HLS_SSR_FFT_UTILITY_TRAITS_H_
#include <complex>
#include <ap_fixed.h>

#include "vitis_fft/fft_complex.hpp"

namespace xf {
namespace dsp {
namespace fft {

///// Utility Traits to retrieve inner type in std::complex type.
template <typename T_complex>
struct UtilityInnerTypeTraits {};

// Specialization for type : float
template <>
struct UtilityInnerTypeTraits<std::complex<float> > {
    typedef float T_inner;
    typedef std::complex<T_inner> T_in;
};

// Specialization for type : double
template <>
struct UtilityInnerTypeTraits<std::complex<double> > {
    typedef double T_inner;
    typedef std::complex<T_inner> T_in;
};

// Specialization for type : int
template <>
struct UtilityInnerTypeTraits<std::complex<int> > {
    typedef int T_inner;
    typedef std::complex<T_inner> T_in;
};

// Specialization for type : ap_fixed<int,int>
template <int t_WL, int t_IL>
struct UtilityInnerTypeTraits<std::complex<ap_fixed<t_WL, t_IL> > > {
    typedef ap_fixed<t_WL, t_IL> T_inner;
    typedef std::complex<T_inner> T_in;
};
// Specialization for type : ap_fixed<int,int>
template <int t_WL, int t_IL, ap_q_mode t_qMode, ap_o_mode t_ovfMode, int t_numBits>
struct UtilityInnerTypeTraits<std::complex<ap_fixed<t_WL, t_IL, t_qMode, t_ovfMode, t_numBits> > > {
    typedef ap_fixed<t_WL, t_IL, t_qMode, t_ovfMode, t_numBits> T_inner;
    typedef std::complex<T_inner> T_in;
};

} // end namespace fft
} // end namespace dsp
} // end namespace xf
#endif //_HLS_SSR_FFT_UTILITY_TRAITS_H_
