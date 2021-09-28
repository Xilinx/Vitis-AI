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

// File Name : hls_ssr_fft_exp_table_traits.hpp
#ifndef HLS_SSR_FFT_EXP_TABLE_TRAITS_H_
#define HLS_SSR_FFT_EXP_TABLE_TRAITS_H_

#include <ap_fixed.h>
//#include <complex>

#include "vitis_fft/fft_complex.hpp"

/*
 =========================================================================================
 -_-                                                                                   -_-
 -_-                                                                                   -_-
 -_-                                                                                   -_-
 -_-                                                                                   -_-
 -_-                                                                                   -_-
 -_-                                                                                   -_-
 -_-

 ComplexExpTableTraits: struct defined traits for exp table. Given the
 input type this struct can be used to define exp table type.

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

namespace xf {
namespace dsp {
namespace fft {

template <typename T>
struct ComplexExpTableTraits {};

template <typename T>
struct ComplexExpTableTraits<std::complex<T> > {
    typedef std::complex<T> t_complexExpTableType;
};

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
} // end namespace fft
} // end namespace dsp
} // end namespace xf

#endif // HLS_SSR_FFT_EXP_TABLE_TRAITS_H_
