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

// File Name  : hls_ssr_fft_complex_multiplier.hpp
#ifndef _HLS_SSR_FFT_COMPLEX_MULTIPLIER_H_
#define _HLS_SSR_FFT_COMPLEX_MULTIPLIER_H_
#include <complex>

#include "vitis_fft/fft_complex.hpp"

namespace xf {
namespace dsp {
namespace fft {
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
template <typename T_op1, typename T_op2, typename T_prd>
void complexMultiply(std::complex<T_op1> p_complexOp1,
                     std::complex<T_op2> p_complexOp2,
                     std::complex<T_prd>& p_product) {
#pragma HLS INLINE // recursive
    // p_product.real(p_complexOp1.real() * p_complexOp2.real() - p_complexOp1.imag() * p_complexOp2.imag());
    // p_product.imag(p_complexOp1.real() * p_complexOp2.imag() + p_complexOp1.imag() * p_complexOp2.real());
    T_op1 real1 = p_complexOp1.real() * p_complexOp2.real();
    T_op1 real2 = p_complexOp1.imag() * p_complexOp2.imag();
    T_op1 real_out = real1 - real2;
    p_product.real(real_out);
    T_op1 imag1 = p_complexOp1.real() * p_complexOp2.imag();
    T_op1 imag2 = p_complexOp1.imag() * p_complexOp2.real();
    T_op1 imag_out = imag1 + imag2;
    p_product.imag(imag_out);
}

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
} // end namespace fft
} // end namespace dsp
} // namespace xf
#endif //_HLS_SSR_FFT_COMPLEX_MULTIPLIER_H_
