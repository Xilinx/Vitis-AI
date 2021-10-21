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

// File Name : hls_ssr_fft_twiddle_table.hpp
#ifndef HLS_SSR_FFT_TWIDDLE_TALBE_H_
#define HLS_SSR_FFT_TWIDDLE_TALBE_H_

//#define HLS_SSR_FFT_TWIDDLE_TALBE_PRINT_DEBUG_MESSAGES
//#define HLS_SSR_FFT_DEBUG

/*
 =========================================================================================
 -_- -_-
 -_- -_-
 -_- -_-
 -_- -_-
 -_- -_-
 -_- -_-
 -_-  initTwiddleTable is defined here. It initializes complex exponential tables    -_-
 -_-  that is used for calculating butterflies. three template parameters are passed
 t_R : The radix of the SSR FFT to be calculated
 t_L : the length of the SSR FFT to be calculated
 -_- -_-
 -_- -_-
 -_- -_-
 -_- -_-
 -_- -_-
 ========================================================================================
 */

#include <ap_fixed.h>
#include <math.h>

#include "vitis_fft/hls_ssr_fft_twiddle_table_traits.hpp"
#include "vitis_fft/hls_ssr_fft_utility_traits.hpp"

namespace xf {
namespace dsp {
namespace fft {

template <int t_L, int t_R, int t_instanceID, typename T_dtype>
struct TwiddleTable {};

template <int t_L, int t_R, int t_instanceID, typename T_dtype>
struct TwiddleTable<t_L, t_R, t_instanceID, std::complex<T_dtype> > {
    static void initTwiddleTable(std::complex<T_dtype> p_table[]) {
#pragma HLS INLINE off
        typedef typename TwiddleTypeCastingTraits<std::complex<T_dtype> >::T_roundingBasedCastType casting_type;

        static const int EXT_LEN = TwiddleTableLENTraits<t_L, t_R>::EXTENDED_TWIDDLE_TALBE_LENGTH;
        for (int i = 0; i < EXT_LEN; i++) {
            //#pragma HLS UNROLL
            double real = cos((2 * i * M_PI) / t_L);
            double imag = -sin((2 * i * M_PI) / t_L);
            p_table[i] = casting_type(real, imag);
        }
    }
};

template <int t_L, int t_R, int t_instanceID, typename T_dtype>
struct TwiddleTableWrapper {};

template <int t_L, int t_R, int t_instanceID, typename T_dtype>
struct TwiddleTableWrapper<t_L, t_R, t_instanceID, std::complex<T_dtype> > {
    std::complex<T_dtype> twiddleTable[TwiddleTableLENTraits<t_L, t_R>::EXTENDED_TWIDDLE_TALBE_LENGTH];
    TwiddleTableWrapper() {
        typedef typename TwiddleTypeCastingTraits<std::complex<T_dtype> >::T_roundingBasedCastType casting_type;
        static const int EXT_LEN = TwiddleTableLENTraits<t_L, t_R>::EXTENDED_TWIDDLE_TALBE_LENGTH;
        for (int i = 0; i < EXT_LEN; i++) {
            double real = cos((2 * i * M_PI) / t_L);
            double imag = -sin((2 * i * M_PI) / t_L);
            twiddleTable[i] = casting_type(real, imag);
        }
    }
};

template <int t_L, int t_R, unsigned int index_bw, typename T_dtype>
T_dtype readTwiddleTable(ap_uint<index_bw> p_index, T_dtype p_table[]) {
#pragma HLS INLINE
    typedef typename TwiddleTraits<T_dtype>::T_twiddleInnerType tableType;
    T_dtype temp = p_table[p_index];
    return temp; // p_table[p_index];
}

template <int t_L, int t_R, unsigned int t_phaseBitWidth, typename T_dtype_in>
T_dtype_in readQuaterTwiddleTable(ap_uint<t_phaseBitWidth> p_index, T_dtype_in p_table[]) {
#pragma HLS INLINE
    const ap_int<2> MAX_OUT = -1;
    ap_uint<t_phaseBitWidth> index_cos = p_index + (3 * t_L / 4);
    ap_uint<t_phaseBitWidth> index_sin = p_index;
    typedef typename TwiddleTypeCastingTraits<T_dtype_in>::T_roundingBasedCastType T_dtype;
    typedef typename TwiddleTraits<T_dtype>::T_twiddleInnerType tableType;
    tableType realCosVal;
    tableType imagSinVal;
    T_dtype complexOut;
    ap_uint<1> index_invert_control_imag = index_sin(t_phaseBitWidth - 2, t_phaseBitWidth - 2);
    ap_uint<1> output_negate_control_imag = index_sin(t_phaseBitWidth - 1, t_phaseBitWidth - 1);
    ap_uint<1> output_saturation_control_imag = (index_sin == (t_L / 4) || (index_sin == (3 * t_L / 4)));
    ap_uint<t_phaseBitWidth - 2> lut_index_imag = index_sin(t_phaseBitWidth - 3, 0);
    if (index_invert_control_imag == 1) lut_index_imag = ((~lut_index_imag) + 1);
    tableType lut_out_imag = p_table[lut_index_imag].imag();

    tableType temp_out_sin;
    if (output_saturation_control_imag)
        temp_out_sin = MAX_OUT;
    else
        temp_out_sin = lut_out_imag;

    if (output_negate_control_imag)
        imagSinVal = -temp_out_sin; // sab
    else
        imagSinVal = temp_out_sin; // sab

    ap_uint<1> index_invert_control_real = index_cos(t_phaseBitWidth - 2, t_phaseBitWidth - 2);
    ap_uint<1> output_negate_control_real = index_cos(t_phaseBitWidth - 1, t_phaseBitWidth - 1);
    ap_uint<1> output_saturation_control_real = (index_cos == (t_L / 4) || (index_cos == (3 * t_L / 4)));
    ap_uint<t_phaseBitWidth - 2> lut_index_real = index_cos(t_phaseBitWidth - 3, 0);
    if (index_invert_control_real == 1) lut_index_real = ((~lut_index_real) + 1);
    tableType lut_out_real = p_table[lut_index_real].imag();
    tableType temp_out_cos;
    if (output_saturation_control_real)
        temp_out_cos = MAX_OUT;
    else
        temp_out_cos = lut_out_real;
    if (output_negate_control_real)
        realCosVal = -temp_out_cos;
    else
        realCosVal = temp_out_cos;
    complexOut.imag(imagSinVal);
    complexOut.real(realCosVal);
    return (T_dtype_in)complexOut;
    // return p_table[p_index];
}

template <int t_L, int t_R, unsigned int t_phaseBitWidth, typename T_dtype_in>
T_dtype_in readQuaterTwiddleTableReverse(ap_uint<t_phaseBitWidth> p_index, T_dtype_in p_table[]) {
#pragma HLS INLINE
    const ap_int<2> MAX_OUT = -1;
    ap_uint<t_phaseBitWidth> index_cos = p_index + (3 * t_L / 4);
    ap_uint<t_phaseBitWidth> index_sin = p_index;
    typedef typename TwiddleTypeCastingTraits<T_dtype_in>::T_roundingBasedCastType T_dtype;
    typedef typename TwiddleTraits<T_dtype>::T_twiddleInnerType tableType;
    tableType realCosVal;
    tableType imagSinVal;
    T_dtype complexOut;
    ap_uint<1> index_invert_control_imag = index_sin(t_phaseBitWidth - 2, t_phaseBitWidth - 2);
    ap_uint<1> output_negate_control_imag = index_sin(t_phaseBitWidth - 1, t_phaseBitWidth - 1);
    ap_uint<1> output_saturation_control_imag = (index_sin == (t_L / 4) || (index_sin == (3 * t_L / 4)));
    ap_uint<t_phaseBitWidth - 2> lut_index_imag = index_sin(t_phaseBitWidth - 3, 0);
    if (index_invert_control_imag == 1) lut_index_imag = ((~lut_index_imag) + 1);
    tableType lut_out_imag = p_table[lut_index_imag].imag();

    tableType temp_out_sin;
    if (output_saturation_control_imag)
        temp_out_sin = MAX_OUT;
    else
        temp_out_sin = lut_out_imag;

    if (output_negate_control_imag)
        imagSinVal = temp_out_sin; // sab
    else
        imagSinVal = -temp_out_sin; // sab

    ap_uint<1> index_invert_control_real = index_cos(t_phaseBitWidth - 2, t_phaseBitWidth - 2);
    ap_uint<1> output_negate_control_real = index_cos(t_phaseBitWidth - 1, t_phaseBitWidth - 1);
    ap_uint<1> output_saturation_control_real = (index_cos == (t_L / 4) || (index_cos == (3 * t_L / 4)));
    ap_uint<t_phaseBitWidth - 2> lut_index_real = index_cos(t_phaseBitWidth - 3, 0);
    if (index_invert_control_real == 1) lut_index_real = ((~lut_index_real) + 1);
    tableType lut_out_real = p_table[lut_index_real].imag();
    tableType temp_out_cos;
    if (output_saturation_control_real)
        temp_out_cos = MAX_OUT;
    else
        temp_out_cos = lut_out_real;
    if (output_negate_control_real)
        realCosVal = -temp_out_cos;
    else
        realCosVal = temp_out_cos;
    complexOut.imag(imagSinVal);
    complexOut.real(realCosVal);
    return (T_dtype_in)complexOut;
}

} // end namespace fft
} // end namespace dsp
} // end namespace xf

#endif // HLS_SSR_FFT_TWIDDLE_TALBE_H_
