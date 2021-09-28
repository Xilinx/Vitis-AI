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

/* hls_ssr_fft_utilities.hpp
*
* Author: Shahzad Ahmad Butt
*
*/ ////////////////////////////////////////////////////////////////////

#ifndef HLS_SSR_FFT_UTILITIES_H_
#define HLS_SSR_FFT_UTILITIES_H_

#ifndef __SYNTHESIS__
#include <iostream>
#endif

/*
 =========================================================================================
 -_- -_-
 -_- -_-
 -_- -_-
 -_- -_-
 -_- -_-
 -_- -_-
 Different utility functions are defined in this file. Including
 functions for calculating log2, pow at compile time. Two other
 utility functions used by SSR FFT called digitReversal_m and
 digitReversal_fracIsLsb are defined in this file. Both of these
 functions are used for digital reversal, digit reversal is the
 process where given integer is broken into log2(R) bit parts
 and these parts as whole are assumed as digit and reversed as
 whole digits ( like bits are reversed in ordinary bit reversal).
 Two different functions are defined for selecting the direction
 of reversal. In the case of digitReversal_fracIsLsb if the total
 number of bits used to represent the number is not multiple of
 digit size then fractional bits are assumed on the lsb side and
 used as one whole digit while reversal is performed.


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

#include <ap_int.h>

namespace xf {
namespace dsp {
namespace fft {

/* template <int L>
 struct powerOf2CheckonL
 {
 void check()
 {

 }
 };

 template <>
 void powerOf2CheckonL<0>::check()
 {
 #error    "----------------The Selected FFT Length is not Power of 2-----------------"
 }


 template <int L>
 struct powerOf2CheckonRadix
 {
 void check()
 {

 }
 };

 template <>
 void powerOf2CheckonRadix<0>::check()
 {
 #error    "----------------The Selected FFT Radix or SSR is not Power of 2-----------------"
 }

 */
void inline check_covearage() {
#ifndef __SYNTHESIS__

    std::cout << "\n\n\n\n\n\n\n\nCovered;;;;;;;;\n" << __FILE__ << __LINE__ << "<<\n\n\n\n\n\n\n\n";
#endif

    // exit(1);
}
template <int t_num> //
struct ssrFFTLog2 {
    static const int val = 1 + ssrFFTLog2<t_num / 2>::val;
};
template <>
struct ssrFFTLog2<1> {
    static const int val = 0;
};

template <unsigned int t_num, unsigned int p_p>
struct ssrFFTPow {
    static const unsigned long int val = t_num * ssrFFTPow<t_num, p_p - 1>::val;
};

template <unsigned int t_num>
struct ssrFFTPow<t_num, 1> {
    static const unsigned long int val = t_num;
};
template <unsigned int t_num>
struct ssrFFTPow<t_num, 0> {
    static const unsigned long int val = 1;
};

template <int t_num>
struct ssrFFTLog2BitwiseAndModMask {
    static const int val = (1 << (ssrFFTLog2<t_num>::val)) - 1;
};
template <unsigned int t_L, unsigned int t_R>
unsigned int digitReversal(unsigned int number) {
// CHECK_COVEARAGE;
#pragma HLS INLINE
    unsigned int log_radix_t_L = (ssrFFTLog2<t_L>::val) / (ssrFFTLog2<t_R>::val);
    unsigned int log2_radix = (ssrFFTLog2<t_R>::val);
    unsigned int mask = (1 << log2_radix) - 1;
    unsigned int result = 0;
    unsigned temp = number;
    for (int i = 0; i < log_radix_t_L; i++) {
#pragma HLS UNROLL
        unsigned int log2_r_bits_lsbs = temp & mask;
        temp = temp >> log2_radix;
        result = (result << log2_radix) | log2_r_bits_lsbs;
    }

    return result;
}

template <unsigned int t_L, unsigned int t_R>
unsigned int digitReversal_m(unsigned int p_number) {
#pragma HLS INLINE
    // CHECK_COVEARAGE;

    ap_uint<ssrFFTLog2<t_L>::val> number = p_number;
    ap_uint<ssrFFTLog2<t_L>::val> reversedNumber = 0;
    ap_uint<ssrFFTLog2<t_R>::val> digitContainer;
    const unsigned int log2_of_R = ssrFFTLog2<t_R>::val;
    const unsigned int log2_of_L = ssrFFTLog2<t_L>::val;
    const unsigned int numDigits = log2_of_L / log2_of_R;
    const unsigned int fracDigitBits = log2_of_L % log2_of_R;
    for (int i = 0; i < numDigits; i++) {
#pragma HLS UNROLL
        // reversedNumber << log2_of_R;

        reversedNumber(log2_of_L - (i * log2_of_R) - 1, log2_of_L - ((i + 1) * log2_of_R)) =
            number((log2_of_R * (i + 1)) - 1, (log2_of_R * i));
    }
    if ((log2_of_L % log2_of_R) != 0)
        reversedNumber(fracDigitBits - 1, 0) = number(log2_of_L - 1, log2_of_L - fracDigitBits);

    unsigned int result = reversedNumber.to_uint();

    return result;
}

template <unsigned int t_L, unsigned int t_R>
unsigned int digitReversalFractionIsLSB(unsigned int p_number) {
#pragma HLS INLINE
    // CHECK_COVEARAGE;

    ap_uint<ssrFFTLog2<t_L>::val> number = p_number;
    ap_uint<ssrFFTLog2<t_L>::val> reversedNumber = 0;
    ap_uint<ssrFFTLog2<t_R>::val> digitContainer;
    const unsigned int log2_of_R = ssrFFTLog2<t_R>::val;
    const unsigned int log2_of_L = ssrFFTLog2<t_L>::val;
    const unsigned int numDigits = log2_of_L / log2_of_R;
    const unsigned int fracDigitBits = log2_of_L % log2_of_R;
    if ((log2_of_L % log2_of_R) != 0)
        reversedNumber(log2_of_L - 1, log2_of_L - fracDigitBits) = number(fracDigitBits - 1, 0);
    for (int i = 0; i < numDigits; i++) {
#pragma HLS UNROLL
        // reversedNumber << log2_of_R;

        reversedNumber((log2_of_L - fracDigitBits - 1) - (log2_of_R * i),
                       (log2_of_L - fracDigitBits) - (log2_of_R * (i + 1))) =
            number((log2_of_R * (i + 1)) + fracDigitBits - 1, (log2_of_R * i + fracDigitBits));
    }
    // if(  (log2_of_L % log2_of_R) != 0 )
    // reversedNumber(fracDigitBits-1,0) =number(log2_of_L-1,log2_of_L-fracDigitBits);

    unsigned int result = reversedNumber.to_uint();

    return result;
}

} // end namespace fft
} // end namespace dsp
} // end namespace xf

#endif /* HLS_SSR_FFT_UTILITIES_H_ */
