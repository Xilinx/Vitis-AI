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

// hls_ssr_fft_read_barrel_shifter.hpp
#ifndef HLS_SSR_FFT_READ_BARREL_SHIFTER_H_
#define HLS_SSR_FFT_READ_BARREL_SHIFTER_H_

#include "vitis_fft/hls_ssr_fft_utilities.hpp"

/*
 =========================================================================================
 -_-                                                                                   -_-
 -_-                                                                                   -_-
 -_-                                                                                   -_-
 -_-                                                                                   -_-
 -_-                                                                                   -_-
 -_-                                                                                   -_-
 -_-

 The digitReversedDataReOrder function is used to perform data re ordering at
 the output of final stage in SSR FFT. The data produced by final stage is
 shuffled in SSR dimension ( R or SSR streams coming from SSR FFT last stage)
 and also it needs re ordering in time dimension. The re or ordering is done
 using two PIPO buffers, in 4 phases.

 1- The input streams are rotated in SSR dimension ( R Streams)
 2- The input stream written to PIPO after rotation
 3- The ping pong memory is read
 4- The read data is shuffled and written to output
 This file defines functions for phases 1,2,3
 cacheData function : deals with phase 1 and 2
 WriteCacheData  function deals with phase  3,4
 and internally calls
 MemReadBarrelShifter::readMemAndBarrelShift
 Note : This function only deals with the cases when t_L is integer power of
 t_R , for the cases when t_L is not integer power of t_R :
 OutputDataReOrder<>::digitReversal2Phase<> is used which
 deals with it through different specializations.

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

template <int t_stage>
struct MemReadBarrelShifter {
    template <int t_R, int t_L, typename T_dtype>
    void readMemAndBarrelShift(int p_lindex, int p_offset, T_dtype p_in[t_R][t_L / t_R], T_dtype p_out[t_R]);
};

template <int t_stage>
template <int t_R, int t_L, typename T_dtype>
void MemReadBarrelShifter<t_stage>::readMemAndBarrelShift(int p_lindex,
                                                          int p_offset,
                                                          T_dtype p_in[t_R][t_L / t_R],
                                                          T_dtype p_out[t_R]) {
#pragma HLS INLINE
    const int c_shift = (ssrFFTLog2<t_L>::val) - (ssrFFTLog2<t_R>::val);
    const int log2_radix = (ssrFFTLog2<t_R>::val);

    if (p_offset == (t_stage - 1)) {
        for (int c = 0; c < t_R; c++) {
#pragma HLS UNROLL
            unsigned int lin_index = (p_lindex << log2_radix) | c; // equivalent to : r*t_R + c;
            unsigned int bitReversedIndex = digitReversal<t_L, t_R>(lin_index);
            unsigned int out_r = bitReversedIndex >> log2_radix; // equivalent to :  bitReversedIndex / t_R;
            // replaced//p_out[c]= p_in[(c+(t_stage-1))%t_R][out_r];
            p_out[c] = p_in[(c + (t_stage - 1)) & (ssrFFTLog2BitwiseAndModMask<t_R>::val)][out_r];
        }
    } else {
        MemReadBarrelShifter<t_stage - 1> obj;
        obj.template readMemAndBarrelShift<t_R, t_L, T_dtype>(p_lindex, p_offset, p_in, p_out);
    }
}

template <>
template <int t_R, int t_L, typename T_dtype>
void MemReadBarrelShifter<1>::readMemAndBarrelShift(int p_lindex,
                                                    int p_offset,
                                                    T_dtype p_in[t_R][t_L / t_R],
                                                    T_dtype p_out[t_R]) {
#pragma HLS INLINE
    const int c_shift = (ssrFFTLog2<t_L>::val) - (ssrFFTLog2<t_R>::val);
    const int log2_radix = (ssrFFTLog2<t_R>::val);

    for (int c = 0; c < t_R; c++) {
#pragma HLS UNROLL
        unsigned int lin_index = (p_lindex << log2_radix) | c; // equivalent to : r*t_R + c;
        unsigned int bitReversedIndex = digitReversal<t_L, t_R>(lin_index);
        unsigned int out_r = bitReversedIndex >> log2_radix; // equivalent to :  bitReversedIndex / t_R;
        p_out[c] = p_in[(c + (1 - 1)) % t_R][out_r];
    }
}
} // end namespace fft
} // end namespace dsp
} // end namespace xf

#endif // HLS_SSR_FFT_READ_BARREL_SHIFTER_H_
