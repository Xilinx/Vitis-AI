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

// File Name:hls_ssr_fft_rotate_criss_cross_multiplexer.hpp
#ifndef HLS_SSR_FFT_ROTATE_CRISS_CROSS_MULTIPLEXER_H_
#define HLS_SSR_FFT_ROTATE_CRISS_CROSS_MULTIPLEXER_H_

#include "vitis_fft/hls_ssr_fft_utilities.hpp"

namespace xf {
namespace dsp {
namespace fft {

template <int t_stage>
struct RotateCrissCrossAndMultiplex {
    template <int t_R, int t_L, typename T_in, typename T_out>
    void rotateCrissCrossMultiplexer(int p_timeIndexAddressOffset,
                                     int p_ssrDimensionAddressOffset,
                                     T_in p_cin[t_R][t_L / t_R],
                                     T_out p_pout[t_R]);
};

template <int t_stage>
template <int t_R, int t_L, typename T_in, typename T_out>
void RotateCrissCrossAndMultiplex<t_stage>::rotateCrissCrossMultiplexer(int p_timeIndexAddressOffset,
                                                                        int p_ssrDimensionAddressOffset,
                                                                        T_in p_cin[t_R][t_L / t_R],
                                                                        T_out p_pout[t_R]) {
#pragma HLS INLINE
    const unsigned int F = ((t_R * t_R) / t_L) > 1 ? ((t_R * t_R) / t_L) : 1;
    const unsigned int outputShuffleAmount = ssrFFTLog2<F>::val;

    if (p_ssrDimensionAddressOffset == (t_stage - 1)) {
    SSR_LOOP:
        for (unsigned int r = 0; r < t_R; r++) {
#pragma HLS UNROLL

            ap_uint<ssrFFTLog2<t_R>::val> rotated_r = r;
            rotated_r.lrotate(outputShuffleAmount);
            // replaced//int pingPongSuperSampleIndex = ((t_stage-1) + rotated_r) % t_R;
            int pingPongSuperSampleIndex = ((t_stage - 1) + rotated_r) & (ssrFFTLog2BitwiseAndModMask<t_R>::val);

            // CHECK_COVEARAGE;
            // replaced//int pingPongTimeIndex=(r+p_timeIndexAddressOffset)%(t_L/t_R);
            int pingPongTimeIndex = (r + p_timeIndexAddressOffset) & (ssrFFTLog2BitwiseAndModMask<t_L / t_R>::val);

            p_pout[r] = p_cin[pingPongSuperSampleIndex][pingPongTimeIndex];
        }
    } else {
        RotateCrissCrossAndMultiplex<t_stage - 1> obj;
        obj.template rotateCrissCrossMultiplexer<t_R, t_L, T_in, T_out>(p_timeIndexAddressOffset,
                                                                        p_ssrDimensionAddressOffset, p_cin, p_pout);
    }
}

template <>
template <int t_R, int t_L, typename T_in, typename T_out>
void RotateCrissCrossAndMultiplex<1>::rotateCrissCrossMultiplexer(int p_timeIndexAddressOffset,
                                                                  int p_ssrDimensionAddressOffset,
                                                                  T_in p_cin[t_R][t_L / t_R],
                                                                  T_out p_pout[t_R]) {
#pragma HLS INLINE
    const unsigned int F = ((t_R * t_R) / t_L) > 1 ? ((t_R * t_R) / t_L) : 1;
    const unsigned int outputShuffleAmount = ssrFFTLog2<F>::val;

SSR_LOOP:
    for (unsigned int r = 0; r < t_R; r++) {
#pragma HLS UNROLL

        ap_uint<ssrFFTLog2<t_R>::val> rotated_r = r;
        rotated_r.lrotate(outputShuffleAmount);
        // replaced//int pingPongSuperSampleIndex = ((1-1) + rotated_r) % t_R;
        int pingPongSuperSampleIndex = ((1 - 1) + rotated_r) & (ssrFFTLog2BitwiseAndModMask<t_R>::val);

        // replaced//int pingPongTimeIndex=(r+p_timeIndexAddressOffset)%(t_L/t_R);
        int pingPongTimeIndex = (r + p_timeIndexAddressOffset) & (ssrFFTLog2BitwiseAndModMask<(t_L / t_R)>::val);
        // CHECK_COVEARAGE;

        p_pout[r] = p_cin[pingPongSuperSampleIndex][pingPongTimeIndex];
    }
}

} // end namespace fft
} // end namespace dsp
} // end namespace xf

#endif // HLS_SSR_FFT_ROTATE_CRISS_CROSS_MULTIPLEXER_H_
