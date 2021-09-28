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

// hls_ssr_fft_data_commute_barrel_shifter.hpp
#ifndef HLS_SSR_FFT_DATA_COMMUTE_BARREL_SHIFTER_H_
#define HLS_SSR_FFT_DATA_COMMUTE_BARREL_SHIFTER_H_

/*
 =========================================================================================
 -_-                                                                                   -_-
 -_-                                                                                   -_-
 -_-                                                                                   -_-
 -_-                                                                                   -_-
 -_-                                                                                   -_-
 -_-                                                                                   -_-
 -_-

 The datacommoutor is a basic block used to implement data re ordering
 at the input SSR FFT and als in between the SSR FFT Stages. The data commutor
 has two stages one read in R streams multiplexing them before storage to ping
 pong buffers in a circular rotate(word level rotation not bit like) fashion.
 The 2nd t_stage reads in R memories and streams it p_out to R different streams.
 The memory to stream mapping changes in every cycle. The Whole transformation
 is 4 phase:
 1- The input streams are rotated
 2- The input stream written to PIPO after rotation
 3- The ping pong memory is read
 4- The read data is shuffled and written to output
 CommuteBarrelShifter::memReadCommuteBarrelShifter
 defined in this file
 is used in phase 4.
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

#include "vitis_fft/hls_ssr_fft_utilities.hpp"

namespace xf {
namespace dsp {
namespace fft {

template <int t_stage>
struct CommuteBarrelShifter {
    template <int t_R, int t_L, int t_PF, typename T_dtpye>
    void memReadCommuteBarrelShifter(int p_offset, int p_p, T_dtpye p_in[t_R][t_PF * t_R], T_dtpye p_out[t_R]);
};
template <int t_stage>
template <int t_R, int t_L, int t_PF, typename T_dtpye>
void CommuteBarrelShifter<t_stage>::memReadCommuteBarrelShifter(int p_offset,
                                                                int p_p,
                                                                T_dtpye p_in[t_R][t_PF * t_R],
                                                                T_dtpye p_out[t_R]) {
#pragma HLS INLINE
    if (p_offset == (t_stage - 1)) {
        for (int c = 0; c < t_R; c++) {
#pragma HLS UNROLL
            // replaced//unsigned int out_r = c*t_PF + p_p ;   // equivalent to :  bitReversedIndex / t_R;
            unsigned int out_r = (c << ssrFFTLog2<t_PF>::val) + p_p; // equivalent to :  bitReversedIndex / t_R;
            // replaced // p_out[c]= p_in[(c+(t_stage-1))%t_R][out_r];
            p_out[c] = p_in[(c + (t_stage - 1)) & ssrFFTLog2BitwiseAndModMask<t_R>::val][out_r];
        }

    } else {
        CommuteBarrelShifter<t_stage - 1> obj;
        obj.template memReadCommuteBarrelShifter<t_R, t_L, t_PF, T_dtpye>(p_offset, p_p, p_in, p_out);
    }
}

template <>
template <int t_R, int t_L, int t_PF, typename T_dtpye>
void CommuteBarrelShifter<1>::memReadCommuteBarrelShifter(int p_offset,
                                                          int p_p,
                                                          T_dtpye p_in[t_R][t_PF * t_R],
                                                          T_dtpye p_out[t_R]) {
#pragma HLS INLINE
    const int c_shift = (ssrFFTLog2<t_L>::val) - (ssrFFTLog2<t_R>::val);
    const int log2_radix = (ssrFFTLog2<t_R>::val);

    for (int c = 0; c < t_R; c++) {
#pragma HLS UNROLL

        // replaced//unsigned int out_r = c*t_PF + p_p ;   // equivalent to :  bitReversedIndex / t_R;
        unsigned int out_r = (c << ssrFFTLog2<t_PF>::val) + p_p; // equivalent to :  bitReversedIndex / t_R;
        // replaced//p_out[c]= p_in[(c+(1-1))%t_R][out_r];
        p_out[c] = p_in[(c + (1 - 1)) & ssrFFTLog2BitwiseAndModMask<t_R>::val][out_r];
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////// PF=1 Specialization
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <int t_stage>
struct CommuteBarrelShifterPF1 {
    template <int XF, int t_R, int t_L, int t_PF, typename T_dtpye>
    void memReadCommuteBarrelShifterPF1(int p_offset, int p_p, T_dtpye p_in[t_R][XF * t_PF * t_R], T_dtpye p_out[t_R]);
};
template <int t_stage>
template <int XF, int t_R, int t_L, int t_PF, typename T_dtpye>
void CommuteBarrelShifterPF1<t_stage>::memReadCommuteBarrelShifterPF1(int p_offset,
                                                                      int p_p,
                                                                      T_dtpye p_in[t_R][XF * t_PF * t_R],
                                                                      T_dtpye p_out[t_R]) {
#pragma HLS INLINE
    if ((p_offset % t_R) == (t_stage - 1)) {
        for (int c = 0; c < t_R; c++) {
#pragma HLS UNROLL
            // replaced//unsigned int out_r = c*t_PF + p_p ;   // equivalent to :  bitReversedIndex / t_R;
            unsigned int out_r = (c << ssrFFTLog2<t_PF>::val) + p_p +
                                 ((p_offset / t_R) * t_R); // equivalent to :  bitReversedIndex / t_R;

            // replaced // p_out[c]= p_in[(c+(t_stage-1))%t_R][out_r];
            p_out[c] = p_in[(c + (t_stage - 1)) & ssrFFTLog2BitwiseAndModMask<t_R>::val][out_r];
        }

    } else {
        CommuteBarrelShifterPF1<t_stage - 1> obj;
        obj.template memReadCommuteBarrelShifterPF1<XF, t_R, t_L, t_PF, T_dtpye>(p_offset, p_p, p_in, p_out);
    }
}

template <>
template <int XF, int t_R, int t_L, int t_PF, typename T_dtpye>
void CommuteBarrelShifterPF1<1>::memReadCommuteBarrelShifterPF1(int p_offset,
                                                                int p_p,
                                                                T_dtpye p_in[t_R][XF * t_PF * t_R],
                                                                T_dtpye p_out[t_R]) {
#pragma HLS INLINE
    const int c_shift = (ssrFFTLog2<t_L>::val) - (ssrFFTLog2<t_R>::val);
    const int log2_radix = (ssrFFTLog2<t_R>::val);

    for (int c = 0; c < t_R; c++) {
#pragma HLS UNROLL

        // replaced//unsigned int out_r = c*t_PF + p_p ;   // equivalent to :  bitReversedIndex / t_R;
        unsigned int out_r =
            (c << ssrFFTLog2<t_PF>::val) + p_p + ((p_offset / t_R) * t_R); // equivalent to :  bitReversedIndex / t_R;
        // replaced//p_out[c]= p_in[(c+(1-1))%t_R][out_r];
        p_out[c] = p_in[(c + (1 - 1)) & ssrFFTLog2BitwiseAndModMask<t_R>::val][out_r];
    }
}

} // end namespace fft
} // end namespace dsp
} // end namespace xf
#endif // HLS_SSR_FFT_DATA_COMMUTE_BARREL_SHIFTER_H_
