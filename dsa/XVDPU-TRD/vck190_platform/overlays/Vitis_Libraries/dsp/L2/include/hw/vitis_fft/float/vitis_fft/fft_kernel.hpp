/*
 * Copyright 2021 Xilinx, Inc.
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

#ifndef FFT_KERNEL_H_
#define FFT_KERNEL_H_

#ifndef __SYNTHESIS__
#include <assert.h>
#endif
#include <hls_stream.h>
#include <ap_int.h>

#include "vitis_fft/hls_ssr_fft.hpp"

namespace xf {
namespace dsp {
namespace fft {

template <int t_L, int t_R, int t_NUM_FFT_MAX, typename T_in>
void readLines(ap_uint<512> p_fftInData[t_L * t_NUM_FFT_MAX / t_R], hls::stream<T_in> strmOut[t_R], int n_frames) {
    for (int n = 0; n < n_frames; n++) {
        for (int i = 0; i < t_L / t_R; i++) {
#pragma HLS pipeline II = 1
            ap_uint<512> singleSuperSample = p_fftInData[n * t_L / t_R + i];
            for (int j = 0; j < t_R; j++) {
#pragma HLS unroll
                T_in singleSample;
                singleSample.real(singleSuperSample.range(31 + 64 * j, 64 * j));
                singleSample.imag(singleSuperSample.range(63 + 64 * j, 32 + 64 * j));
                strmOut[j].write(singleSample);
            }
        }
    }
}

template <int t_L, int t_R, int t_NUM_FFT_MAX, typename T_out>
void writeLines(hls::stream<T_out> strmIn[t_R], ap_uint<512> p_fftOutData[t_L * t_NUM_FFT_MAX / t_R], int n_frames) {
    for (int n = 0; n < n_frames; n++) {
        for (int i = 0; i < t_L / t_R; i++) {
#pragma HLS pipeline II = 1
            ap_uint<512> singleSuperSample;
            for (int j = 0; j < t_R; j++) {
#pragma HLS unroll
                T_out singleSample = strmIn[j].read();
                singleSuperSample.range(31 + 64 * j, 64 * j) = singleSample.real();
                singleSuperSample.range(63 + 64 * j, 32 + 64 * j) = singleSample.imag();
            }
            p_fftOutData[n * t_L / t_R + i] = singleSuperSample;
        }
    }
}

template <typename ssr_fft_param_struct, int t_instanceID, typename T_in>
void fftKernel(
    ap_uint<512> p_fftInData[ssr_fft_param_struct::N * ssr_fft_param_struct::NUM_FFT_MAX / ssr_fft_param_struct::R],
    ap_uint<512> p_fftOutData[ssr_fft_param_struct::N * ssr_fft_param_struct::NUM_FFT_MAX / ssr_fft_param_struct::R],
    int n_frames) {
    enum { FIFO_SIZE = 2 * ssr_fft_param_struct::N / ssr_fft_param_struct::R };
    //#pragma HLS INLINE
    //#pragma HLS DATAFLOW disable_start_propagation

    static const int t_L = ssr_fft_param_struct::N;
    static const int t_R = ssr_fft_param_struct::R;
    static const int t_NUM_FFT_MAX = ssr_fft_param_struct::NUM_FFT_MAX;
    static const scaling_mode_enum t_scalingMode = ssr_fft_param_struct::scaling_mode;
    static const fft_output_order_enum tp_output_data_order = ssr_fft_param_struct::output_data_order;
    static const int tw_WL = ssr_fft_param_struct::twiddle_table_word_length;
    static const int tw_IL = ssr_fft_param_struct::twiddle_table_intger_part_length;
    static const transform_direction_enum transform_direction = ssr_fft_param_struct::transform_direction;
    static const butterfly_rnd_mode_enum butterfly_rnd_mode = ssr_fft_param_struct::butterfly_rnd_mode;
    typedef typename FFTInputTraits<T_in>::T_castedType casted_type;
#pragma HLS dataflow

#ifndef __SYNTHESIS__
    checkFFTparams<t_L, t_R, tw_WL, tw_IL>();
#endif
    typedef typename InputBasedTwiddleTraits<ssr_fft_param_struct, casted_type>::T_twiddleType T_fftTwiddleType;
    typedef typename InputBasedTwiddleTraits<ssr_fft_param_struct, casted_type>::T_expTabType T_complexExpTableType;

#ifndef __SYNTHESIS__
    assert((t_R) == (ssrFFTPow<2, ssrFFTLog2<t_R>::val>::val)); // radix should be power of 2 always
    assert((t_L) == (ssrFFTPow<2, ssrFFTLog2<t_L>::val>::val)); // Length of FFt should be power of 2 always
#endif
    hls::stream<T_in> fftInStrm[t_R];
#pragma HLS stream variable = fftInStrm depth = FIFO_SIZE
    hls::stream<typename FFTOutputTraits<t_L, t_R, t_scalingMode, transform_direction, butterfly_rnd_mode,
                                         casted_type>::T_FFTOutType>
        fftOutStrm[t_R];
#pragma HLS stream variable = fftOutStrm depth = FIFO_SIZE

    readLines<t_L, t_R, t_NUM_FFT_MAX, T_in>(p_fftInData, fftInStrm, n_frames);
    FFTWrapper<(((ssrFFTLog2<t_L>::val) % (ssrFFTLog2<t_R>::val)) > 0), (t_L) < ((t_R * t_R)), t_instanceID>
        ssr_fft_wrapper_obj;
    for (int n = 0; n < n_frames; n++) {
        ssr_fft_wrapper_obj
            .template innerFFT<t_L, t_R, t_instanceID, t_scalingMode, transform_direction, butterfly_rnd_mode,
                               tp_output_data_order, T_complexExpTableType, T_fftTwiddleType, T_in,
                               typename FFTOutputTraits<t_L, t_R, t_scalingMode, transform_direction,
                                                        butterfly_rnd_mode, casted_type>::T_FFTOutType>(fftInStrm,
                                                                                                        fftOutStrm);
    }
    writeLines<t_L, t_R, t_NUM_FFT_MAX, typename FFTOutputTraits<t_L, t_R, t_scalingMode, transform_direction,
                                                                 butterfly_rnd_mode, casted_type>::T_FFTOutType>(
        fftOutStrm, p_fftOutData, n_frames);
}

} // namespace fft
} // namespace dsp
} // namespace xf

#endif // !FFT_KERNEL_H_
