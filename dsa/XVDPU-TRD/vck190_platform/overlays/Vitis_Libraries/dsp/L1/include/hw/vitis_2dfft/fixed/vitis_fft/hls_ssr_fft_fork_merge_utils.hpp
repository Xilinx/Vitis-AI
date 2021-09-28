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

// hls_ssr_fft_fork_merge_utils.hpp
#ifndef HLS_SSR_FFT_FORK_MERGE_UTILS_H_
#define HLS_SSR_FFT_FORK_MERGE_UTILS_H_

#include "vitis_fft/hls_ssr_fft_utilities.hpp"
#include "vitis_fft/hls_ssr_fft_super_sample.hpp"
#include "vitis_fft/hls_ssr_fft_enums.hpp"

/*
 =========================================================================================
 -_-                                                                                   -_-
 -_-                                                                                   -_-
 -_-                                                                                   -_-
 -_-                                                                                   -_-
 -_-                                                                                   -_-
 -_-                                                                                   -_-
 -_-

 SSR FFT in last few stages break its kernel into small FFT kernel
 by factor called forking factor. Generally most of the SSR FFT
 block in dataflow pipeline process blocks of data sized [t_R][t_L/t_R]
 but for certain special cases when t_L is not integer power of t_R
 in last stage the large FFT kernel block breaks in small kernels of
 whose size is smaller by factor called forking factor. So dataflow pipeline
 that is processing blocks of [t_R][t_L/t_R] forks into multiple
 streams each of size [t_R/t_forkingFactor][t_L/t_R]. This forking
 happens is fft stage kernel and also in blocks that do data re oredering.
 These utility functions are used to create a smaller forked stream
 and also to merge a small stream into big one for final output.

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

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

template <int t_L, int t_R, int t_forkNumber, int t_forkingFactor, typename T_dtype>
void copyToLocalBuff(T_dtype p_in[t_R][t_L / t_R], T_dtype p_out[t_R / t_forkingFactor][t_L / t_R]) {
#pragma HLS INLINE off

    const int fork_width = t_R / t_forkingFactor;
    const int offset = (t_forkNumber - 1) * fork_width;

    for (int t = 0; t < t_L / t_R; t++) {
#pragma HLS PIPELINE II = 1 rewind
        for (int r = 0; r < t_R / t_forkingFactor; r++) {
#pragma HLS UNROLL

            p_out[r][t] = p_in[offset + r][t];
        }
    }
}

template <int t_L, int t_R, int t_forkNumber, int t_forkingFactor, typename T_dtype>
void copyFromLocalBuffToOuput(T_dtype p_in[t_R / t_forkingFactor][t_L / t_R], T_dtype p_out[t_R][t_L / t_R]) {
#pragma HLS INLINE off

    const int fork_width = t_R / t_forkingFactor;
    const int offset = (t_forkNumber - 1) * fork_width;
    // CHECK_COVEARAGE;
    for (int t = 0; t < t_L / t_R; t++) {
#pragma HLS PIPELINE II = 1 rewind
        for (int r = 0; r < t_R / t_forkingFactor; r++) {
#pragma HLS UNROLL

            // p_out[t][2*t_forkNumber-1-r]=p_in[t][t_R/t_forkingFactor-1-r];
            //			p_out[t][offset+r]=p_in[t][r];
            p_out[(r << (ssrFFTLog2<t_forkingFactor>::val)) + t_forkNumber - 1][t] = p_in[r][t];
        }
    }
}

template <int t_L, int t_R, int t_forkNumber, int t_forkingFactor, typename T_dtype>
void copyBuffToOutNonInvert(T_dtype p_in[t_R / t_forkingFactor][t_L / t_R], T_dtype p_out[t_R][t_L / t_R]) {
#pragma HLS INLINE off
    const int fork_width = t_R / t_forkingFactor;
    const int offset = (t_forkNumber - 1) * fork_width;

    // CHECK_COVEARAGE;

    for (int t = 0; t < t_L / t_R; t++) {
#pragma HLS PIPELINE II = 1 rewind
        for (int r = 0; r < t_R / t_forkingFactor; r++) {
#pragma HLS UNROLL

            p_out[fork_width * (t_forkNumber - 1) + r][t] = p_in[r][t];
            //			p_out[t][offset+r]=p_in[t][r];
            // p_out[t][t_forkingFactor*r + t_forkNumber-1]=p_in[t][r];
        }
    }
}

template <int t_stage, int t_id, int t_L, int t_R, typename T_dtype>
void convertArrayToSuperStream(T_dtype p_inDataArray[t_R][t_L / t_R],
                               hls::stream<SuperSampleContainer<t_R, T_dtype> >& p_out) {
#pragma HLS INLINE off
//#pragma HLS STREAM variable=p_inDataArray dim=2
#pragma HLS ARRAY_PARTITION variable = p_inDataArray complete dim = 1

    for (int i = 0; i < t_L / t_R; i++) {
#pragma HLS PIPELINE II = 1 rewind
        SuperSampleContainer<t_R, T_dtype> temp;
#pragma HLS ARRAY_PARTITION variable = temp.superSample complete dim = 1
        for (int r = 0; r < t_R; r++) {
#pragma HLS UNROLL
            temp.superSample[r] = p_inDataArray[r][i];
        }
        p_out.write(temp);
    }
}
template <int t_stage, int t_id, int t_L, int t_R, typename T_dtype>
void convertArrayToSuperStream(hls::stream<T_dtype> p_inDataArray[t_R],
                               hls::stream<SuperSampleContainer<t_R, T_dtype> >& p_out) {
#pragma HLS INLINE off
    //#pragma HLS STREAM variable=p_inDataArray dim=2
    //#pragma HLS ARRAY_PARTITION variable = p_inDataArray complete dim = 1

    for (int i = 0; i < t_L / t_R; i++) {
#pragma HLS PIPELINE II = 1 rewind
        SuperSampleContainer<t_R, T_dtype> temp;
#pragma HLS ARRAY_PARTITION variable = temp.superSample complete dim = 1
        for (int r = 0; r < t_R; r++) {
#pragma HLS UNROLL
            temp.superSample[r] = p_inDataArray[r].read(); // p_inDataArray[r][i];
        }
        p_out.write(temp);
    }
}

template <int t_stage, int t_id, int t_L, int t_R, typename T_dtype, typename T_dtype1>
void convertSuperStreamToArray(hls::stream<SuperSampleContainer<t_R, T_dtype> >& p_in,
                               T_dtype1 p_outDataArray[t_R][t_L / t_R]) {
#pragma HLS INLINE off
    SuperSampleContainer<t_R, T_dtype> temp;
#pragma HLS ARRAY_PARTITION variable = temp.superSample complete dim = 1
#pragma HLS ARRAY_PARTITION variable = p_outDataArray complete dim = 1

    for (int i = 0; i < t_L / t_R; i++) {
#pragma HLS PIPELINE II = 1 rewind

        p_in.read(temp);

        for (int r = 0; r < t_R; r++) {
#pragma HLS UNROLL
            p_outDataArray[r][i] = temp.superSample[r]; // p_in[r].read();
        }
    }
}

template <int t_stage,
          transform_direction_enum transform_direction,
          int t_id,
          int t_L,
          int t_R,
          typename T_dtype,
          typename T_dtype1>
void convertSuperStreamToArrayNScale(hls::stream<SuperSampleContainer<t_R, T_dtype> >& p_in,
                                     T_dtype1 p_outDataArray[t_R][t_L / t_R]) {
#pragma HLS INLINE off

    static const int SCALING_FACTOR = (transform_direction == REVERSE_TRANSFORM) ? t_L : 1;
    SuperSampleContainer<t_R, T_dtype> temp;
#pragma HLS ARRAY_PARTITION variable = temp.superSample complete dim = 1
    //#pragma HLS ARRAY_PARTITION variable = p_outDataArray complete dim = 1

    for (int i = 0; i < t_L / t_R; i++) {
#pragma HLS PIPELINE II = 1 rewind

        p_in.read(temp);

        for (int r = 0; r < t_R; r++) {
#pragma HLS UNROLL
            p_outDataArray[r][i] =
                T_dtype1(temp.superSample[r].real() / SCALING_FACTOR, temp.superSample[r].imag() / SCALING_FACTOR);
        }
    }
}

template <int t_stage,
          transform_direction_enum transform_direction,
          int t_id,
          int t_L,
          int t_R,
          typename T_dtype,
          typename T_dtype1>
void convertSuperStreamToArrayNScale(hls::stream<SuperSampleContainer<t_R, T_dtype> >& p_in,
                                     hls::stream<T_dtype1> p_outDataArray[t_R]) {
#pragma HLS INLINE off

    static const int SCALING_FACTOR = (transform_direction == REVERSE_TRANSFORM) ? t_L : 1;
    SuperSampleContainer<t_R, T_dtype> temp;
#pragma HLS ARRAY_PARTITION variable = temp.superSample complete dim = 1
    //#pragma HLS ARRAY_PARTITION variable = p_outDataArray complete dim = 1

    for (int i = 0; i < t_L / t_R; i++) {
#pragma HLS PIPELINE II = 1 rewind

        p_in.read(temp);

        for (int r = 0; r < t_R; r++) {
#pragma HLS UNROLL
            p_outDataArray[r].write(
                T_dtype1(temp.superSample[r].real() / SCALING_FACTOR, temp.superSample[r].imag() / SCALING_FACTOR));
        }
    }
}

template <int t_stage,
          transform_direction_enum transform_direction,
          int t_id,
          int t_L,
          int t_R,
          typename T_dtype,
          typename T_dtype1>
void superStreamNScale(hls::stream<SuperSampleContainer<t_R, T_dtype> >& p_in,
                       hls::stream<SuperSampleContainer<t_R, T_dtype1> >& p_out) {
#pragma HLS INLINE off
    static const int SCALING_FACTOR = (transform_direction == REVERSE_TRANSFORM) ? t_L : 1;

    for (int t = 0; t < t_L / t_R; t++) {
#pragma HLS PIPELINE II = 1 rewind
        SuperSampleContainer<t_R, T_dtype> temp = p_in.read();
        SuperSampleContainer<t_R, T_dtype1> temp2;
        for (int r = 0; r < t_R; r++) {
#pragma HLS UNROLL

            temp2.superSample[r] =
                T_dtype1(temp.superSample[r].real() / SCALING_FACTOR, temp.superSample[r].imag() / SCALING_FACTOR);
        }
        p_out.write(temp2);
    }
}

template <int t_L, int t_R, typename T_dtype>
void streamJoinUtility(T_dtype p_in[t_R][t_L / t_R], T_dtype p_out[t_R][t_L / t_R]) {
#pragma HLS INLINE off
    for (int t = 0; t < t_L / t_R; t++) {
#pragma HLS PIPELINE II = 1 rewind
        for (int r = 0; r < t_R; r++) {
#pragma HLS UNROLL
            p_out[r][t] = p_in[r][t];
        }
    }
}
template <int t_L, int t_R, typename T_dtype>
void streamJoinUtilitySISO(hls::stream<SuperSampleContainer<t_R, T_dtype> >& p_in, T_dtype p_out[t_R][t_L / t_R]) {
#pragma HLS INLINE off
    for (int t = 0; t < t_L / t_R; t++) {
#pragma HLS PIPELINE II = 1 rewind
        SuperSampleContainer<t_R, T_dtype> temp_super_sample_in;
        temp_super_sample_in = p_in.read();
        for (int r = 0; r < t_R; r++) {
#pragma HLS UNROLL
            // p_out[r][t]=p_in[r][t];
            p_out[r][t] = temp_super_sample_in.superSample[r];
        }
    }
}
template <int t_L, int t_R, typename T_dtype>
void streamJoinUtilitySISO(hls::stream<SuperSampleContainer<t_R, T_dtype> >& p_in, hls::stream<T_dtype> p_out[t_R]) {
#pragma HLS INLINE off
    for (int t = 0; t < t_L / t_R; t++) {
#pragma HLS PIPELINE II = 1 rewind
        SuperSampleContainer<t_R, T_dtype> temp_super_sample_in;
        temp_super_sample_in = p_in.read();
        for (int r = 0; r < t_R; r++) {
#pragma HLS UNROLL
            // p_out[r][t]=p_in[r][t];
            p_out[r].write(temp_super_sample_in.superSample[r]);
        }
    }
}

template <int t_L, int t_R, int t_forkNumber, int t_forkingFactor, typename T_dtype>
void mergeSuperSampleStreamNonInvertOut(
    hls::stream<SuperSampleContainer<t_R / t_forkingFactor, T_dtype> > p_in[t_forkingFactor],
    hls::stream<SuperSampleContainer<t_R, T_dtype> >& p_outSuperStream) {
#pragma HLS INLINE off
    SuperSampleContainer<t_R / t_forkingFactor, T_dtype> sub_sample_in;
    SuperSampleContainer<t_R, T_dtype> super_sample_sample_out;
    const unsigned int fork_size = t_R / t_forkingFactor;

    for (int t = 0; t < t_L / t_R; ++t) {
#pragma HLS PIPELINE II = 1 rewind
        for (int fork_no = 0; fork_no < t_forkingFactor; fork_no++) {
#pragma HLS UNROLL
            sub_sample_in = p_in[fork_no].read();
            for (int sample_no = 0; sample_no < fork_size; sample_no++) {
#pragma HLS UNROLL
                super_sample_sample_out.superSample[fork_no * fork_size + sample_no] =
                    sub_sample_in.superSample[sample_no];
            }
        }
        p_outSuperStream.write(super_sample_sample_out);
    }
}

template <int t_L, int t_R, int t_forkNumber, int t_forkingFactor, typename T_dtype>
void forkSuperSampleStream(
    hls::stream<SuperSampleContainer<t_R, T_dtype> >& p_in,
    hls::stream<SuperSampleContainer<t_R / t_forkingFactor, T_dtype> > p_outStreamArray[t_forkingFactor]) {
#pragma HLS INLINE off

    const unsigned int fork_size = t_R / t_forkingFactor;

    SuperSampleContainer<t_R, T_dtype> temp_in_super_sample;

    SuperSampleContainer<t_R / t_forkingFactor, T_dtype> temp_factored;

    for (int t = 0; t < t_L / t_R; ++t) {
#pragma HLS PIPELINE II = 1 rewind
        temp_in_super_sample = p_in.read();

        for (int fork_no = 0; fork_no < t_forkingFactor; fork_no++) {
#pragma HLS UNROLL

            for (int sample_no = 0; sample_no < t_R / t_forkingFactor; sample_no++) {
#pragma HLS UNROLL
                temp_factored.superSample[sample_no] =
                    temp_in_super_sample.superSample[fork_no * fork_size + sample_no];
            }
            // temp_factored_array[fork_no]=temp_factored;
            p_outStreamArray[fork_no].write(temp_factored);
        }
    }
}

template <int t_L, int t_R, int t_forkNumber, int t_forkingFactor, typename T_dtype>
void mergeSuperSampleStream(hls::stream<SuperSampleContainer<t_R / t_forkingFactor, T_dtype> > p_in[t_forkingFactor],
                            hls::stream<SuperSampleContainer<t_R, T_dtype> >& p_outSuperStream) {
#pragma HLS INLINE off
    SuperSampleContainer<t_R / t_forkingFactor, T_dtype> sub_sample_in;
    SuperSampleContainer<t_R, T_dtype> super_sample_sample_out;
    const unsigned int fork_size = t_R / t_forkingFactor;

    for (int t = 0; t < t_L / t_R; ++t) {
#pragma HLS PIPELINE II = 1 rewind
        for (int fork_no = 0; fork_no < t_forkingFactor; fork_no++) {
#pragma HLS UNROLL
            sub_sample_in = p_in[fork_no].read();
            for (int sample_no = 0; sample_no < t_R / t_forkingFactor; sample_no++) {
#pragma HLS UNROLL

                super_sample_sample_out.superSample[(sample_no << (ssrFFTLog2<t_forkingFactor>::val)) + fork_no] =
                    sub_sample_in.superSample[sample_no];
            }
        }
        p_outSuperStream.write(super_sample_sample_out);
    }
}

} // end namespace fft
} // end namespace dsp
} // end namespace xf

#endif // HLS_SSR_FFT_FORK_MERGE_UTILS_H_
