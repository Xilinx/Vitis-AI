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

// hls_ssr_fft_transposer.hpp
#ifndef HLS_SSR_FFT_TRANSPOSER_H_
#define HLS_SSR_FFT_TRANSPOSER_H_

#ifndef __SYNTHESIS__
#include <iostream>
#include <assert.h>
#endif
#include "vitis_fft/hls_ssr_fft_utilities.hpp"
#include "vitis_fft/hls_ssr_fft_data_commutor.hpp"

namespace xf {
namespace dsp {
namespace fft {

template <int t_numOfStages, int t_outputForkingFactor>
struct InputTransposeChain {
    template <int t_L, int t_R, int t_PF, typename t_dtype>
    void swap(t_dtype p_in[t_R][t_L / t_R], t_dtype p_out[t_R][t_L / t_R]);
};

// These  declarations cover base case for modeling input swap functions were Length of SSR FFT is always integer
// power of Radix R or SSR
////////////////////////////////////////////////////////////////////////
template <int t_numOfStages>
struct InputTransposeChain<t_numOfStages, 1> {
    template <int t_L, int t_R, int t_PF, typename t_dtype>
    void swap(t_dtype p_in[t_R][t_L / t_R], t_dtype p_out[t_R][t_L / t_R]);
};
/////////////////////////////////////////////////////////////////////////
/*
 1.)
 */
//  The base case when Forking Factor=1 ( Non Forked Version essentially and also the last stage of recursion)
template <>
struct InputTransposeChain<1, 1> {
    template <int t_L, int t_R, int t_PF, typename t_dtype>
    void swap(t_dtype p_in[t_R][t_L / t_R], t_dtype p_out[t_R][t_L / t_R]) {
#pragma HLS INLINE

        static const int t_isLargeMemFlag = (((t_PF * t_R) > SSR_FFT_URAM_SELECTION_THRESHHOLD) && SSR_FFT_USE_URAMS);
        DataCommutations<10000, 1, t_PF, t_isLargeMemFlag> commutor;
        commutor.template dataCommutor<t_L, t_R, t_dtype>(p_in, p_out);
    }
};
/////////////////////////////////////////////////////////////////////////
// Struct declaration for the case where forking factor is > 1 and stage is 1
// which is actually the terminal case to stop recursion

template <int t_outputForkingFactor>
struct InputTransposeChain<1, t_outputForkingFactor> {
    template <int t_L, int t_R, int t_PF, typename t_dtype>
    void swap(t_dtype p_in[t_R][t_L / t_R], t_dtype p_out[t_R][t_L / t_R]);
};
////////////////////////////////////////////////////////////////////////
/*
 2) Base Case Implementation where L is integer power of radix ///start*/
template <int t_numOfStages>
template <int t_L, int t_R, int t_PF, typename t_dtype>
void InputTransposeChain<t_numOfStages, 1>::swap(t_dtype p_in[t_R][t_L / t_R], t_dtype p_out[t_R][t_L / t_R]) {
#pragma HLS INLINE // The swap is p_in-lined p_in a data-flow region to expose dataCommutor to become a process.
    // Recursion will create a chain of processes....

    t_dtype temp[t_R][t_L / t_R];
#pragma HLS DATA_PACK variable = temp
#pragma HLS STREAM variable = temp depth = 8 dim = 2
#pragma HLS RESOURCE variable = temp core = FIFO_LUTRAM

#ifdef SSR_FFT_PARTITION_INTERFACE_ARRAYS
#pragma HLS ARRAY_PARTITION variable = temp complete dim = 1
#else
#pragma HLS ARRAY_RESHAPE variable = temp complete dim = 1
#endif

    static const int t_isLargeMemFlag = (((t_PF * t_R) > SSR_FFT_URAM_SELECTION_THRESHHOLD) && SSR_FFT_USE_URAMS);

    DataCommutations<10000 + t_numOfStages, 0, t_PF, t_isLargeMemFlag> commutor;
    commutor.template dataCommutor<t_L, t_R, t_dtype>(p_in, temp);

    // Instantiate Next State....Recursion
    InputTransposeChain<t_numOfStages - 1, 1> nextStage;
    nextStage.template swap<t_L, t_R, (t_PF * t_R), t_dtype>(temp, p_out); // Supply Next Stage PF
}

// Base Case Implementation where L is integer power of radix

////////////////////////////////////////////////////////////////////////
/*
 3) The generic case where s > 1 and forking factor is > 1 : it structure is same like case when s>1 and f=1*/
template <int t_numOfStages, int t_outputForkingFactor>
template <int t_L, int t_R, int t_PF, typename t_dtype>
void InputTransposeChain<t_numOfStages, t_outputForkingFactor>::swap(t_dtype p_in[t_R][t_L / t_R],
                                                                     t_dtype p_out[t_R][t_L / t_R]) {
#pragma HLS INLINE // The swap is p_in-lined p_in a data-flow region to expose dataCommutor to become a process.
    // Recursion will create a chain of processes....
    t_dtype temp[t_R][t_L / t_R];

#pragma HLS STREAM variable = temp depth = 8 dim = 2
#pragma HLS RESOURCE variable = temp core = FIFO_LUTRAM

#pragma HLS ARRAY_PARTITION variable = temp complete dim = 1
#pragma HLS DATA_PACK variable = temp

    static const int t_isLargeMemFlag = (((t_PF * t_R) > SSR_FFT_URAM_SELECTION_THRESHHOLD) && SSR_FFT_USE_URAMS);

    DataCommutations<10000 + t_numOfStages, 0, t_PF, t_isLargeMemFlag> commutor;
    commutor.template dataCommutor<t_L, t_R, t_dtype>(p_in, temp);

    // Instantiate Next State....Recursion
    InputTransposeChain<t_numOfStages - 1, t_outputForkingFactor> nextStage;
    nextStage.template swap<t_L, t_R, (t_PF * t_R), t_dtype>(temp, p_out); // Supply Next Stage PF
}

////////////////////////////////////////////////////////////////////////
/*
 4)  The ther terminal case for forking output.
 */
template <int t_outputForkingFactor>
template <int t_L, int t_R, int t_PF, typename t_dtype>
void InputTransposeChain<1, t_outputForkingFactor>::swap(t_dtype p_in[t_R][t_L / t_R], t_dtype p_out[t_R][t_L / t_R]) {
#pragma HLS INLINE
    DataCommutorFork<10000 + 1, t_outputForkingFactor, t_outputForkingFactor> DataCommutorFork_obj;
    DataCommutorFork_obj.template copyForkCommuteAndMerge<t_PF, t_L, t_R, t_dtype>(p_in, p_out);
}
} // end namespace fft
} // end namespace dsp
} // end namespace xf

#endif // HLS_SSR_FFT_TRANSPOSER_H_
