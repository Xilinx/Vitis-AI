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

// File Name : hls_ssr_fft_streaming_transposer.hpp
/*
 =========================================================================================
 -_-                                                                                   -_-
 -_-                                                                                   -_-
 -_-                                                                                   -_-
 -_-                                                                                   -_-
 -_-                                                                                   -_-
 -_-                                                                                   -_-
 -_-

 21-sep-2018-:This file defines a streaming transposer block. This block
 will perform data.	re oredring on the input to ssr fft. The data re
 ordering is implemented using data commutor blocks which are defined using
 hls::streams as compared to the original version which is based on array
 interfaces and and PIPOs. The streaming	transposer design is intended for
 better QoR p_in terms of block ram usage.

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
#ifndef __HLS_SSR_FFT_STREAMING_TRANSPOSER__
#define __HLS_SSR_FFT_STREAMING_TRANSPOSER__

#ifndef __SYNTHESIS__
#include <iostream>
#include <assert.h>
#endif

#include "vitis_fft/hls_ssr_fft_utilities.hpp"
//#include "vitis_fft/hls_ssr_fft_data_commutor.hpp"
#include "vitis_fft/hls_ssr_fft_streaming_data_commutor.hpp"
#include "hls_ssr_fft_streaming_forking_data_commutor.hpp"
#include "vitis_fft/hls_ssr_fft_pragma_controls.hpp"

namespace xf {
namespace dsp {
namespace fft {

template <int t_instanceID,
          int t_stage,
          int t_subStage,
          int t_forkNumber,
          int t_L,
          int t_R,
          int t_PF,
          int t_outputForkingFactor>
struct InputTransposeChainStreaming {
    template <typename T_dtype>
    void swap(T_dtype p_in[t_R][t_L / t_R], T_dtype p_out[t_R][t_L / t_R]);
};

////////////////////////////////////////////////////////////////////////
// Specialization for Forking Factor(t_outputForkingFactor) = 1
//                         &&
// template<int tp_stages, int t_instanceID>
template <int t_instanceID, int t_stage, int t_subStage, int t_forkNumber, int t_L, int t_R, int t_PF>
struct InputTransposeChainStreaming<t_instanceID, t_stage, t_subStage, t_forkNumber, t_L, t_R, t_PF, 1> {
    template <typename T_dtype>
    void swap(T_dtype p_in[t_R][t_L / t_R], T_dtype p_out[t_R][t_L / t_R]);
};

// 1-ID)--------------------------------------------------------------------------------
template <int t_instanceID, int t_subStage, int t_forkNumber, int t_L, int t_R, int t_PF>
struct InputTransposeChainStreaming<t_instanceID, 1, t_subStage, t_forkNumber, t_L, t_R, t_PF, 1> {
    template <typename T_dtype>
    void swap(T_dtype p_in[t_R][t_L / t_R], T_dtype p_out[t_R][t_L / t_R]) {
#pragma HLS INLINE
        static const int t_isLargeMemFlag = (((t_PF * t_R) > SSR_FFT_URAM_SELECTION_THRESHHOLD) && SSR_FFT_USE_URAMS);

        streamingDataCommutations<t_instanceID, 1, t_subStage, t_forkNumber, t_L, t_R, t_PF, t_isLargeMemFlag> commutor;

        commutor.template streamingDataCommutor<T_dtype>(p_in, p_out);
    }
};
// 3-D)--------------------------------------------------------------------------------

/////////////////////////////////////////////////////////////////////////
// Struct declaration for the case where forking factor is > 1 and t_stage==1
// which is actually the terminal case to stop recursion
template <int t_instanceID, int t_subStage, int t_forkNumber, int t_L, int t_R, int t_PF, int t_outputForkingFactor>
struct InputTransposeChainStreaming<t_instanceID, 1, t_subStage, t_forkNumber, t_L, t_R, t_PF, t_outputForkingFactor> {
    template <typename T_dtype>
    void swap(T_dtype p_in[t_R][t_L / t_R], T_dtype p_out[t_R][t_L / t_R]);
};

////////////////////////////////////////////////////////////////////////
// Base Case Implementation where L is integer power of radix ///start
template <int t_instanceID, int t_stage, int t_subStage, int t_forkNumber, int t_L, int t_R, int t_PF>
template <typename T_dtype>
void InputTransposeChainStreaming<t_instanceID, t_stage, t_subStage, t_forkNumber, t_L, t_R, t_PF, 1>::swap(
    T_dtype p_in[t_R][t_L / t_R], T_dtype p_out[t_R][t_L / t_R]) {
#pragma HLS INLINE // The swap is p_in-lined p_in a data-flow region to expose dataCommutor to become a process.
    // Recursion will create a chain of processes....
    T_dtype temp[t_R][t_L / t_R];
#pragma HLS data_pack variable = temp
#pragma HLS STREAM variable = temp depth = 8
#pragma HLS RESOURCE variable = temp core = FIFO_LUTRAM
#ifdef SSR_FFT_PARTITION_INTERFACE_ARRAYS
#pragma HLS ARRAY_PARTITION variable = temp complete dim = 1
#else
#pragma HLS ARRAY_RESHAPE variable = temp complete dim = 1
#endif
    static const int t_isLargeMemFlag = (((t_PF * t_R) > SSR_FFT_URAM_SELECTION_THRESHHOLD) && SSR_FFT_USE_URAMS);
    streamingDataCommutations<t_instanceID, t_stage, t_subStage, t_forkNumber, t_L, t_R, t_PF, t_isLargeMemFlag>
        commutor;
    commutor.template streamingDataCommutor<T_dtype>(p_in, temp);
    // Instantiate Next State....Recursion
    InputTransposeChainStreaming<t_instanceID, t_stage - 1, t_subStage, t_forkNumber, t_L, t_R, t_PF * t_R, 1>
        nextStage; // Supply Next t_stage PF
    nextStage.template swap<T_dtype>(temp, p_out);
}

// Base Case Implementation where L is integer power of radix ///END
////////////////////////////////////////////////////////////////////////
// The generic case where s > 1 and forking factor is > 1 : it structure is same like case when s>1 and f=1
template <int t_instanceID,
          int t_stage,
          int t_subStage,
          int t_forkNumber,
          int t_L,
          int t_R,
          int t_PF,
          int t_outputForkingFactor>
template <typename T_dtype>
void InputTransposeChainStreaming<t_instanceID,
                                  t_stage,
                                  t_subStage,
                                  t_forkNumber,
                                  t_L,
                                  t_R,
                                  t_PF,
                                  t_outputForkingFactor>::swap(T_dtype p_in[t_R][t_L / t_R],
                                                               T_dtype p_out[t_R][t_L / t_R]) {
#pragma HLS INLINE // The swap is p_in-lined p_in a data-flow region to expose dataCommutor to become a process.
    // Recursion will create a chain of processes....
    T_dtype temp[t_R][t_L / t_R];
#pragma HLS STREAM variable = temp depth = 8
#pragma HLS RESOURCE variable = temp core = FIFO_LUTRAM
#pragma HLS ARRAY_PARTITION variable = temp complete dim = 1
#pragma HLS data_pack variable = temp
    static const int t_isLargeMemFlag = (((t_PF * t_R) > SSR_FFT_URAM_SELECTION_THRESHHOLD) && SSR_FFT_USE_URAMS);

    streamingDataCommutations<t_instanceID, t_stage, t_subStage, t_forkNumber, t_L, t_R, t_PF, t_isLargeMemFlag>
        commutor;
    commutor.template streamingDataCommutor<T_dtype>(p_in, temp);
    // Instantiate Next State....Recursion
    InputTransposeChainStreaming<t_instanceID, t_stage - 1, t_subStage, t_forkNumber, t_L, t_R, (t_PF * t_R),
                                 t_outputForkingFactor>
        nextStage; // Supply Next t_stage PF

    nextStage.template swap<T_dtype>(temp, p_out);
}

////////////////////////////////////////////////////////////////////////
template <int t_instanceID, int t_subStage, int t_forkNumber, int t_L, int t_R, int t_PF, int t_outputForkingFactor>
template <typename T_dtype>
void InputTransposeChainStreaming<t_instanceID, 1, t_subStage, t_forkNumber, t_L, t_R, t_PF, t_outputForkingFactor>::
    swap(T_dtype p_in[t_R][t_L / t_R], T_dtype p_out[t_R][t_L / t_R]) {
#pragma HLS INLINE

    StreamingDataCommutorFork<t_instanceID, 1, t_subStage, t_outputForkingFactor, t_L, t_R, t_PF, t_outputForkingFactor>
        StreamingDataCommutorFork_obj;
    StreamingDataCommutorFork_obj.template copyForkCommuteAndMerge<T_dtype>(p_in, p_out);
}

#if 1
template <int t_instanceID,
          int t_stage,
          int t_subStage,
          int tp_t_forkNumber,
          int t_L,
          int t_R,
          int t_PF,
          int t_outputForkingFactor>
struct InputTransposeChainStreamingS2S {
    template <typename T_dtype>
    void swap(hls::stream<SuperSampleContainer<t_R, T_dtype> >& p_in,
              hls::stream<SuperSampleContainer<t_R, T_dtype> >& p_out);
};

// These  declarations cover base case for modeling input swap functions were Length of SSR FFT is always integer
// power of Radix R or SSR

////////////////////////////////////////////////////////////////////////
// Specialization for Forking Factor(t_outputForkingFactor) = 1
//                         &&
// template<int tp_stages, int t_instanceID>
template <int t_instanceID, int t_stage, int t_subStage, int t_forkNumber, int t_L, int t_R, int t_PF>
struct InputTransposeChainStreamingS2S<t_instanceID, t_stage, t_subStage, t_forkNumber, t_L, t_R, t_PF, 1> {
    template <typename T_dtype>
    void swap(hls::stream<SuperSampleContainer<t_R, T_dtype> >& p_in,
              hls::stream<SuperSampleContainer<t_R, T_dtype> >& p_out);
};

// template<int t_instanceID>
template <int t_instanceID, int t_subStage, int t_forkNumber, int t_L, int t_R, int t_PF>
struct InputTransposeChainStreamingS2S<t_instanceID, 1, t_subStage, t_forkNumber, t_L, t_R, t_PF, 1> {
    template <typename T_dtype>
    void swap(hls::stream<SuperSampleContainer<t_R, T_dtype> >& p_in,
              hls::stream<SuperSampleContainer<t_R, T_dtype> >& p_out) {
#pragma HLS INLINE
        static const int t_isLargeMemFlag = (((t_PF * t_R) > SSR_FFT_URAM_SELECTION_THRESHHOLD) && SSR_FFT_USE_URAMS);
        DataCommutationsS2Streaming<t_instanceID, 1, t_subStage, t_forkNumber, t_L, t_R, t_PF, t_isLargeMemFlag>
            commutor;
        commutor.template streamingDataCommutor<T_dtype>(p_in, p_out);
    }
};

/////////////////////////////////////////////////////////////////////////
// Struct declaration for the case where forking factor is > 1 and t_stage is 1
// which is actually the terminal case to stop recursion
// template< int t_outputForkingFactor, int t_instanceID>
// template <int t_instanceID, int t_L, int t_R, int t_PF, int t_outputForkingFactor>
template <int t_instanceID, int t_subStage, int t_forkNumber, int t_L, int t_R, int t_PF, int t_outputForkingFactor>

struct InputTransposeChainStreamingS2S<t_instanceID,
                                       1,
                                       t_subStage,
                                       t_forkNumber,
                                       t_L,
                                       t_R,
                                       t_PF,
                                       t_outputForkingFactor> {
    template <typename T_dtype>
    void swap(hls::stream<SuperSampleContainer<t_R, T_dtype> >& p_in,
              hls::stream<SuperSampleContainer<t_R, T_dtype> >& p_out);
};

////////////////////////////////////////////////////////////////////////
// Base Case Implementation where L is integer power of radix ///start forking factor=1
template <int t_instanceID, int t_stage, int t_subStage, int t_forkNumber, int t_L, int t_R, int t_PF>
template <typename T_dtype>
void InputTransposeChainStreamingS2S<t_instanceID, t_stage, t_subStage, t_forkNumber, t_L, t_R, t_PF, 1>::swap(
    hls::stream<SuperSampleContainer<t_R, T_dtype> >& p_in, hls::stream<SuperSampleContainer<t_R, T_dtype> >& p_out) {
#pragma HLS dataflow
    //#pragma HLS INLINE // The swap is p_in-lined p_in a data-flow region to expose dataCommutor to become a process.
    // Recursion will create a chain of processes....
    static const int t_isLargeMemFlag = (((t_PF * t_R) > SSR_FFT_URAM_SELECTION_THRESHHOLD) && SSR_FFT_USE_URAMS);
    hls::stream<SuperSampleContainer<t_R, T_dtype> > temp;
//#pragma HLS data_pack variable = temp
#pragma HLS RESOURCE variable = temp core = FIFO_LUTRAM
#pragma HLS STREAM variable = temp depth = 8

    DataCommutationsS2Streaming<t_instanceID, t_stage, t_subStage, t_forkNumber, t_L, t_R, t_PF, t_isLargeMemFlag>
        commutor;
    commutor.template streamingDataCommutor<T_dtype>(p_in, temp);
    // Instantiate Next State....Recursion
    InputTransposeChainStreamingS2S<t_instanceID, t_stage - 1, t_subStage, t_forkNumber, t_L, t_R, (t_PF * t_R), 1>
        nextStage;
    nextStage.template swap<T_dtype>(temp, p_out); // Supply Next t_stage PF = PF*t_R
}

////////////////////////////////////////////////////////////////////////
template <int t_instanceID, int t_subStage, int t_forkNumber, int t_L, int t_R, int t_PF, int t_outputForkingFactor>

template <typename T_dtype>
void InputTransposeChainStreamingS2S<t_instanceID, 1, t_subStage, t_forkNumber, t_L, t_R, t_PF, t_outputForkingFactor>::
    swap(hls::stream<SuperSampleContainer<t_R, T_dtype> >& p_in,
         hls::stream<SuperSampleContainer<t_R, T_dtype> >& p_out) {
#pragma HLS INLINE off
#pragma HLS dataflow disable_start_propagation

    hls::stream<SuperSampleContainer<t_R / t_outputForkingFactor, T_dtype> > temp[t_outputForkingFactor];
//#pragma HLS data_pack variable = temp
#pragma HLS RESOURCE variable = temp core = FIFO_LUTRAM
//#pragma HLS ARRAY_PARTITION variable = temp complete dim = 1
#pragma HLS STREAM variable = temp depth = 8
    hls::stream<SuperSampleContainer<t_R / t_outputForkingFactor, T_dtype> > temp2[t_outputForkingFactor];
//#pragma HLS data_pack variable = temp2
#pragma HLS RESOURCE variable = temp2 core = FIFO_LUTRAM
//#pragma HLS ARRAY_PARTITION variable = temp complete dim = 1
#pragma HLS STREAM variable = temp2 depth = 8

    forkSuperSampleStream<t_L, t_R, t_forkNumber, t_outputForkingFactor, T_dtype>(p_in, temp);
    StreamingDataCommutorForkS2S<t_instanceID, 1, t_subStage, t_outputForkingFactor, t_L, t_R, t_PF,
                                 t_outputForkingFactor>
        StreamingDataCommutorFork_obj;
    StreamingDataCommutorFork_obj.template forkedCompute<T_dtype>(temp, temp2);

    mergeSuperSampleStream<t_L, t_R, t_forkNumber, t_outputForkingFactor, T_dtype>(temp2, p_out);
}

// Base Case Implementation where L is integer power of radix ///END
////////////////////////////////////////////////////////////////////////
// The generic case where s > 1 and forking factor is > 1 : it structure is same like case when s>1 and f=1
template <int t_instanceID,
          int t_stage,
          int t_subStage,
          int t_forkNumber,
          int t_L,
          int t_R,
          int t_PF,
          int t_outputForkingFactor>
template <typename T_dtype>
void InputTransposeChainStreamingS2S<
    t_instanceID,
    t_stage,
    t_subStage,
    t_forkNumber,
    t_L,
    t_R,
    t_PF,
    t_outputForkingFactor>::swap(hls::stream<SuperSampleContainer<t_R, T_dtype> >& p_in,
                                 hls::stream<SuperSampleContainer<t_R, T_dtype> >& p_out) {
#pragma HLS dataflow disable_start_propagation
    //#pragma HLS INLINE // The swap is p_in-lined p_in a data-flow region to expose dataCommutor to become a process.
    // Recursion will create a chain of processes....
    hls::stream<SuperSampleContainer<t_R, T_dtype> > temp;
//#pragma HLS array_partition variable = temp.superSample dim = 0
//#pragma HLS data_pack variable = temp
#pragma HLS RESOURCE variable = temp core = FIFO_LUTRAM
#pragma HLS STREAM variable = temp depth = 8

    static const int t_isLargeMemFlag = (((t_PF * t_R) > SSR_FFT_URAM_SELECTION_THRESHHOLD) && SSR_FFT_USE_URAMS);

    DataCommutationsS2Streaming<t_instanceID, t_stage, t_subStage, t_forkNumber, t_L, t_R, t_PF, t_isLargeMemFlag>
        commutor;
    commutor.template streamingDataCommutor<T_dtype>(p_in, temp);
    // Instantiate Next State....Recursion
    InputTransposeChainStreamingS2S<t_instanceID, t_stage - 1, t_subStage, t_forkNumber, t_L, t_R, (t_PF * t_R),
                                    t_outputForkingFactor>
        nextStage;
    nextStage.template swap<T_dtype>(temp, p_out); // Supply Next t_stage PF = PF*t_R
}

#endif

} // end namespace fft
} // end namespace dsp
} // end namespace xf

#endif //__HLS_SSR_FFT_STREAMING_TRANSPOSER__
