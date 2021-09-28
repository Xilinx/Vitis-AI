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

// File Name : hls_ssr_fft_streaming_forking_data_commutor.hpp
#ifndef HLS_SSR_FFT_STREAMING_FORKING_DATA_COMMUTOR_H_
#define HLS_SSR_FFT_STREAMING_FORKING_DATA_COMMUTOR_H_

#include "vitis_fft/hls_ssr_fft_fork_merge_utils.hpp"
#include "vitis_fft/hls_ssr_fft_streaming_data_commutor.hpp"
#include "vitis_fft/hls_ssr_fft_fork_merge_utils.hpp"

namespace xf {
namespace dsp {
namespace fft {

//////////////////////////////////////////New class for FORKING-STREAMING data commutor////////////////////////////////
/* StreamingDataCommutorFork : This class will take a [L/R][R] type stream with R sample stream and
 * break it down to R/F streams creating F new streams. Functionally it will take [L/R][R] 2 dimensional array
 * and break it down to F new 2 dimensional arrays of size [L/R][R/F] to be used by F dataflow processes
 */

template <int t_instanceID,
          int t_stage,
          int t_subStage,
          int t_forkNumber,
          int t_L,
          int t_R,
          int t_PF,
          int t_forkingFactor>
struct StreamingDataCommutorFork {
    template <typename T_dtype>
    void copyForkCommuteAndMerge(T_dtype p_in[t_R][t_L / t_R], T_dtype p_out[t_R][t_L / t_R]);
};

// template<int t_stage,int t_instanceID,int t_forkingFactor>
template <int t_instanceID, int t_stage, int t_subStage, int t_L, int t_R, int t_PF, int t_forkingFactor>
// struct StreamingDataCommutorFork<t_stage,t_instanceID,1,t_forkingFactor>
struct StreamingDataCommutorFork<t_instanceID, t_stage, t_subStage, 1, t_L, t_R, t_PF, t_forkingFactor> {
    template <typename T_dtype>
    void copyForkCommuteAndMerge(T_dtype p_in[t_R][t_L / t_R], T_dtype p_out[t_R][t_L / t_R]);
};

/* Forking Function: Copies data t_R/Forking Factor Buffers for forking p_out the output
 * The input is [t_L][t_R] array and output is a also [t_L][t_R] array , but internally
 * The input array is split into smaller arrays like : [t_L][t_R/forking_factor] to create
 * t_R/forking factor input arrays, each of these arrays is assumed to be a seperate function
 * and finally in hardware every such function will map to a seperate process. Essentially it will
 * create a fork, once process feeding multiple processes;
 */

template <int t_instanceID,
          int t_stage,
          int t_subStage,
          int t_forkNumber,
          int t_L,
          int t_R,
          int t_PF,
          int t_forkingFactor>
template <typename T_dtype>
void StreamingDataCommutorFork<t_instanceID, t_stage, t_subStage, t_forkNumber, t_L, t_R, t_PF, t_forkingFactor>::
    copyForkCommuteAndMerge(T_dtype p_in[t_R][t_L / t_R], T_dtype p_out[t_R][t_L / t_R]) {
#pragma HLS INLINE
    T_dtype localFactoredInputBuff[t_R / t_forkingFactor][t_L / t_R];
#pragma HLS DATA_PACK variable = localFactoredInputBuff
#pragma HLS STREAM variable = localFactoredInputBuff depth = 8
#pragma HLS RESOURCE variable = localFactoredInputBuff core = FIFO_LUTRAM

#ifdef SSR_FFT_PARTITION_INTERFACE_ARRAYS
#pragma HLS ARRAY_PARTITION variable = localFactoredInputBuff complete dim = 1
#else
#pragma HLS ARRAY_RESHAPE variable = localFactoredInputBuff complete dim = 1
#endif
    T_dtype localFactoredOutputBuff[t_R / t_forkingFactor][t_L / t_R];
#pragma HLS DATA_PACK variable = localFactoredOutputBuff
#pragma HLS STREAM variable = localFactoredOutputBuff depth = 8
#pragma HLS RESOURCE variable = localFactoredOutputBuff core = FIFO_LUTRAM

//#pragma HLS ARRAY_RESHAPE variable=localFactoredOutputBuff complete dim=1
#ifdef SSR_FFT_PARTITION_INTERFACE_ARRAYS
#pragma HLS ARRAY_PARTITION variable = localFactoredOutputBuff complete dim = 1
#else
#pragma HLS ARRAY_RESHAPE variable = localFactoredOutputBuff complete dim = 1
#endif

    static const int t_isLargeMemFlag =
        ((t_PF * (t_R / t_forkingFactor) > SSR_FFT_URAM_SELECTION_THRESHHOLD) && SSR_FFT_USE_URAMS);

    copyToLocalBuff<t_L, t_R, t_forkNumber, t_forkingFactor, T_dtype>(p_in, localFactoredInputBuff);
    streamingDataCommutations<t_instanceID, t_stage, t_subStage, t_forkNumber, t_L / t_forkingFactor,
                              t_R / t_forkingFactor, t_PF, t_isLargeMemFlag>
        commutor;

    commutor.template streamingDataCommutor<T_dtype>(localFactoredInputBuff, localFactoredOutputBuff);

    copyFromLocalBuffToOuput<t_L, t_R, t_forkNumber, t_forkingFactor, T_dtype>(localFactoredOutputBuff, p_out);

    StreamingDataCommutorFork<t_instanceID, t_stage, t_subStage - 1, t_forkNumber - 1, t_L, t_R, t_PF, t_forkingFactor>
        StreamingDataCommutorFork_obj;
    StreamingDataCommutorFork_obj.template copyForkCommuteAndMerge<T_dtype>(p_in, p_out);
}

template <int t_instanceID, int t_stage, int t_subStage, int t_L, int t_R, int t_PF, int t_forkingFactor>
template <typename T_dtype>
void StreamingDataCommutorFork<t_instanceID, t_stage, t_subStage, 1, t_L, t_R, t_PF, t_forkingFactor>::
    copyForkCommuteAndMerge(T_dtype p_in[t_R][t_L / t_R], T_dtype p_out[t_R][t_L / t_R]) {
#pragma HLS INLINE
    T_dtype localFactoredInputBuff[t_R / t_forkingFactor][t_L / t_R];
#pragma HLS DATA_PACK variable = localFactoredInputBuff
#pragma HLS STREAM variable = localFactoredInputBuff depth = 8
#pragma HLS RESOURCE variable = localFactoredInputBuff core = FIFO_LUTRAM

#ifdef SSR_FFT_PARTITION_INTERFACE_ARRAYS
#pragma HLS ARRAY_PARTITION variable = localFactoredInputBuff complete dim = 1
#else
#pragma HLS ARRAY_RESHAPE variable = localFactoredInputBuff complete dim = 1
#endif

    T_dtype localFactoredOutputBuff[t_R / t_forkingFactor][t_L / t_R];
#pragma HLS DATA_PACK variable = localFactoredOutputBuff
#pragma HLS STREAM variable = localFactoredOutputBuff depth = 8
#pragma HLS RESOURCE variable = localFactoredOutputBuff core = FIFO_LUTRAM
//#pragma HLS ARRAY_RESHAPE variable=localFactoredOutputBuff complete dim=1
#ifdef SSR_FFT_PARTITION_INTERFACE_ARRAYS
#pragma HLS ARRAY_PARTITION variable = localFactoredOutputBuff complete dim = 1
#else
#pragma HLS ARRAY_RESHAPE variable = localFactoredOutputBuff complete dim = 1
#endif
    static const int t_isLargeMemFlag =
        ((t_PF * (t_R / t_forkingFactor) > SSR_FFT_URAM_SELECTION_THRESHHOLD) && SSR_FFT_USE_URAMS);

    copyToLocalBuff<t_L, t_R, 1, t_forkingFactor, T_dtype>(p_in, localFactoredInputBuff);
    streamingDataCommutations<t_instanceID, t_stage, t_subStage, 1, t_L / t_forkingFactor, t_R / t_forkingFactor, t_PF,
                              t_isLargeMemFlag>
        commutor;
    commutor.template streamingDataCommutor<T_dtype>(localFactoredInputBuff, localFactoredOutputBuff);
    copyFromLocalBuffToOuput<t_L, t_R, 1, t_forkingFactor, T_dtype>(localFactoredOutputBuff, p_out);
}

/* DataCommutorFork : This class will take a [L/R][R] type stream with R sample stream and
 * break it down to R/F streams creating F new streams. Functionally it will take [L/R][R] 2 dimensional array
 * and break it down to F new 2 dimensional arrays of size [L/R][R/F] to be used by F dataflow processes
 */
template <int t_instanceID,
          int t_stage,
          int t_subStage,
          int t_forkNumber,
          int t_L,
          int t_R,
          int t_PF,
          int t_forkingFactor>
struct StreamingDataCommutorForkNonInvertOut {
    template <typename T_dtype>
    void copyForkCommuteAndMergeNonInvert(T_dtype p_in[t_R][t_L / t_R], T_dtype p_out[t_R][t_L / t_R]);
};

// Base case for t_forkNumber=1
template <int t_instanceID, int t_stage, int t_subStage, int t_L, int t_R, int t_PF, int t_forkingFactor>
struct StreamingDataCommutorForkNonInvertOut<t_instanceID, t_stage, t_subStage, 1, t_L, t_R, t_PF, t_forkingFactor> {
    template <typename T_dtype>
    void copyForkCommuteAndMergeNonInvert(T_dtype p_in[t_R][t_L / t_R], T_dtype p_out[t_R][t_L / t_R]);
};

/* Forking Function: Copies data t_R/Forking Factor Buffers for forking p_out the output
 * The input is [t_L][t_R] array and output is a also [t_L][t_R] array , but internally
 * The input array is split into smaller arrays like : [t_L][t_R/forking_factor] to create
 * t_R/forking factor input arrays, each of these arrays is assumed to be a seperate function
 * and finally in hardware every such function will map to a seperate process. Essentially it will
 * create a fork, once process feeding multiple processes;
 */
template <int t_instanceID,
          int t_stage,
          int t_subStage,
          int t_forkNumber,
          int t_L,
          int t_R,
          int t_PF,
          int t_forkingFactor>
template <typename T_dtype>

void StreamingDataCommutorForkNonInvertOut<
    t_instanceID,
    t_stage,
    t_subStage,
    t_forkNumber,
    t_L,
    t_R,
    t_PF,
    t_forkingFactor>::copyForkCommuteAndMergeNonInvert(T_dtype p_in[t_R][t_L / t_R], T_dtype p_out[t_R][t_L / t_R]) {
#pragma HLS INLINE
    T_dtype localFactoredInputBuff[t_R / t_forkingFactor][t_L / t_R];
#pragma HLS DATA_PACK variable = localFactoredInputBuff
#pragma HLS STREAM variable = localFactoredInputBuff depth = 8
#pragma HLS RESOURCE variable = localFactoredInputBuff core = FIFO_LUTRAM
#ifdef SSR_FFT_PARTITION_INTERFACE_ARRAYS
#pragma HLS ARRAY_PARTITION variable = localFactoredInputBuff complete dim = 1
#else
#pragma HLS ARRAY_RESHAPE variable = localFactoredInputBuff complete dim = 1
#endif
    T_dtype localFactoredOutputBuff[t_R / t_forkingFactor][t_L / t_R];
#pragma HLS DATA_PACK variable = localFactoredOutputBuff
#pragma HLS STREAM variable = localFactoredOutputBuff depth = 8
#pragma HLS RESOURCE variable = localFactoredOutputBuff core = FIFO_LUTRAM
#ifdef SSR_FFT_PARTITION_INTERFACE_ARRAYS
#pragma HLS ARRAY_PARTITION variable = localFactoredOutputBuff complete dim = 1
#else
#pragma HLS ARRAY_RESHAPE variable = localFactoredOutputBuff complete dim = 1
#endif

    copyToLocalBuff<t_L, t_R, t_forkNumber, t_forkingFactor, T_dtype>(p_in, localFactoredInputBuff);
    static const int t_isLargeMemFlag =
        ((t_PF * (t_R / t_forkingFactor) > SSR_FFT_URAM_SELECTION_THRESHHOLD) && SSR_FFT_USE_URAMS);
    streamingDataCommutations<t_instanceID, t_stage, t_subStage, t_forkNumber, t_L / t_forkingFactor,
                              t_R / t_forkingFactor, t_PF, t_isLargeMemFlag>
        commutor;

    commutor.template streamingDataCommutor<T_dtype>(localFactoredInputBuff, localFactoredOutputBuff);

    copyBuffToOutNonInvert<t_L, t_R, t_forkNumber, t_forkingFactor, T_dtype>(localFactoredOutputBuff, p_out);
    StreamingDataCommutorForkNonInvertOut<t_instanceID, t_stage, t_subStage - 1, t_forkNumber - 1, t_L, t_R, t_PF,
                                          t_forkingFactor>
        StreamingDataCommutorFork_obj_NI;

    StreamingDataCommutorFork_obj_NI.template copyForkCommuteAndMergeNonInvert<T_dtype>(p_in, p_out);
}

// copyForkCommuteAndMerge base case specialization for fork number = 1, terminates forking/recursion
template <int t_instanceID, int t_stage, int t_subStage, int t_L, int t_R, int t_PF, int t_forkingFactor>
template <typename T_dtype>
void StreamingDataCommutorForkNonInvertOut<t_instanceID, t_stage, t_subStage, 1, t_L, t_R, t_PF, t_forkingFactor>::
    copyForkCommuteAndMergeNonInvert(T_dtype p_in[t_R][t_L / t_R], T_dtype p_out[t_R][t_L / t_R]) {
#pragma HLS INLINE

    T_dtype localFactoredInputBuff[t_R / t_forkingFactor][t_L / t_R];
#pragma HLS DATA_PACK variable = localFactoredInputBuff
#pragma HLS STREAM variable = localFactoredInputBuff depth = 8
#pragma HLS RESOURCE variable = localFactoredInputBuff core = FIFO_LUTRAM
#ifdef SSR_FFT_PARTITION_INTERFACE_ARRAYS
#pragma HLS ARRAY_PARTITION variable = localFactoredInputBuff complete dim = 1
#else
#pragma HLS ARRAY_RESHAPE variable = localFactoredInputBuff complete dim = 1
#endif

    T_dtype localFactoredOutputBuff[t_R / t_forkingFactor][t_L / t_R];
#pragma HLS DATA_PACK variable = localFactoredOutputBuff
#pragma HLS STREAM variable = localFactoredOutputBuff depth = 8
#pragma HLS RESOURCE variable = localFactoredOutputBuff core = FIFO_LUTRAM

#pragma HLS ARRAY_RESHAPE variable = localFactoredOutputBuff complete dim = 1
#ifdef SSR_FFT_PARTITION_INTERFACE_ARRAYS
#pragma HLS ARRAY_PARTITION variable = localFactoredOutputBuff complete dim = 1
#else
#pragma HLS ARRAY_RESHAPE variable = localFactoredOutputBuff complete dim = 1
#endif
    static const int t_isLargeMemFlag =
        ((t_PF * (t_R / t_forkingFactor) > SSR_FFT_URAM_SELECTION_THRESHHOLD) && SSR_FFT_USE_URAMS);
    copyToLocalBuff<t_L, t_R, 1, t_forkingFactor, T_dtype>(p_in, localFactoredInputBuff);
    streamingDataCommutations<t_instanceID, t_stage, t_subStage, 1, t_L / t_forkingFactor, t_R / t_forkingFactor, t_PF,
                              t_isLargeMemFlag>
        commutor;

    commutor.template streamingDataCommutor<T_dtype>(localFactoredInputBuff, localFactoredOutputBuff);
    copyBuffToOutNonInvert<t_L, t_R, 1, t_forkingFactor, T_dtype>(localFactoredOutputBuff, p_out);
}
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

} // end namespace fft
} // end namespace dsp
} // end namespace xf

/////////////////////////////////////// New Class Purely Streaming : without conversion from array to stream

namespace xf {
namespace dsp {
namespace fft {

//////////////////////////////////////////New class for FORKING-STREAMING data commutor////////////////////////////////
/* StreamingDataCommutorForkS2S : This class will take a [L/R][R] type stream with R sample stream and
 * break it down to R/F streams creating F new streams. Functionally it will take [L/R][R] 2 dimensional array
 * and break it down to F new 2 dimensional arrays of size [L/R][R/F] to be used by F dataflow processes
 */
template <int t_instanceID,
          int t_stage,
          int t_subStage,
          int t_forkNumber,
          int t_L,
          int t_R,
          int t_PF,
          int t_forkingFactor>
struct StreamingDataCommutorForkS2S {
    template <typename T_dtype>
    void forkedCompute(hls::stream<SuperSampleContainer<t_R / t_forkingFactor, T_dtype> > p_in[t_forkingFactor],
                       hls::stream<SuperSampleContainer<t_R / t_forkingFactor, T_dtype> > p_out[t_forkingFactor]) {
//#pragma HLS INLINE
#pragma HLS dataflow

        static const int t_isLargeMemFlag =
            ((t_PF * (t_R / t_forkingFactor) > SSR_FFT_URAM_SELECTION_THRESHHOLD) && SSR_FFT_USE_URAMS);

        DataCommutationsS2Streaming<t_instanceID, t_stage, t_subStage, t_forkNumber, t_L / t_forkingFactor,
                                    t_R / t_forkingFactor, t_PF, t_isLargeMemFlag>
            commutor;
        commutor.template streamingDataCommutor<T_dtype>(p_in[t_forkNumber - 1], p_out[t_forkNumber - 1]);

        StreamingDataCommutorForkS2S<t_instanceID, t_stage, t_subStage - 1, t_forkNumber - 1, t_L, t_R, t_PF,
                                     t_forkingFactor>
            StreamingDataCommutorFork_obj;
        StreamingDataCommutorFork_obj.template forkedCompute<T_dtype>(p_in, p_out);
    }
};

template <int t_instanceID, int t_stage, int t_subStage, int t_L, int t_R, int t_PF, int t_forkingFactor>
struct StreamingDataCommutorForkS2S<t_instanceID, t_stage, t_subStage, 1, t_L, t_R, t_PF, t_forkingFactor> {
    template <typename T_dtype>
    void forkedCompute(hls::stream<SuperSampleContainer<t_R / t_forkingFactor, T_dtype> > p_in[t_forkingFactor],
                       hls::stream<SuperSampleContainer<t_R / t_forkingFactor, T_dtype> > p_out[t_forkingFactor]) {
#pragma HLS dataflow
        //#pragma HLS INLINE
        static const int t_isLargeMemFlag =
            ((t_PF * (t_R / t_forkingFactor) > SSR_FFT_URAM_SELECTION_THRESHHOLD) && SSR_FFT_USE_URAMS);

        DataCommutationsS2Streaming<t_instanceID, t_stage, t_subStage, 1, t_L / t_forkingFactor, t_R / t_forkingFactor,
                                    t_PF, t_isLargeMemFlag>
            commutor;
        commutor.template streamingDataCommutor<T_dtype>(p_in[1 - 1], p_out[1 - 1]);
    }
};

} // end namespace fft
} // end namespace dsp
} // end namespace xf

#endif // HLS_SSR_FFT_STREAMING_FORKING_DATA_COMMUTOR_H_
