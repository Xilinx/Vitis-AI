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

// filename: hls_ssr_fft_output_reorder.hpp
#ifndef HLS_SSR_FFT_OUTPUT_REORDER_h_
#define HLS_SSR_FFT_OUTPUT_REORDER_h_
#include <iostream>
#include <hls_stream.h>

#include "vitis_fft/hls_ssr_fft_criss_cross_multiplexer.hpp"
#include "vitis_fft/hls_ssr_fft_rotate_criss_cross_multiplexer.hpp"
#include "vitis_fft/hls_ssr_fft_super_sample.hpp"

/*
 =========================================================================================
 -_- -_-
 -_- -_-
 -_- -_-
 -_- -_-
 -_- -_-
 -_- -_-
 -_-

 The digitReversedDataReOrder  defined in other file function is used to
 perform data re ordering at the output of final stage in SSR FFT.
 The data produced by final stage is
 shuffled in SSR dimension ( R or SSR streams coming from SSR FFT last stage)
 and also it needs re ordering in time dimension. The re ordering is done
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
 deals with it through different specializations. is defined
 in this file

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

namespace xf {
namespace dsp {
namespace fft {

template <int t_L, int t_R, typename T_dtype, typename T_dtype2>
void crissCrossAndTimeDigitSwap(T_dtype p_inData[t_R][t_L / t_R], T_dtype2 p_pingPongBuffOut[t_R][t_L / t_R]) {
#pragma HLS INLINE off

    unsigned int inputTimeIndex;
    unsigned int outputTimeIndex;
    const unsigned int F = ((t_R * t_R) / t_L) > 1 ? ((t_R * t_R) / t_L) : 1;

///////////////////////////////////// Shuffle the Data and Write to PingPong Buffer ////////////////////////////////////
TIME_LOOP:
    for (unsigned int t = 0; t < t_L / t_R; t++) {
#pragma HLS PIPELINE II = 1 // rewind
        inputTimeIndex = t;
        outputTimeIndex = digitReversalFractionIsLSB<t_L / t_R, t_R>(inputTimeIndex);
        T_dtype sampleShufflingBuffer[t_R];
#pragma HLS ARRAY_PARTITION variable = sampleShufflingBuffer complete dim = 1
#pragma HLS data_pack variable = sampleShufflingBuffer

    SSR_READ_LOOP:
        for (unsigned int r = 0; r < t_R; r++) {
#pragma HLS UNROLL
            sampleShufflingBuffer[(r + outputTimeIndex) & (ssrFFTLog2BitwiseAndModMask<t_R>::val)] =
                p_inData[r][inputTimeIndex];
        }
    SSR_WRITE_LOOP:
        for (unsigned int r = 0; r < t_R; r++) {
#pragma HLS UNROLL
            p_pingPongBuffOut[r][outputTimeIndex] = sampleShufflingBuffer[r]; // digit reversal write to the memory
        }
    }
}
template <int t_L, int t_R, typename T_dtype, typename T_dtype2>
void crissCrossAndTimeDigitSwap(hls::stream<T_dtype> p_inData[t_R], T_dtype2 p_pingPongBuffOut[t_R][t_L / t_R]) {
#pragma HLS INLINE off

    unsigned int inputTimeIndex;
    unsigned int outputTimeIndex;
    const unsigned int F = ((t_R * t_R) / t_L) > 1 ? ((t_R * t_R) / t_L) : 1;

///////////////////////////////////// Shuffle the Data and Write to PingPong Buffer ////////////////////////////////////
TIME_LOOP:
    for (unsigned int t = 0; t < t_L / t_R; t++) {
#pragma HLS PIPELINE II = 1 // rewind
        inputTimeIndex = t;
        outputTimeIndex = digitReversalFractionIsLSB<t_L / t_R, t_R>(inputTimeIndex);
        T_dtype sampleShufflingBuffer[t_R];
#pragma HLS ARRAY_PARTITION variable = sampleShufflingBuffer complete dim = 1
    //#pragma HLS data_pack variable = sampleShufflingBuffer

    SSR_READ_LOOP:
        for (unsigned int r = 0; r < t_R; r++) {
#pragma HLS UNROLL
            sampleShufflingBuffer[(r + outputTimeIndex) & (ssrFFTLog2BitwiseAndModMask<t_R>::val)] = p_inData[r].read();
            //                p_inData[r][inputTimeIndex];
        }
    SSR_WRITE_LOOP:
        for (unsigned int r = 0; r < t_R; r++) {
#pragma HLS UNROLL
            p_pingPongBuffOut[r][outputTimeIndex] = sampleShufflingBuffer[r]; // digit reversal write to the memory
        }
    }
}
template <int t_L, int t_R, typename T_dtype, typename T_dtype2>
void crissCrossAndTimeDigitSwapSpecial(T_dtype p_inData[t_R][t_L / t_R], T_dtype2 p_pingPongBuffOut[t_R][t_L / t_R]) {
#pragma HLS INLINE off

    unsigned int inputTimeIndex;
    unsigned int outputTimeIndex;
    const unsigned int F = ((t_R * t_R) / t_L) > 1 ? ((t_R * t_R) / t_L) : 1;
///////////////////////////////////// Shuffle the Data and Write to Buffer ////////////////////////////////////
TIME_LOOP:
    for (unsigned int t = 0; t < t_L / t_R; t++) {
#pragma HLS PIPELINE II = 1 rewind
        inputTimeIndex = t;
        outputTimeIndex = digitReversal_m<t_L / t_R, t_R>(inputTimeIndex);
        // replaced//unsigned int InputShuffleAmount = (t*F) % t_R;//   minOf_R_and_L_over_R; // can be converted to
        // masking
        unsigned int InputShuffleAmount =
            (t << (ssrFFTLog2<F>::val)) &
            (ssrFFTLog2BitwiseAndModMask<t_R>::val); //   minOf_R_and_L_over_R; // can be converted to masking
        T_dtype sampleShufflingBuffer[t_R];
#pragma HLS ARRAY_PARTITION variable = sampleShufflingBuffer complete dim = 1
#pragma HLS data_pack variable = sampleShufflingBuffer

    SSR_READ_LOOP:
        for (unsigned int r = 0; r < t_R; r++) {
#pragma HLS UNROLL
            // replaced//sampleShufflingBuffer[(r + InputShuffleAmount) % t_R] = p_inData[r][inputTimeIndex];
            sampleShufflingBuffer[(r + InputShuffleAmount) & (ssrFFTLog2BitwiseAndModMask<t_R>::val)] =
                p_inData[r][inputTimeIndex];
            // sampleShufflingBuffer[(outputTimeIndex) % t_R] = p_inData[inputTimeIndex][r];
        }
    SSR_WRITE_LOOP:
        for (unsigned int r = 0; r < t_R; r++) {
#pragma HLS UNROLL
            p_pingPongBuffOut[r][outputTimeIndex] = sampleShufflingBuffer[r]; // digit reversal write to the memory
        }
    }
}
template <int t_L, int t_R, typename T_dtype, typename T_dtype2>
void crissCrossAndTimeDigitSwapSpecial(hls::stream<T_dtype> p_inData[t_R], T_dtype2 p_pingPongBuffOut[t_R][t_L / t_R]) {
#pragma HLS INLINE off

    unsigned int inputTimeIndex;
    unsigned int outputTimeIndex;
    const unsigned int F = ((t_R * t_R) / t_L) > 1 ? ((t_R * t_R) / t_L) : 1;
///////////////////////////////////// Shuffle the Data and Write to Buffer ////////////////////////////////////
TIME_LOOP:
    for (unsigned int t = 0; t < t_L / t_R; t++) {
#pragma HLS PIPELINE II = 1 // rewind enable_flush
        inputTimeIndex = t;
        outputTimeIndex = digitReversal_m<t_L / t_R, t_R>(inputTimeIndex);
        // replaced//unsigned int InputShuffleAmount = (t*F) % t_R;//   minOf_R_and_L_over_R; // can be converted to
        // masking
        unsigned int InputShuffleAmount =
            (t << (ssrFFTLog2<F>::val)) &
            (ssrFFTLog2BitwiseAndModMask<t_R>::val); //   minOf_R_and_L_over_R; // can be converted to masking
        T_dtype sampleShufflingBuffer[t_R];
#pragma HLS ARRAY_PARTITION variable = sampleShufflingBuffer complete dim = 1
#pragma HLS data_pack variable = sampleShufflingBuffer

    SSR_READ_LOOP:
        for (unsigned int r = 0; r < t_R; r++) {
#pragma HLS UNROLL
            //#pragma HLS latency max = 1 min = 1
            // replaced//sampleShufflingBuffer[(r + InputShuffleAmount) % t_R] = p_inData[r][inputTimeIndex];
            sampleShufflingBuffer[(r + InputShuffleAmount) & (ssrFFTLog2BitwiseAndModMask<t_R>::val)] =
                p_inData[r].read();
            //                p_inData[r][inputTimeIndex];
            // sampleShufflingBuffer[(outputTimeIndex) % t_R] = p_inData[inputTimeIndex][r];
        }
    SSR_WRITE_LOOP:
        for (unsigned int r = 0; r < t_R; r++) {
#pragma HLS UNROLL
            p_pingPongBuffOut[r][outputTimeIndex] = sampleShufflingBuffer[r]; // digit reversal write to the memory
        }
    }
}

template <int t_L, int t_R, typename T_dtype, typename T_dtype2>
void diagonalWrapIndexReadFromPIPOAndStreamOut(T_dtype p_pingPongBuffOut[t_R][t_L / t_R],
                                               T_dtype2 p_outData[t_R][t_L / t_R]) {
#pragma HLS INLINE off

    const unsigned int F = ((t_R * t_R) / t_L) > 1 ? ((t_R * t_R) / t_L) : 1;
    const unsigned int outputShuffleAmount = ssrFFTLog2<F>::val;
    unsigned int pingPongTimeIndex;
    unsigned int pingPongSuperSampleIndex;
TIME_LOOP:
    for (unsigned int t = 0; t < t_L / t_R; t++) {
        T_dtype2 temp[t_R];
#pragma HLS ARRAY_PARTITION variable = temp complete dim = 1
#pragma HLS data_pack variable = temp

#pragma HLS PIPELINE II = 1 // rewind      // This loop has a rewind issue.

        // replced// int ssrDimensionAddressOffset = ( t /(  t_L/(t_R*t_R)) ) % t_R;
        int ssrDimensionAddressOffset =
            (t >> (ssrFFTLog2<t_L / (t_R * t_R)>::val)) & (ssrFFTLog2BitwiseAndModMask<t_R>::val);
        int timeIndexAddressOffset = (t & (ssrFFTLog2BitwiseAndModMask<(t_L / (t_R * t_R))>::val))
                                     << (ssrFFTLog2<t_R>::val);
    // int timeIndexAddressOffset = (t % (t_L/(t_R*t_R)))*t_R;
    SSR_LOOP0:
        for (unsigned int r = 0; r < t_R; r++) {
#pragma HLS UNROLL
            /*This expression is replaced below*/ // int pingPongSuperSampleIndex = ((stage-1) + r) % t_R;
            int pingPongSuperSampleIndex = ((ssrDimensionAddressOffset) + r) & ssrFFTLog2BitwiseAndModMask<t_R>::val;
            int pingPongTimeIndex = ((t_R + r - ssrDimensionAddressOffset) % t_R) + timeIndexAddressOffset;
            temp[(t_R + r - ssrDimensionAddressOffset) % t_R] = p_pingPongBuffOut[r][pingPongTimeIndex];
        }
    SSR_LOOP:
        for (unsigned int r = 0; r < t_R; r++) {
#pragma HLS UNROLL
            p_outData[r][t] = temp[r];
        }
    }
}
template <int t_L, int t_R, typename T_dtype, typename T_dtype2>
void diagonalWrapIndexReadFromPIPOAndStreamOut(T_dtype p_pingPongBuffOut[t_R][t_L / t_R],
                                               hls::stream<T_dtype2> p_outData[t_R]) {
#pragma HLS INLINE off

    const unsigned int F = ((t_R * t_R) / t_L) > 1 ? ((t_R * t_R) / t_L) : 1;
    const unsigned int outputShuffleAmount = ssrFFTLog2<F>::val;
    unsigned int pingPongTimeIndex;
    unsigned int pingPongSuperSampleIndex;
TIME_LOOP:
    for (unsigned int t = 0; t < t_L / t_R; t++) {
        T_dtype2 temp[t_R];
#pragma HLS ARRAY_PARTITION variable = temp complete dim = 1
//#pragma HLS data_pack variable = temp

#pragma HLS PIPELINE II = 1 // rewind      // This loop has a rewind issue.

        // replced// int ssrDimensionAddressOffset = ( t /(  t_L/(t_R*t_R)) ) % t_R;
        int ssrDimensionAddressOffset =
            (t >> (ssrFFTLog2<t_L / (t_R * t_R)>::val)) & (ssrFFTLog2BitwiseAndModMask<t_R>::val);
        int timeIndexAddressOffset = (t & (ssrFFTLog2BitwiseAndModMask<(t_L / (t_R * t_R))>::val))
                                     << (ssrFFTLog2<t_R>::val);
    // int timeIndexAddressOffset = (t % (t_L/(t_R*t_R)))*t_R;
    SSR_LOOP0:
        for (unsigned int r = 0; r < t_R; r++) {
#pragma HLS UNROLL
            /*This expression is replaced below*/ // int pingPongSuperSampleIndex = ((stage-1) + r) % t_R;
            int pingPongSuperSampleIndex = ((ssrDimensionAddressOffset) + r) & ssrFFTLog2BitwiseAndModMask<t_R>::val;
            int pingPongTimeIndex = ((t_R + r - ssrDimensionAddressOffset) % t_R) + timeIndexAddressOffset;
            temp[(t_R + r - ssrDimensionAddressOffset) % t_R] = p_pingPongBuffOut[r][pingPongTimeIndex];
        }
    SSR_LOOP:
        for (unsigned int r = 0; r < t_R; r++) {
#pragma HLS UNROLL
            p_outData[r].write(temp[r]);
        }
    }
}

template <int t_L, int t_R, typename T_dtype, typename T_dtype2>
void diagonalWrapIndexReadFromPIPOAndStreamOutSpecial(T_dtype p_pingPongBuffOut[t_R][t_L / t_R],
                                                      T_dtype2 p_outData[t_R][t_L / t_R]) {
#pragma HLS INLINE off
    ///////////////////////////////////// Shuffle the Data and Write to Buffer ////////////////////////////////////
    const unsigned int F = ((t_R * t_R) / t_L) > 1 ? ((t_R * t_R) / t_L) : 1; // t_R is always greater than F
    unsigned int pingPongTimeIndex;
    unsigned int pingPongSuperSampleIndex;
    const unsigned int outputShuffleAmount = ssrFFTLog2<F>::val;
TIME_LOOP:
    for (unsigned int t = 0; t < t_L / t_R; t++) {
#pragma HLS PIPELINE II = 1 // rewind    // This loop has a reqind issue
        T_dtype2 temp[t_R];
#pragma HLS ARRAY_PARTITION variable = temp complete dim = 1
#pragma HLS data_pack variable = temp

        // replaced//int ssrDimensionAddressOffset =( F*t ) % t_R;
        int ssrDimensionAddressOffset = (t << (ssrFFTLog2<F>::val)) & (ssrFFTLog2BitwiseAndModMask<t_R>::val);

        // replaced//	int timeIndexAddressOffset = ( t_R * (t / t_R) ) % (t_L/t_R);
        int timeIndexAddressOffset =
            ((t >> (ssrFFTLog2<t_R>::val)) << (ssrFFTLog2<t_R>::val)) & (ssrFFTLog2BitwiseAndModMask<t_L / t_R>::val);
    SSR_LOOP0:
        for (unsigned int r = 0; r < t_R; r++) {
#pragma HLS UNROLL
            ap_uint<ssrFFTLog2<t_R>::val> rotated_r = r;
            rotated_r.lrotate(outputShuffleAmount);
            // replaced//int pingPongSuperSampleIndex = ((stage-1) + rotated_r) % t_R;
            int pingPongSuperSampleIndex =
                r; // ((ssrDimensionAddressOffset) + rotated_r) & (ssrFFTLog2BitwiseAndModMask<t_R>::val);

            // replaced//int pingPongTimeIndex=(r+timeIndexAddressOffset)%(t_L/t_R);
            ap_uint<ssrFFTLog2<t_R>::val> rev_rotated_r = (t_R + r - ssrDimensionAddressOffset) % t_R;
            rev_rotated_r.rrotate(outputShuffleAmount);
            int pingPongTimeIndex =
                (rev_rotated_r + timeIndexAddressOffset) & (ssrFFTLog2BitwiseAndModMask<t_L / t_R>::val);
            temp[rev_rotated_r] = p_pingPongBuffOut[r][pingPongTimeIndex];
            // CHECK_COVEARAGE;
        }
    SSR_LOOP:
        for (unsigned int r = 0; r < t_R; r++) {
#pragma HLS UNROLL
            p_outData[r][t] = temp[r];
        }
    }
}
template <int t_L, int t_R, typename T_dtype, typename T_dtype2>
void diagonalWrapIndexReadFromPIPOAndStreamOutSpecial(T_dtype p_pingPongBuffOut[t_R][t_L / t_R],
                                                      hls::stream<T_dtype2> p_outData[t_R]) {
#pragma HLS INLINE off
    ///////////////////////////////////// Shuffle the Data and Write to Buffer ////////////////////////////////////
    const unsigned int F = ((t_R * t_R) / t_L) > 1 ? ((t_R * t_R) / t_L) : 1; // t_R is always greater than F
    unsigned int pingPongTimeIndex;
    unsigned int pingPongSuperSampleIndex;
    const unsigned int outputShuffleAmount = ssrFFTLog2<F>::val;
TIME_LOOP:
    for (unsigned int t = 0; t < t_L / t_R; t++) {
#pragma HLS PIPELINE II = 1 // rewind    // This loop has a reqind issue
        T_dtype2 temp[t_R];
#pragma HLS ARRAY_PARTITION variable = temp complete dim = 1
#pragma HLS data_pack variable = temp

        // replaced//int ssrDimensionAddressOffset =( F*t ) % t_R;
        int ssrDimensionAddressOffset = (t << (ssrFFTLog2<F>::val)) & (ssrFFTLog2BitwiseAndModMask<t_R>::val);

        // replaced//	int timeIndexAddressOffset = ( t_R * (t / t_R) ) % (t_L/t_R);
        int timeIndexAddressOffset =
            ((t >> (ssrFFTLog2<t_R>::val)) << (ssrFFTLog2<t_R>::val)) & (ssrFFTLog2BitwiseAndModMask<t_L / t_R>::val);
    SSR_LOOP0:
        for (unsigned int r = 0; r < t_R; r++) {
#pragma HLS UNROLL
            ap_uint<ssrFFTLog2<t_R>::val> rotated_r = r;
            rotated_r.lrotate(outputShuffleAmount);
            // replaced//int pingPongSuperSampleIndex = ((stage-1) + rotated_r) % t_R;
            int pingPongSuperSampleIndex =
                r; // ((ssrDimensionAddressOffset) + rotated_r) & (ssrFFTLog2BitwiseAndModMask<t_R>::val);

            // replaced//int pingPongTimeIndex=(r+timeIndexAddressOffset)%(t_L/t_R);
            ap_uint<ssrFFTLog2<t_R>::val> rev_rotated_r = (t_R + r - ssrDimensionAddressOffset) % t_R;
            rev_rotated_r.rrotate(outputShuffleAmount);
            int pingPongTimeIndex =
                (rev_rotated_r + timeIndexAddressOffset) & (ssrFFTLog2BitwiseAndModMask<t_L / t_R>::val);
            temp[rev_rotated_r] = p_pingPongBuffOut[r][pingPongTimeIndex];
            // CHECK_COVEARAGE;
        }
    SSR_LOOP:
        for (unsigned int r = 0; r < t_R; r++) {
#pragma HLS UNROLL
            p_outData[r].write(temp[r]);
        }
    }
}

//// OutputDataReOrder<skew>::digitReversal2Phase : is used for output data re-ordering for the case when the length of
/// the SSR FFT is not integer power of length, It has also a specialization
template <int skew>
struct OutputDataReOrder {
    template <int t_L, int t_R, typename T_dtype, typename T_dtype2>
    void digitReversal2Phase(hls::stream<SuperSampleContainer<t_R, T_dtype> >& p_inData,
                             T_dtype2 p_outData[t_R][t_L / t_R]);
    template <int t_L, int t_R, typename T_dtype, typename T_dtype2>
    void digitReversal2Phase(hls::stream<SuperSampleContainer<t_R, T_dtype> >& p_inData,
                             hls::stream<T_dtype2> p_outData[t_R]);
};

#if 1
template <int t_skew>
template <int t_L, int t_R, typename T_dtype, typename T_dtype2>
void OutputDataReOrder<t_skew>::digitReversal2Phase(hls::stream<SuperSampleContainer<t_R, T_dtype> >& p_inData,
                                                    T_dtype2 p_outData[t_R][t_L / t_R]) {
#pragma HLS INLINE

    ///////////////////additions for adding a reshaping module.
    // T_dtype inDatac_reshaped[t_R][t_L / t_R];
    hls::stream<T_dtype> inDatac_reshaped[t_R];
//#pragma HLS data_pack variable = inDatac_reshaped
#pragma HLS STREAM variable = inDatac_reshaped depth = 8 // dim = 2
#pragma HLS RESOURCE variable = inDatac_reshaped core = FIFO_LUTRAM
    //#pragma HLS ARRAY_RESHAPE variable = inDatac_reshaped complete dim = 1
    streamJoinUtilitySISO<t_L, t_R, T_dtype>(p_inData, inDatac_reshaped);

    T_dtype2 pingPongBuff[t_R][t_L / t_R];
#pragma HLS ARRAY_PARTITION variable = pingPongBuff complete dim = 1
#pragma HLS data_pack variable = pingPongBuff

    // Process:1:Create a Process for buffering streaming data in digit swapped order also do the
    // criss cross from different memories
    crissCrossAndTimeDigitSwap<t_L, t_R>(inDatac_reshaped, pingPongBuff);

    // Process:2:Create 2nd process that connect to process-1 through PIPO and generates an output stream
    diagonalWrapIndexReadFromPIPOAndStreamOut<t_L, t_R>(pingPongBuff, p_outData);
}
template <int t_skew>
template <int t_L, int t_R, typename T_dtype, typename T_dtype2>
void OutputDataReOrder<t_skew>::digitReversal2Phase(hls::stream<SuperSampleContainer<t_R, T_dtype> >& p_inData,
                                                    hls::stream<T_dtype2> p_outData[t_R]) {
#pragma HLS INLINE

    ///////////////////additions for adding a reshaping module.
    // T_dtype inDatac_reshaped[t_R][t_L / t_R];
    hls::stream<T_dtype> inDatac_reshaped[t_R];
//#pragma HLS data_pack variable = inDatac_reshaped
#pragma HLS STREAM variable = inDatac_reshaped depth = 8 // dim = 2
#pragma HLS RESOURCE variable = inDatac_reshaped core = FIFO_LUTRAM
    //#pragma HLS ARRAY_RESHAPE variable = inDatac_reshaped complete dim = 1
    streamJoinUtilitySISO<t_L, t_R, T_dtype>(p_inData, inDatac_reshaped);

    T_dtype2 pingPongBuff[t_R][t_L / t_R];
#pragma HLS ARRAY_PARTITION variable = pingPongBuff complete dim = 1
    //#pragma HLS data_pack variable = pingPongBuff

    // Process:1:Create a Process for buffering streaming data in digit swapped order also do the
    // criss cross from different memories
    crissCrossAndTimeDigitSwap<t_L, t_R>(inDatac_reshaped, pingPongBuff);

    // Process:2:Create 2nd process that connect to process-1 through PIPO and generates an output stream
    diagonalWrapIndexReadFromPIPOAndStreamOut<t_L, t_R>(pingPongBuff, p_outData);
}
#endif

template <>
template <int t_L, int t_R, typename T_dtype, typename T_dtype2>
void OutputDataReOrder<0>::digitReversal2Phase(hls::stream<SuperSampleContainer<t_R, T_dtype> >& p_inData,
                                               T_dtype2 p_outData[t_R][t_L / t_R]) {
#pragma HLS INLINE

    ///////////////////additions to adding a reshaping module.
    // T_dtype inDatac_reshaped[t_R][t_L / t_R];
    hls::stream<T_dtype> inDatac_reshaped[t_R];
//#pragma HLS data_pack variable = inDatac_reshaped
#pragma HLS STREAM variable = inDatac_reshaped depth = 8
#pragma HLS RESOURCE variable = inDatac_reshaped core = FIFO_LUTRAM
    //#pragma HLS ARRAY_RESHAPE variable = inDatac_reshaped complete dim = 1
    streamJoinUtilitySISO<t_L, t_R, T_dtype>(p_inData, inDatac_reshaped);

    T_dtype2 pingPongBuff[t_R][t_L / t_R];
#pragma HLS ARRAY_PARTITION variable = pingPongBuff complete dim = 1
#pragma HLS data_pack variable = pingPongBuff

    ///////////////////////////////////// Shuffle the Data and Write to Buffer ////////////////////////////////////
    // recursivelyProcess:1:Create a Process for buffering streaming data in digit swapped order
    // also do the criss cross from different memories %%%%%%%%%%%%%%
    crissCrossAndTimeDigitSwapSpecial<t_L, t_R>(inDatac_reshaped, pingPongBuff);

    // Process:2:Create 2nd process that connect to process-1 through PIPO and generates an output stream
    diagonalWrapIndexReadFromPIPOAndStreamOutSpecial<t_L, t_R>(pingPongBuff, p_outData);
}
template <>
template <int t_L, int t_R, typename T_dtype, typename T_dtype2>
void OutputDataReOrder<0>::digitReversal2Phase(hls::stream<SuperSampleContainer<t_R, T_dtype> >& p_inData,
                                               hls::stream<T_dtype2> p_outData[t_R]) {
#pragma HLS INLINE

    ///////////////////additions to adding a reshaping module.
    // T_dtype inDatac_reshaped[t_R][t_L / t_R];
    hls::stream<T_dtype> inDatac_reshaped[t_R];
//#pragma HLS data_pack variable = inDatac_reshaped
#pragma HLS STREAM variable = inDatac_reshaped depth = 8
#pragma HLS RESOURCE variable = inDatac_reshaped core = FIFO_LUTRAM
    //#pragma HLS ARRAY_RESHAPE variable = inDatac_reshaped complete dim = 1
    streamJoinUtilitySISO<t_L, t_R, T_dtype>(p_inData, inDatac_reshaped);

    T_dtype2 pingPongBuff[t_R][t_L / t_R];
#pragma HLS ARRAY_PARTITION variable = pingPongBuff complete dim = 1
#pragma HLS data_pack variable = pingPongBuff

    ///////////////////////////////////// Shuffle the Data and Write to Buffer ////////////////////////////////////
    // recursivelyProcess:1:Create a Process for buffering streaming data in digit swapped order
    // also do the criss cross from different memories %%%%%%%%%%%%%%
    crissCrossAndTimeDigitSwapSpecial<t_L, t_R>(inDatac_reshaped, pingPongBuff);

    // Process:2:Create 2nd process that connect to process-1 through PIPO and generates an output stream
    diagonalWrapIndexReadFromPIPOAndStreamOutSpecial<t_L, t_R>(pingPongBuff, p_outData);
}

} // end namespace fft
} // end namespace dsp
} // end namespace xf

#endif // HLS_SSR_FFT_OUTPUT_REORDER_h_
