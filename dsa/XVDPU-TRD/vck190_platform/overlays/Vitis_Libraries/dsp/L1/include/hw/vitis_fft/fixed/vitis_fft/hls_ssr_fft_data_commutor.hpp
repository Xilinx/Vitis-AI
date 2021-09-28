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

// File Name : hls_ssr_fft_data_commutor.hpp
#ifndef HLS_SSR_FFR_DATA_COMMUTOR_H_
#define HLS_SSR_FFR_DATA_COMMUTOR_H_

/*
 =========================================================================================
 -_- -_-
 -_- -_-
 -_- -_-
 -_- -_-
 -_- -_-
 -_- -_-
 -_-

 The data commuter is a basic block used to implement data re ordering
 at the input SSR FFT and also in between the SSR FFT Stages. The data commuter
 has two stages one read in R streams multiplexing them before storage to ping
 pong buffers in a circular rotate(word level rotation not bit like) fashion.
 The 2nd stage reads in R memories and streams it out to R different streams.
 The memory to stream mapping changes in every cycle. The Whole transformation
 is 4 phase:
 1- The input streams are rotated
 2- The input stream written to PIPO after rotation
 3- The ping pong memory is read
 4- The read data is shuffled and written to output
 This file defines functions for phases 1,2,3
 cacheData function : deals with phase 1 and 2
 WriteCacheData  function deals with phase  3,4
 and internally calls
 CommuteBarrelShifter::memReadCommuteBarrelShifter

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

#include "vitis_fft/hls_ssr_fft_fork_merge_utils.hpp"
#include "vitis_fft/hls_ssr_fft_data_commute_barrel_shifter.hpp"
#include "vitis_fft/hls_ssr_fft_pragma_controls.hpp"
namespace xf {
namespace dsp {
namespace fft {

template <int t_stageNumber, int t_id> // wrapper is created to have a unique copy of writeOutput whenever it is called
                                       // with different t_id and stage such that static variables don't get shared.
struct WriteOutputWrapper {
    template <int t_L, int t_R, int t_PF, typename T_dtype>
    void writeOutput(T_dtype p_in[t_R], T_dtype p_out[t_R][t_L / t_R]) {
#pragma HLS PIPELINE II = 1
#pragma HLS INLINE off

        static int f = 0;
        static int n = 0;
    writeOutput_SSRLoop:
        for (int s = 0; s < t_R; s++) {
#pragma HLS UNROLL
            int p = n % t_PF;
            int r1 = n / t_PF;
            // replaced//int out1_index = (f*(t_PF*t_R)) + r1*t_PF + p;
            int out1_index = (f << (ssrFFTLog2<(t_PF * t_R)>::val)) + (r1 << (ssrFFTLog2<t_PF>::val)) + p;
            int out2_index = s;
            // int cshift = (s*t_PF + p)/t_PF;
            p_out[s][out1_index] = p_in[s];
        }
        n++;
        if (n == t_PF * t_R) {
            n = 0;
            f++;
            if (f == t_L / (t_PF * t_R * t_R)) {
                f = 0;
            }
        }
    }
};

template <int t_stageNumber, int t_id> // wrapper is created to have a unique copy of writeOutput whenever it is called
                                       // with different t_id and stage such that static variables don't get shared.
struct writeOutputPF1Wrapper {
    template <int t_XF, int t_L, int t_R, int t_PF, typename T_dtype>
    void writeOutputPF1(T_dtype p_in[t_R], T_dtype p_out[t_R][t_L / t_R]) {
#pragma HLS PIPELINE II = 1
#pragma HLS INLINE off
        static int test = 0;

        static int f = 0;
        static int n = 0;

    writeOutput_SSRLoop:
        for (int s = 0; s < t_R; s++) {
            // int  p = 0;
            int r1 = n / t_PF;
#pragma HLS UNROLL
            int out1_index = test; //(f*(t_PF*t_R)) + r1*t_PF;
            int out2_index = s;
            // int cshift = (s*t_PF + p)/t_PF;
            p_out[s][out1_index] = p_in[s];
        }

        n++;
        if (n == t_PF * t_R) {
            n = 0;
            f++;
            if (f == t_L / (t_PF * t_R * t_R * t_XF)) {
                f = 0;
            }
        }

        if (test == (t_L / t_R - 1))
            test = 0;
        else
            test++;
    }
};
template <int t_stageNumber, int t_id> // wrapper is created to have a unique copy of writeOutput whenever it is called
                                       // with different t_id and stage such that static variables don't get shared.
struct BarrelShiftWrapperCover {
    template <int t_L, int t_R, int t_PF, typename T_dtype>
    void barrelShiftWrapperFunction(T_dtype p_buff[t_R][t_PF * t_R], T_dtype p_in[t_R]) {
#pragma HLS PIPELINE II = 1
        static int n = 0;
        int p = n % t_PF;
        int r1 = n / t_PF; // n>>(ssrFFTLog2<t_PF>::val);
        CommuteBarrelShifter<t_R> CommuteBarrelShifterObj;
        CommuteBarrelShifterObj.template memReadCommuteBarrelShifter<t_R, t_L, t_PF, T_dtype>(((r1 + (p) / t_PF) % t_R),
                                                                                              p, p_buff, p_in);
        n++;
        if (n == t_PF * t_R) n = 0;
    }
};
template <int t_stageNumber, int t_id> // wrapper is created to have a unique copy of writeOutput whenever it is called
                                       // with different t_id and stage such that static variables don't get shared.
struct barrelShiftWrapperFunctionpPF1wrapper {
    template <int t_XF, int t_L, int t_R, int t_PF, typename T_dtype>
    void barrelShiftWrapperFunctionpf1(T_dtype p_buff[t_R][t_XF * t_PF * t_R], T_dtype p_in[t_R]) {
#pragma HLS PIPELINE II = 1
        static int n = 0;

        // int  p = n % t_PF;
        int r1 = n; // n / t_PF;
#pragma HLS INLINE off
        CommuteBarrelShifterPF1<t_R> CommuteBarrelShifterObj;
        CommuteBarrelShifterObj.template memReadCommuteBarrelShifterPF1<t_XF, t_R, t_L, t_PF, T_dtype>(r1, 0, p_buff,
                                                                                                       p_in);

        n++;
        if (n == t_PF * t_R * t_XF) n = 0;
    }
};
template <int t_stageNumber, int t_id> // wrapper is created to have a unique copy of writeOutput whenever it is called
                                       // with different t_id and stage such that static variables don't get shared.
struct CacheDataWrapper {
    template <int t_L, int t_R, int t_PF, typename T_dtype>
    void CacheData(T_dtype p_in[t_R][t_L / t_R], T_dtype p_buff[t_R][t_PF * t_R]) {
        static int f = 0;

#pragma HLS INLINE off
    CacheData_BuffLoop:
        for (int rw = 0; rw < t_PF * t_R; rw++) {
#pragma HLS PIPELINE II = 1 rewind
            T_dtype temp[t_R];
#pragma HLS ARRAY_PARTITION variable = temp complete dim = 1
#pragma HLS DATA_PACK variable = temp
        CacheData_SSRLoop1:
            for (int s = 0; s < t_R; s++) {
                // replaced//int cshift= (rw+f*(t_PF*t_R))/t_PF;
                int cshift = (rw + (f << (ssrFFTLog2<(t_PF * t_R)>::val))) >> (ssrFFTLog2<t_PF>::val);

                // replaced//temp[(s+cshift)%t_R]=p_in[s][rw+f*(t_PF*t_R)];
                temp[(s + cshift) & (ssrFFTLog2BitwiseAndModMask<t_R>::val)] =
                    p_in[s][rw + (f << ssrFFTLog2<(t_PF * t_R)>::val)];
                // CHECK_COVEARAGE;
            }
        CacheData_SSRLoop2:
            for (int s = 0; s < t_R; s++) {
                p_buff[s][rw] = temp[s];
            }
            if (rw == t_PF * t_R - 1) {
                f++;
                if (f == t_L / (t_PF * t_R * t_R)) {
                    f = 0;
                }
            }
        }
    }
};

template <int t_stage, int t_id> // wrapper is created to have a unique copy of writeOutput whenever it is called with
                                 // different t_id and t_stage such that static variables don't get shared.
struct WriteCachedDataWrapper {
    template <int t_L, int t_R, int t_PF, typename T_dtype>
    void writeCachedData(T_dtype p_buff[t_R][t_PF * t_R], T_dtype p_out[t_R][t_L / t_R]) {
#pragma HLS INLINE off

        BarrelShiftWrapperCover<t_stage, t_id> barrelShiftWrapperCover_obj;
        WriteOutputWrapper<t_stage, t_id> writeOutputWrapper_obj;
    dataCommutorReOrder_radix_x_pf_LOOP:
        for (int r1 = 0; r1 < (t_R * t_PF); r1++) {
#pragma HLS PIPELINE II = 1
            T_dtype temp[t_R];
#pragma HLS ARRAY_PARTITION variable = temp complete dim = 1
#pragma HLS DATA_PACK variable = temp

            for (int c = 0; c < t_R; c++) {
#pragma HLS UNROLL

                unsigned int out_r =
                    ((t_R + c - (r1 / t_PF)) % t_R) * t_PF + (r1 % t_PF); // equivalent to :  bitReversedIndex / t_R;
                temp[(t_R + c - (r1 / t_PF)) % t_R] = p_buff[c][out_r];
            }

            writeOutputWrapper_obj.template writeOutput<t_L, t_R, t_PF, T_dtype>(temp, p_out);

            //}
        } // re_order_loop;
    }
};

template <int t_stageNumber, int t_id> // wrapper is created to have a unique copy of writeOutput whenever it is called
                                       // with different t_id and t_stage such that static variables don't get shared.
struct CacheDataPF1Wrapper {
    template <int t_XF, int t_L, int t_R, int t_PF, typename T_dtype>
    void CacheData_pf1(T_dtype p_in[t_R][t_L / t_R], T_dtype p_buff[t_R][t_XF * t_R]) {
#pragma HLS INLINE off

        static int f = 0;
        static int test = 0;
    CacheData_loop_pf1:
        for (int rw = 0; rw < (t_R * t_XF); rw++) {
#pragma HLS PIPELINE II = 1 rewind

            T_dtype temp[t_R];
#pragma HLS ARRAY_PARTITION variable = temp complete dim = 1
#pragma HLS DATA_PACK variable = temp

            for (int s = 0; s < t_R; s++) {
#pragma HLS UNROLL
                // replaced//temp[(s+rw)%t_R]=p_in[s][rw+f*(t_R)];
                temp[(s + rw) % t_R] = p_in[s][test]; // p_in[s][(rw%t_R)+((rw/t_R)*t_R)+ f*(t_R)];
                // temp[(s+rw)%t_R]=p_in[rw+f*(t_XF*t_R)+(rw%t_R)][s];
                // temp[(s+rw) & (ssrFFTLog2BitwiseAndModMask<t_XF*t_R>::val)]=p_in[s][rw + ( f<<(ssrFFTLog2<t_R>::val)
                // )];  CHECK_COVEARAGE;
            }
            for (int s = 0; s < t_R; s++) {
#pragma HLS UNROLL
                p_buff[s][rw] = temp[s];
            }
            if (rw == t_L * t_R - 1) {
                f++;
                if (f == t_L / (t_PF * t_R * t_R * t_XF)) {
                    f = 0;
                }
            }

            if (test == (t_L / t_R - 1))
                test = 0;
            else
                test++;
        }
    }
};

template <int t_stage, int t_id> // wrapper is created to have a unique copy of writeOutput whenever it is called with
                                 // different t_id and t_stage such that static variables don't get shared.
struct WriteCachedDataPF1Wrapper {
    template <int t_XF, int t_L, int t_R, int t_PF, typename T_dtype>
    void writeCachedDataPF1(T_dtype p_buff[t_R][t_R * t_XF], T_dtype p_out[t_R][t_L / t_R]) {
#pragma HLS dataflow
#pragma HLS INLINE off

        barrelShiftWrapperFunctionpPF1wrapper<t_stage, t_id> barrelShiftWrapperFunctionpPF1wrapper_obj;
        writeOutputPF1Wrapper<t_stage, t_id> writeOutputPF1Wrapper_obj;
    writeCachedDataPF1_loop:
        for (int r1d = 0; r1d < t_XF * t_R; r1d++) {
#pragma HLS PIPELINE II = 1 // rewind  /// This loop has rewinding issues : VERIFIED

            int r1 = r1d % t_R;
            int offset = (r1d / t_R) * t_R;
            T_dtype temp[t_R];

#pragma HLS ARRAY_PARTITION variable = temp complete dim = 1
#pragma HLS DATA_PACK variable = temp
            for (int c = 0; c < t_R; c++) {
#pragma HLS UNROLL

                unsigned int out_r = ((t_R + c - (r1 / t_PF)) % t_R) * t_PF + (r1 % t_PF) +
                                     offset; // equivalent to :  bitReversedIndex / t_R;
                temp[(t_R + c - (r1 / t_PF)) % t_R] = p_buff[c][out_r];
            }

            writeOutputPF1Wrapper_obj.template writeOutputPF1<t_XF, t_L, t_R, t_PF, T_dtype>(temp, p_out);
        } // re_order_loop;
    }
};

template <int t_stage, int t_id, int t_PF, int tp_t_isLargeMemory>
struct DataCommutations {
    template <int t_L, int t_R, typename T_dtype>
    void dataCommutor(T_dtype p_in[t_R][t_L / t_R], T_dtype p_out[t_R][t_L / t_R]);
};
template <int t_stage, int t_id, int t_PF>
struct DataCommutations<t_stage, t_id, t_PF, 1> {
    template <int t_L, int t_R, typename T_dtype>
    void dataCommutor(T_dtype p_in[t_R][t_L / t_R], T_dtype p_out[t_R][t_L / t_R]);
};

template <int t_stage, int t_id, int t_PF>
struct DataCommutations<t_stage, t_id, t_PF, 0> {
    template <int t_L, int t_R, typename T_dtype>
    void dataCommutor(T_dtype p_in[t_R][t_L / t_R], T_dtype p_out[t_R][t_L / t_R]);
};
template <int t_stage, int t_id>
struct DataCommutations<t_stage, t_id, 1, 0> {
    template <int t_L, int t_R, typename T_dtype>
    void dataCommutor(T_dtype p_in[t_R][t_L / t_R], T_dtype p_out[t_R][t_L / t_R]);
};

template <int t_stage, int t_id>
struct DataCommutations<t_stage, t_id, 1, 1> {
    template <int t_L, int t_R, typename T_dtype>
    void dataCommutor(T_dtype p_in[t_R][t_L / t_R], T_dtype p_out[t_R][t_L / t_R]);
};

/*===================================================================================================================
 * tp_t_isLargeMemory=false  and Generic Packing Factor:: This specialization will be used for the case when the buffer
 * ping pong memory is small and can be implemented using BRAMs are registers.
 * ==================================================================================================================
 */
template <int t_stage, int t_id, int t_PF>
template <int t_L, int t_R, typename T_dtype>
void DataCommutations<t_stage, t_id, t_PF, 0>::dataCommutor(T_dtype p_in[t_R][t_L / t_R],
                                                            T_dtype p_out[t_R][t_L / t_R]) {
#pragma HLS DATAFLOW
#pragma HLS INLINE off

    CacheDataWrapper<t_stage, t_id> cacheDataWrapper_obj;
    WriteCachedDataWrapper<t_stage, t_id> writeCachedDataWrapper_obj;
dataCommutorBuffFrameLoop:
    for (int f = 0; f < (t_L / (t_PF * t_R * t_R)); f++) {
#pragma HLS DATAFLOW
        T_dtype buff[t_R][t_PF * t_R];
#pragma HLS DATA_PACK variable = buff
#pragma HLS ARRAY_PARTITION variable = buff complete dim = 1

        cacheDataWrapper_obj.template CacheData<t_L, t_R, t_PF, T_dtype>(p_in, buff);
        writeCachedDataWrapper_obj.template writeCachedData<t_L, t_R, t_PF, T_dtype>(buff, p_out);
    }
}
/*===================================================================================================================
 * tp_t_isLargeMemory=false  and Packing Factor=1:: This specialization will be used for the case when the buffer
 * ping pong memory is SMALL and can be implemented using BRAMS or registers.
 * ==================================================================================================================
 */
template <int t_stage, int t_id>
template <int t_L, int t_R, typename T_dtype>
void DataCommutations<t_stage, t_id, 1, 0>::dataCommutor(T_dtype p_in[t_R][t_L / t_R], T_dtype p_out[t_R][t_L / t_R]) {
#pragma HLS DATAFLOW
#pragma HLS INLINE off
#ifndef __SYNTHESIS__
    assert((t_L) == (ssrFFTPow<2, ssrFFTLog2<t_L>::val>::val)); // Length should be power of 2 always
#endif
    CacheDataPF1Wrapper<t_stage, t_id> cacheDataWrapperPF1_obj;
    WriteCachedDataPF1Wrapper<t_stage, t_id> writeCachedDataWrapperPF1_obj;
    const int t_XF = (t_L > t_R * t_R) ? (t_L / (t_R * t_R) > 2 ? 4 : t_L / (t_R * t_R)) : 1;
// This factor t_XF makes insides buffers t_XF times the size of minimum required but improves
// throughput : from ( L/R  * (1+  3/R)  ) to : ( L/R  * (1+  3/(R*t_XF))  ) only when ( R*R > L )
dataCommutorBuffFrameLoop_PF1:
    for (int f = 0; f < (t_L / (t_R * t_R * t_XF)); f++) {
//#pragma HLS PIPELINE II=1 rewind
#pragma HLS DATAFLOW
        T_dtype buff[t_R][t_XF * t_R];
#pragma HLS ARRAY_PARTITION variable = buff complete dim = 1
#pragma HLS DATA_PACK variable = buff

        cacheDataWrapperPF1_obj.template CacheData_pf1<t_XF, t_L, t_R, 1, T_dtype>(p_in, buff);

        writeCachedDataWrapperPF1_obj.template writeCachedDataPF1<t_XF, t_L, t_R, 1, T_dtype>(buff, p_out);
    }
}

/*===================================================================================================================
 * tp_t_isLargeMemory=TRUE  and Generic Packing Factor:: This specialization will be used for the case when the buffer
 * ping pong memory is LARGEG and can be implemented using URAMS.
 * ==================================================================================================================
 */
template <int t_stage, int t_id, int t_PF>
template <int t_L, int t_R, typename T_dtype>
void DataCommutations<t_stage, t_id, t_PF, 1>::dataCommutor(T_dtype p_in[t_R][t_L / t_R],
                                                            T_dtype p_out[t_R][t_L / t_R]) {
#pragma HLS DATAFLOW
#pragma HLS INLINE off

    CacheDataWrapper<t_stage, t_id> cacheDataWrapper_obj;
    WriteCachedDataWrapper<t_stage, t_id> writeCachedDataWrapper_obj;
dataCommutorBuffFrameLoop:
    for (int f = 0; f < (t_L / (t_PF * t_R * t_R)); f++) {
#pragma HLS DATAFLOW
        T_dtype buff[t_R][t_PF * t_R];
#pragma HLS DATA_PACK variable = buff
#pragma HLS ARRAY_PARTITION variable = buff complete dim = 1
#pragma HLS RESOURCE variable = buff core = XPM_MEMORY uram

        cacheDataWrapper_obj.template CacheData<t_L, t_R, t_PF, T_dtype>(p_in, buff);
        writeCachedDataWrapper_obj.template writeCachedData<t_L, t_R, t_PF, T_dtype>(buff, p_out);
    }
}
/*===================================================================================================================
 * tp_t_isLargeMemory=TRUE  and Packing Factor=1:: This specialization will be used for the case when the buffer
 * ping pong memory is SMALL and can be implemented using URAMS.
 * ==================================================================================================================
 */
template <int t_stage, int t_id>
template <int t_L, int t_R, typename T_dtype>
void DataCommutations<t_stage, t_id, 1, 1>::dataCommutor(T_dtype p_in[t_R][t_L / t_R], T_dtype p_out[t_R][t_L / t_R]) {
#pragma HLS DATAFLOW
#pragma HLS INLINE off
    assert((t_L) == (ssrFFTPow<2, ssrFFTLog2<t_L>::val>::val)); // Length should be power of 2 always
    CacheDataPF1Wrapper<t_stage, t_id> cacheDataWrapperPF1_obj;
    WriteCachedDataPF1Wrapper<t_stage, t_id> writeCachedDataWrapperPF1_obj;
dataCommutorBuffFrameLoop_PF1:
    for (int f = 0; f < (t_L / (t_R * t_R)); f++) {
#pragma HLS DATAFLOW
        T_dtype buff[t_R][t_R];
#pragma HLS ARRAY_PARTITION variable = buff complete dim = 1
#pragma HLS DATA_PACK variable = buff
#pragma HLS RESOURCE variable = buff core = XPM_MEMORY uram

        cacheDataWrapperPF1_obj.template CacheData_pf1<t_L, t_R, 1, T_dtype>(p_in, buff);
        writeCachedDataWrapperPF1_obj.template writeCachedDataPF1<t_L, t_R, 1, T_dtype>(buff, p_out);
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////New Class for Forking Data Commutor///////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/* DataCommutorFork : This class will take a [L/R][R] type stream with R sample stream and
 * break it down to R/F streams creating F new streams. Functionally it will take [L/R][R] 2 dimensional array
 * and break it down to F new 2 dimensional arrays of size [L/R][R/F] to be used by F dataflow processes
 */

template <int t_stage, int t_forkNumber, int t_forkingFactor>
struct DataCommutorFork {
    template <int t_PF, int t_L, int t_R, typename T_dtype>
    void copyForkCommuteAndMerge(T_dtype p_in[t_R][t_L / t_R], T_dtype p_out[t_R][t_L / t_R]);
};

template <int t_stage, int t_forkingFactor>
struct DataCommutorFork<t_stage, 1, t_forkingFactor> {
    template <int t_PF, int t_L, int t_R, typename T_dtype>
    void copyForkCommuteAndMerge(T_dtype p_in[t_R][t_L / t_R], T_dtype p_out[t_R][t_L / t_R]);
};

// Important Comments :::
/* Forking Function: Copies data t_R/Forking Factor Buffers for forking p_out the output
 * The input is [t_L][t_R] array and output is a also [t_L][t_R] array , but internally
 * The input array is split into smaller arrays like : [t_L][t_R/forking_factor] to create
 * t_R/forking factor input arrays, each of these arrays is assumed to be a seperate function
 * and finally in hardware every such function will map to a seperate process. Essentially it will
 * create a fork, once process feeding multiple processes;
 */

template <int t_stage, int t_forkNumber, int t_forkingFactor>
template <int t_PF, int t_L, int t_R, typename T_dtype>
void DataCommutorFork<t_stage, t_forkNumber, t_forkingFactor>::copyForkCommuteAndMerge(T_dtype p_in[t_R][t_L / t_R],
                                                                                       T_dtype p_out[t_R][t_L / t_R]) {
#pragma HLS INLINE
    T_dtype localFactoredInputBuff[t_R / t_forkingFactor][t_L / t_R];
#pragma HLS DATA_PACK variable = localFactoredInputBuff
#pragma HLS STREAM variable = localFactoredInputBuff depth = 8 dim = 1
#pragma HLS RESOURCE variable = localFactoredInputBuff core = FIFO_LUTRAM

#ifdef SSR_FFT_PARTITION_INTERFACE_ARRAYS
#pragma HLS ARRAY_PARTITION variable = localFactoredInputBuff complete dim = 1
#else
#pragma HLS ARRAY_RESHAPE variable = localFactoredInputBuff complete dim = 1
#endif

    T_dtype localFactoredOutputBuff[t_R / t_forkingFactor][t_L / t_R];
#pragma HLS DATA_PACK variable = localFactoredOutputBuff
#pragma HLS STREAM variable = localFactoredOutputBuff depth = 8 dim = 1
#pragma HLS RESOURCE variable = localFactoredOutputBuff core = FIFO_LUTRAM

//#pragma HLS ARRAY_RESHAPE variable=localFactoredOutputBuff complete dim=1
#ifdef SSR_FFT_PARTITION_INTERFACE_ARRAYS
#pragma HLS ARRAY_PARTITION variable = localFactoredOutputBuff complete dim = 1
#else
#pragma HLS ARRAY_RESHAPE variable = localFactoredOutputBuff complete dim = 1
#endif

    /*
     * Function Signature :
     * template <int t_L, int t_R,int t_forkNumber, int t_forkingFactor, typename T_dtype>
     *
     *			void copyToLocalBuff(T_dtype p_in[t_R][t_L/t_R],T_dtype p_out[t_L/t_R][t_R/t_forkingFactor])
     */

    copyToLocalBuff<t_L, t_R, t_forkNumber, t_forkingFactor, T_dtype>(p_in, localFactoredInputBuff);

    static const int t_isLargeMemFlag =
        ((t_PF * (t_R / t_forkingFactor) > SSR_FFT_URAM_SELECTION_THRESHHOLD) && SSR_FFT_USE_URAMS);

    DataCommutations<t_stage, t_forkNumber, t_PF, t_isLargeMemFlag> commutor;
    commutor.template dataCommutor<t_L / t_forkingFactor, t_R / t_forkingFactor, T_dtype>(localFactoredInputBuff,
                                                                                          localFactoredOutputBuff);

    copyFromLocalBuffToOuput<t_L, t_R, t_forkNumber, t_forkingFactor, T_dtype>(localFactoredOutputBuff, p_out);
    // Create a New Fork to Commute next factor of the output
    DataCommutorFork<t_stage, t_forkNumber - 1, t_forkingFactor> DataCommutorFork_obj;
    DataCommutorFork_obj.template copyForkCommuteAndMerge<t_PF, t_L, t_R, T_dtype>(p_in, p_out);
}

// copyForkCommuteAndMerge base case specialization for fork number = 1, terminates forking/recursion
template <int t_stage, int t_forkingFactor>
template <int t_PF, int t_L, int t_R, typename T_dtype>
void DataCommutorFork<t_stage, 1, t_forkingFactor>::copyForkCommuteAndMerge(T_dtype p_in[t_R][t_L / t_R],
                                                                            T_dtype p_out[t_R][t_L / t_R]) {
#pragma HLS INLINE
    T_dtype localFactoredInputBuff[t_R / t_forkingFactor][t_L / t_R];
#pragma HLS DATA_PACK variable = localFactoredInputBuff
#pragma HLS STREAM variable = localFactoredInputBuff depth = 8 dim = 1
#pragma HLS RESOURCE variable = localFactoredInputBuff core = FIFO_LUTRAM

#ifdef SSR_FFT_PARTITION_INTERFACE_ARRAYS
#pragma HLS ARRAY_PARTITION variable = localFactoredInputBuff complete dim = 1
#else
#pragma HLS ARRAY_RESHAPE variable = localFactoredInputBuff complete dim = 1
#endif

    T_dtype localFactoredOutputBuff[t_R / t_forkingFactor][t_L / t_R];
#pragma HLS DATA_PACK variable = localFactoredOutputBuff
#pragma HLS STREAM variable = localFactoredOutputBuff depth = 8 dim = 1
#pragma HLS RESOURCE variable = localFactoredOutputBuff core = FIFO_LUTRAM

#ifdef SSR_FFT_PARTITION_INTERFACE_ARRAYS
#pragma HLS ARRAY_PARTITION variable = localFactoredOutputBuff complete dim = 1
#else
#pragma HLS ARRAY_RESHAPE variable = localFactoredOutputBuff complete dim = 1
#endif

    copyToLocalBuff<t_L, t_R, 1, t_forkingFactor, T_dtype>(p_in, localFactoredInputBuff);

    static const int t_isLargeMemFlag =
        ((t_PF * (t_R / t_forkingFactor) > SSR_FFT_URAM_SELECTION_THRESHHOLD) && SSR_FFT_USE_URAMS);

    DataCommutations<t_stage, 1, t_PF, t_isLargeMemFlag> commutor;
    commutor.template dataCommutor<t_L / t_forkingFactor, t_R / t_forkingFactor, T_dtype>(localFactoredInputBuff,
                                                                                          localFactoredOutputBuff);
    copyFromLocalBuffToOuput<t_L, t_R, 1, t_forkingFactor, T_dtype>(localFactoredOutputBuff, p_out);
}

/* DataCommutorFork : This class will take a [L/R][R] type stream with R sample stream and
 * break it down to R/F streams creating F new streams. Functionally it will take [L/R][R] 2 dimensional array
 * and break it down to F new 2 dimensional arrays of size [L/R][R/F] to be used by F dataflow processes
 */

template <int t_stage, int t_forkNumber, int t_forkingFactor>
struct DataCommutorForkNonInvertOut {
    template <int t_PF, int t_L, int t_R, typename T_dtype>
    void copyForkCommuteAndMergeNonInvert(T_dtype p_in[t_R][t_L / t_R], T_dtype p_out[t_R][t_L / t_R]);
};

template <int t_stage, int t_forkingFactor>
struct DataCommutorForkNonInvertOut<t_stage, 1, t_forkingFactor> {
    template <int t_PF, int t_L, int t_R, typename T_dtype>
    void copyForkCommuteAndMergeNonInvert(T_dtype p_in[t_R][t_L / t_R], T_dtype p_out[t_R][t_L / t_R]);
};

// Important Comments :::
/* Forking Function: Copies data t_R/Forking Factor Buffers for forking p_out the output
 * The input is [t_L][t_R] array and output is a also [t_L][t_R] array , but internally
 * The input array is split into smaller arrays like : [t_L][t_R/forking_factor] to create
 * t_R/forking factor input arrays, each of these arrays is assumed to be a seperate function
 * and finally in hardware every such function will map to a seperate process. Essentially it will
 * create a fork, once process feeding multiple processes;
 *
 *
 */

template <int t_stage, int t_forkNumber, int t_forkingFactor>
template <int t_PF, int t_L, int t_R, typename T_dtype>
void DataCommutorForkNonInvertOut<t_stage, t_forkNumber, t_forkingFactor>::copyForkCommuteAndMergeNonInvert(
    T_dtype p_in[t_R][t_L / t_R], T_dtype p_out[t_R][t_L / t_R]) {
#pragma HLS INLINE
    T_dtype localFactoredInputBuff[t_R / t_forkingFactor][t_L / t_R];
#pragma HLS DATA_PACK variable = localFactoredInputBuff
#pragma HLS STREAM variable = localFactoredInputBuff depth = 8 dim = 1
#pragma HLS RESOURCE variable = localFactoredInputBuff core = FIFO_LUTRAM

#ifdef SSR_FFT_PARTITION_INTERFACE_ARRAYS
#pragma HLS ARRAY_PARTITION variable = localFactoredInputBuff complete dim = 1
#else
#pragma HLS ARRAY_RESHAPE variable = localFactoredInputBuff complete dim = 1
#endif

    T_dtype localFactoredOutputBuff[t_R / t_forkingFactor][t_L / t_R];
#pragma HLS DATA_PACK variable = localFactoredOutputBuff
#pragma HLS STREAM variable = localFactoredOutputBuff depth = 8 dim = 1
#pragma HLS RESOURCE variable = localFactoredOutputBuff core = FIFO_LUTRAM

#ifdef SSR_FFT_PARTITION_INTERFACE_ARRAYS
#pragma HLS ARRAY_PARTITION variable = localFactoredOutputBuff complete dim = 1
#else
#pragma HLS ARRAY_RESHAPE variable = localFactoredOutputBuff complete dim = 1
#endif

    copyToLocalBuff<t_L, t_R, t_forkNumber, t_forkingFactor, T_dtype>(p_in, localFactoredInputBuff);

    static const int t_isLargeMemFlag =
        ((t_PF * (t_R / t_forkingFactor) > SSR_FFT_URAM_SELECTION_THRESHHOLD) && SSR_FFT_USE_URAMS);
    DataCommutations<t_stage, t_forkNumber, t_PF, t_isLargeMemFlag> commutor;
    commutor.template dataCommutor<t_L / t_forkingFactor, t_R / t_forkingFactor, T_dtype>(localFactoredInputBuff,
                                                                                          localFactoredOutputBuff);

    copyBuffToOutNonInvert<t_L, t_R, t_forkNumber, t_forkingFactor, T_dtype>(localFactoredOutputBuff, p_out);

    // Create a New Fork to Commute next factor of the output
    DataCommutorForkNonInvertOut<t_stage, t_forkNumber - 1, t_forkingFactor> DataCommutorFork_obj_NI;
    DataCommutorFork_obj_NI.template copyForkCommuteAndMergeNonInvert<t_PF, t_L, t_R, T_dtype>(p_in, p_out);
}

// copyForkCommuteAndMerge base case specialization for fork number = 1, terminates forking/recursion
template <int t_stage, int t_forkingFactor>
template <int t_PF, int t_L, int t_R, typename T_dtype>
void DataCommutorForkNonInvertOut<t_stage, 1, t_forkingFactor>::copyForkCommuteAndMergeNonInvert(
    T_dtype p_in[t_R][t_L / t_R], T_dtype p_out[t_R][t_L / t_R]) {
#pragma HLS INLINE
    T_dtype localFactoredInputBuff[t_R / t_forkingFactor][t_L / t_R];
#pragma HLS DATA_PACK variable = localFactoredInputBuff
#pragma HLS STREAM variable = localFactoredInputBuff depth = 8 dim = 1
#pragma HLS RESOURCE variable = localFactoredInputBuff core = FIFO_LUTRAM

#ifdef SSR_FFT_PARTITION_INTERFACE_ARRAYS
#pragma HLS ARRAY_PARTITION variable = localFactoredInputBuff complete dim = 1
#else
#pragma HLS ARRAY_RESHAPE variable = localFactoredInputBuff complete dim = 1
#endif

    T_dtype localFactoredOutputBuff[t_R / t_forkingFactor][t_L / t_R];
#pragma HLS DATA_PACK variable = localFactoredOutputBuff
#pragma HLS STREAM variable = localFactoredOutputBuff depth = 8 dim = 1
#pragma HLS RESOURCE variable = localFactoredOutputBuff core = FIFO_LUTRAM

#pragma HLS ARRAY_RESHAPE variable = localFactoredOutputBuff complete dim = 1
#ifdef SSR_FFT_PARTITION_INTERFACE_ARRAYS
#pragma HLS ARRAY_PARTITION variable = localFactoredOutputBuff complete dim = 1
#else
#pragma HLS ARRAY_RESHAPE variable = localFactoredOutputBuff complete dim = 1
#endif
    copyToLocalBuff<t_L, t_R, 1, t_forkingFactor, T_dtype>(p_in, localFactoredInputBuff);

    static const int t_isLargeMemFlag =
        ((t_PF * (t_R / t_forkingFactor) > SSR_FFT_URAM_SELECTION_THRESHHOLD) && SSR_FFT_USE_URAMS);

    DataCommutations<t_stage, 1, t_PF, t_isLargeMemFlag> commutor;

    commutor.template dataCommutor<t_L / t_forkingFactor, t_R / t_forkingFactor, T_dtype>(localFactoredInputBuff,
                                                                                          localFactoredOutputBuff);

    copyBuffToOutNonInvert<t_L, t_R, 1, t_forkingFactor, T_dtype>(localFactoredOutputBuff, p_out);
}
} // end namespace fft
} // end namespace dsp
} // end namespace xf

#endif // HLS_SSR_FFR_DATA_COMMUTOR_H_
