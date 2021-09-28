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

#ifndef HLS_SSR_FFT_DATA_REORDER_H
#define HLS_SSR_FFT_DATA_REORDER_H

#ifndef __SYNTHESIS__
#include <math.h>
#include <iostream>
#endif
//#include <complex>
#include "vitis_fft/hls_ssr_fft_utilities.hpp"
#include "vitis_fft/hls_ssr_fft_read_barrel_shifter.hpp"
#include "vitis_fft/hls_ssr_fft_output_reorder.hpp"
#include "vitis_fft/fft_complex.hpp"

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

// SSR_FFT_VIVADO_BEGIN
template <int t_L, int t_R, typename T_dtype>
void cacheDataDR(complex_wrapper<T_dtype> p_inData[t_R][t_L / t_R],
                 complex_wrapper<T_dtype> p_digitReseversedOutputBuff[t_R][t_L / t_R]) {
#pragma HLS INLINE off
/*
 #pragma HLS ARRAY_PARTITION variable=p_inData complete dim=1
 #pragma HLS ARRAY_PARTITION variable=p_digitReseversedOutputBuff complete dim=1
 */

cacheDataDR_LOverRLooP:
    for (int r = 0; r < (t_L / t_R); r++) {
#pragma HLS PIPELINE II = 1 rewind

        complex_wrapper<T_dtype> temp[t_R];
#pragma HLS ARRAY_PARTITION variable = temp complete dim = 1
    //#pragma HLS data_pack variable = temp

    cacheDataDR_SSRLoop1:
        for (int c = 0; c < t_R; c++) {
            // int cdash = (c +  r / ( t_L / (t_R*t_R) )   )%t_R;
            // replaced//int cdash = (c +  (r*t_R*t_R) / ( t_L  )   )%t_R;
            int cdash = (c + ((r) >> (ssrFFTLog2<t_L / (t_R * t_R)>::val))) & (ssrFFTLog2BitwiseAndModMask<t_R>::val);
            // CHECK_COVEARAGE;
            temp[cdash] = p_inData[c][r]; // Read in Order :: Should be a stream
        }
    cacheDataDR_SSRLoop2:
        for (int c = 0; c < t_R; c++) {
            p_digitReseversedOutputBuff[c][r] = temp[c];
        }
    }
}
// SSR_FFT_VIVADO_END
template <int t_L, int t_R, typename T_dtype>
void cacheDataDR(hls::stream<complex_wrapper<T_dtype> > p_inData[t_R],
                 complex_wrapper<T_dtype> p_digitReseversedOutputBuff[t_R][t_L / t_R]) {
#pragma HLS INLINE off

cacheDataDR_LOverRLooP:
    for (int r = 0; r < (t_L / t_R); r++) {
#pragma HLS PIPELINE II = 1 rewind

        complex_wrapper<T_dtype> temp[t_R];
#pragma HLS ARRAY_PARTITION variable = temp complete dim = 1
    //#pragma HLS data_pack variable = temp

    cacheDataDR_SSRLoop1:
        for (int c = 0; c < t_R; c++) {
            // int cdash = (c +  r / ( t_L / (t_R*t_R) )   )%t_R;
            // replaced//int cdash = (c +  (r*t_R*t_R) / ( t_L  )   )%t_R;
            int cdash = (c + ((r) >> (ssrFFTLog2<t_L / (t_R * t_R)>::val))) & (ssrFFTLog2BitwiseAndModMask<t_R>::val);
            // CHECK_COVEARAGE;
            p_inData[c].read(temp[cdash]);
        }
    cacheDataDR_SSRLoop2:
        for (int c = 0; c < t_R; c++) {
            p_digitReseversedOutputBuff[c][r] = temp[c];
        }
    }
}

// SSR_FFT_VIVADO_BEGIN
template <int t_L, int t_R, typename T_in, typename T_out>
void cacheDataDR(complex_wrapper<T_in> p_inData[t_R][t_L / t_R],
                 complex_wrapper<T_out> p_digitReseversedOutputBuff[t_R][t_L / t_R]) {
#pragma HLS INLINE off
    const unsigned int log2_radix = (ssrFFTLog2<t_R>::val);
cacheDataDR_LOverRLooP:
    for (int r = 0; r < (t_L / t_R); r++) {
#pragma HLS PIPELINE II = 1 rewind

        complex_wrapper<T_in> temp[t_R];
#pragma HLS ARRAY_PARTITION variable = temp complete dim = 1
    //#pragma HLS data_pack variable = temp

    cacheDataDR_SSRLoop1:
        for (int c = 0; c < t_R; c++) {
            // replaced//int cdash = (c +  (r*t_R*t_R) / ( t_L  )   )%t_R;
            int cdash = (c + (r >> (ssrFFTLog2<t_L / (t_R * t_R)>::val))) & (ssrFFTLog2BitwiseAndModMask<t_R>::val);
            // CHECK_COVEARAGE;

            temp[cdash] = p_inData[c][r]; // Read in Order :: Should be a stream
        }
    cacheDataDR_SSRLoop2:
        for (int c = 0; c < t_R; c++) {
            p_digitReseversedOutputBuff[c][r] = temp[c];
        }
    }
}
// SSR_FFT_VIVADO_END
template <int t_L, int t_R, typename T_in, typename T_out>
void cacheDataDR(hls::stream<complex_wrapper<T_in> > p_inData[t_R],
                 complex_wrapper<T_out> p_digitReseversedOutputBuff[t_R][t_L / t_R]) {
#pragma HLS INLINE off
    const unsigned int log2_radix = (ssrFFTLog2<t_R>::val);
cacheDataDR_LOverRLooP:
    for (int r = 0; r < (t_L / t_R); r++) {
#pragma HLS PIPELINE II = 1 rewind

        complex_wrapper<T_in> temp[t_R];
#pragma HLS ARRAY_PARTITION variable = temp complete dim = 1
    //#pragma HLS data_pack variable = temp

    cacheDataDR_SSRLoop1:
        for (int c = 0; c < t_R; c++) {
            // replaced//int cdash = (c +  (r*t_R*t_R) / ( t_L  )   )%t_R;
            int cdash = (c + (r >> (ssrFFTLog2<t_L / (t_R * t_R)>::val))) & (ssrFFTLog2BitwiseAndModMask<t_R>::val);
            // CHECK_COVEARAGE;

            p_inData[c].read(temp[cdash]);
        }
    cacheDataDR_SSRLoop2:
        for (int c = 0; c < t_R; c++) {
            p_digitReseversedOutputBuff[c][r] = temp[c];
        }
    }
}

// SSR_FFT_VIVADO_BEGIN
template <int t_L, int t_R, typename T_dtype>
void writeBackCacheDataDR(complex_wrapper<T_dtype> p_digitReseversedOutputBuff[t_R][t_L / t_R],
                          complex_wrapper<T_dtype> p_outData[t_R][t_L / t_R]) {
#pragma HLS INLINE off

    const unsigned int log2_radix = (ssrFFTLog2<t_R>::val);

writeBackCacheDataDR_LOverRLoop:
    for (int r = 0; r < (t_L / t_R); r++) {
#pragma HLS PIPELINE II = 1 rewind // This loop has rewind issue : VERIFIED

        complex_wrapper<T_dtype> temp[t_R];
#pragma HLS ARRAY_PARTITION variable = temp complete dim = 1
#pragma HLS data_pack variable = temp

        unsigned int lin_index = (r << log2_radix) | 0; // equivalent to : r*t_R + c;
        unsigned int bitReversedIndex = digitReversalFractionIsLSB<t_L, t_R>(lin_index);
        unsigned int out_r = bitReversedIndex >> log2_radix;             // equivalent to :  bitReversedIndex / t_R;
        unsigned int out_c = bitReversedIndex & ((1 << log2_radix) - 1); // equivalent to:bitReversedIndex % t_R;
        // int offset = (out_c  +  (out_r  /  ( t_L / (t_R*t_R) )    ) ) %t_R;//int out_cDash = (out_c  +  (out_r/t_R) )
        // %t_R; // ((r>>log2_radix) + c)%t_R;     //  int offset = (out_c  +  ( (out_r *t_R*t_R) /  ( t_L  )    ) )
        // %t_R;//int out_cDash = (out_c  +  (out_r/t_R) ) %t_R; // ((r>>log2_radix) + c)%t_R;     //  replaced//
        // int offset = (out_c  +  ( (out_r *t_R*t_R) /  ( t_L  )    ) ) %t_R;//int out_cDash = (out_c  +  (out_r/t_R) )
        // %t_R; // ((r>>log2_radix) + c)%t_R;     //
        int offset = (out_c + (out_r >> (ssrFFTLog2<t_L / (t_R * t_R)>::val))) &
                     (ssrFFTLog2BitwiseAndModMask<t_R>::val); // int out_cDash = (out_c  +  (out_r/t_R) ) %t_R; //
                                                              // ((r>>log2_radix) + c)%t_R;     //

        MemReadBarrelShifter<t_R> readBarrelShifterObj;
        readBarrelShifterObj.template readMemAndBarrelShift<t_R, t_L, complex_wrapper<T_dtype> >(
            r, offset, p_digitReseversedOutputBuff, temp);
        for (int c = 0; c < t_R; c++) {
            p_outData[c][r] = temp[c]; // p_outData is written in order should be a stream
        }
    }
}
// SSR_FFT_VIVADO_END
template <int t_L, int t_R, typename T_dtype>
void writeBackCacheDataDR(complex_wrapper<T_dtype> p_digitReseversedOutputBuff[t_R][t_L / t_R],
                          hls::stream<complex_wrapper<T_dtype> > p_outData[t_R]) {
#pragma HLS INLINE off

    const unsigned int log2_radix = (ssrFFTLog2<t_R>::val);

writeBackCacheDataDR_LOverRLoop:
    for (int r = 0; r < (t_L / t_R); r++) {
#pragma HLS PIPELINE II = 1 rewind // This loop has rewind issue : VERIFIED

        complex_wrapper<T_dtype> temp[t_R];
#pragma HLS ARRAY_PARTITION variable = temp complete dim = 1
#pragma HLS data_pack variable = temp

        unsigned int lin_index = (r << log2_radix) | 0; // equivalent to : r*t_R + c;
        unsigned int bitReversedIndex = digitReversalFractionIsLSB<t_L, t_R>(lin_index);
        unsigned int out_r = bitReversedIndex >> log2_radix;             // equivalent to :  bitReversedIndex / t_R;
        unsigned int out_c = bitReversedIndex & ((1 << log2_radix) - 1); // equivalent to:bitReversedIndex % t_R;
        // int offset = (out_c  +  (out_r  /  ( t_L / (t_R*t_R) )    ) ) %t_R;//int out_cDash = (out_c  +  (out_r/t_R) )
        // %t_R; // ((r>>log2_radix) + c)%t_R;     //  int offset = (out_c  +  ( (out_r *t_R*t_R) /  ( t_L  )    ) )
        // %t_R;//int out_cDash = (out_c  +  (out_r/t_R) ) %t_R; // ((r>>log2_radix) + c)%t_R;     //  replaced//
        // int offset = (out_c  +  ( (out_r *t_R*t_R) /  ( t_L  )    ) ) %t_R;//int out_cDash = (out_c  +  (out_r/t_R) )
        // %t_R; // ((r>>log2_radix) + c)%t_R;     //
        int offset = (out_c + (out_r >> (ssrFFTLog2<t_L / (t_R * t_R)>::val))) &
                     (ssrFFTLog2BitwiseAndModMask<t_R>::val); // int out_cDash = (out_c  +  (out_r/t_R) ) %t_R; //
                                                              // ((r>>log2_radix) + c)%t_R;     //

        MemReadBarrelShifter<t_R> readBarrelShifterObj;
        readBarrelShifterObj.template readMemAndBarrelShift<t_R, t_L, complex_wrapper<T_dtype> >(
            r, offset, p_digitReseversedOutputBuff, temp);
        for (int c = 0; c < t_R; c++) {
            p_outData[c].write(temp[c]); // p_outData is written in order should be a stream
        }
    }
}

// SSR_FFT_VIVADO_BEGIN
template <int t_L, int t_R, typename T_in, typename T_out>
void writeBackCacheDataDR(complex_wrapper<T_in> p_digitReseversedOutputBuff[t_R][t_L / t_R],
                          complex_wrapper<T_out> p_outData[t_R][t_L / t_R]) {
#pragma HLS INLINE off

    const unsigned int log2_radix = (ssrFFTLog2<t_R>::val);

writeBackCacheDataDR_LOverRLoop:
    for (int r = 0; r < (t_L / t_R); r++) {
#pragma HLS PIPELINE II = 1 rewind // This loop has rewind issue

        complex_wrapper<T_in> temp[t_R];
#pragma HLS ARRAY_PARTITION variable = temp complete dim = 1
#pragma HLS data_pack variable = temp

        unsigned int lin_index = (r << log2_radix) | 0; // equivalent to : r*t_R + c;
        unsigned int bitReversedIndex = digitReversalFractionIsLSB<t_L, t_R>(lin_index);
        unsigned int out_r = bitReversedIndex >> log2_radix;             // equivalent to :  bitReversedIndex / t_R;
        unsigned int out_c = bitReversedIndex & ((1 << log2_radix) - 1); // equivalent to:bitReversedIndex % t_R;
        // int offset = (out_c  +  (out_r  /  ( t_L / (t_R*t_R) )    ) ) %t_R;//int out_cDash = (out_c  +  (out_r/t_R) )
        // %t_R; // ((r>>log2_radix) + c)%t_R;     //  int offset = (out_c  +  ( (out_r*t_R*t_R)/( t_L )    ) )
        // %t_R;//int out_cDash = (out_c  +  (out_r/t_R) ) %t_R; // ((r>>log2_radix) + c)%t_R;     //  replaced// int
        // offset = (out_c  +  ( (out_r *t_R*t_R) /  ( t_L  )    ) ) %t_R;//int out_cDash = (out_c  +  (out_r/t_R) )
        // %t_R; // ((r>>log2_radix) + c)%t_R;     //
        int offset = (out_c + (out_r >> (ssrFFTLog2<t_L / (t_R * t_R)>::val))) &
                     (ssrFFTLog2BitwiseAndModMask<t_R>::val); // int out_cDash = (out_c  +  (out_r/t_R) ) %t_R; //
                                                              // ((r>>log2_radix) + c)%t_R;     //
        for (int c = 0; c < t_R; c++) {
#pragma HLS UNROLL
            unsigned int lin_index1 = (r << log2_radix) | ((t_R + c - offset) % t_R); // equivalent to : r*t_R + c;
            unsigned int bitReversedIndex1 = digitReversal<t_L, t_R>(lin_index1);
            unsigned int out_r = bitReversedIndex1 >> log2_radix; // equivalent to :  bitReversedIndex / t_R;
            // replaced//out[c]= in[(c+(stage-1))%t_R][out_r];
            temp[(t_R + c - offset) % t_R] = p_digitReseversedOutputBuff[c][out_r];
        }
        //			CHECK_COVEARAGE;
        for (int c = 0; c < t_R; c++) {
            p_outData[c][r] = temp[c]; // p_outData is written in order should be a stream
        }
    }
}
// SSR_FFT_VIVADO_END
template <int t_L, int t_R, typename T_in, typename T_out>
void writeBackCacheDataDR(complex_wrapper<T_in> p_digitReseversedOutputBuff[t_R][t_L / t_R],
                          hls::stream<complex_wrapper<T_out> > p_outData[t_R]) {
#pragma HLS INLINE off

    const unsigned int log2_radix = (ssrFFTLog2<t_R>::val);

writeBackCacheDataDR_LOverRLoop:
    for (int r = 0; r < (t_L / t_R); r++) {
#pragma HLS PIPELINE II = 1 rewind // This loop has rewind issue

        complex_wrapper<T_in> temp[t_R];
#pragma HLS ARRAY_PARTITION variable = temp complete dim = 1
#pragma HLS data_pack variable = temp

        unsigned int lin_index = (r << log2_radix) | 0; // equivalent to : r*t_R + c;
        unsigned int bitReversedIndex = digitReversalFractionIsLSB<t_L, t_R>(lin_index);
        unsigned int out_r = bitReversedIndex >> log2_radix;             // equivalent to :  bitReversedIndex / t_R;
        unsigned int out_c = bitReversedIndex & ((1 << log2_radix) - 1); // equivalent to:bitReversedIndex % t_R;
        // int offset = (out_c  +  (out_r  /  ( t_L / (t_R*t_R) )    ) ) %t_R;//int out_cDash = (out_c  +  (out_r/t_R) )
        // %t_R; // ((r>>log2_radix) + c)%t_R;     //  int offset = (out_c  +  ( (out_r*t_R*t_R)/( t_L )    ) )
        // %t_R;//int out_cDash = (out_c  +  (out_r/t_R) ) %t_R; // ((r>>log2_radix) + c)%t_R;     //  replaced// int
        // offset = (out_c  +  ( (out_r *t_R*t_R) /  ( t_L  )    ) ) %t_R;//int out_cDash = (out_c  +  (out_r/t_R) )
        // %t_R; // ((r>>log2_radix) + c)%t_R;     //
        int offset = (out_c + (out_r >> (ssrFFTLog2<t_L / (t_R * t_R)>::val))) &
                     (ssrFFTLog2BitwiseAndModMask<t_R>::val); // int out_cDash = (out_c  +  (out_r/t_R) ) %t_R; //
                                                              // ((r>>log2_radix) + c)%t_R;     //
        for (int c = 0; c < t_R; c++) {
#pragma HLS UNROLL
            unsigned int lin_index1 = (r << log2_radix) | ((t_R + c - offset) % t_R); // equivalent to : r*t_R + c;
            unsigned int bitReversedIndex1 = digitReversal<t_L, t_R>(lin_index1);
            unsigned int out_r = bitReversedIndex1 >> log2_radix; // equivalent to :  bitReversedIndex / t_R;
            // replaced//out[c]= in[(c+(stage-1))%t_R][out_r];
            temp[(t_R + c - offset) % t_R] = p_digitReseversedOutputBuff[c][out_r];
        }
        //			CHECK_COVEARAGE;
        for (int c = 0; c < t_R; c++) {
            p_outData[c].write(temp[c]); // p_outData is written in order should be a stream
        }
    }
}

// SSR_FFT_VIVADO_BEGIN
template <int t_L, int t_R, typename T_in, typename T_out>
void digitReversedDataReOrder(complex_wrapper<T_in> p_inData[t_R][t_L / t_R],
                              complex_wrapper<T_out> p_outData[t_R][t_L / t_R]) {
#pragma HLS INLINE

    const unsigned int log2_radix = (ssrFFTLog2<t_R>::val);

    complex_wrapper<T_in> digitReverseBuff[t_R][t_L / t_R];
#pragma HLS ARRAY_PARTITION variable = digitReverseBuff complete dim = 1
#pragma HLS data_pack variable = digitReverseBuff
    //#pragma HLS STREAM variable=digitReverseBuff off depth=4 dim=2

    cacheDataDR<t_L, t_R, T_in, T_in>(p_inData, digitReverseBuff);
    writeBackCacheDataDR<t_L, t_R, T_in, T_out>(digitReverseBuff, p_outData);
}

template <int t_L, int t_R, typename T_in, typename T_out>
void digitReversedDataReOrder(hls::stream<complex_wrapper<T_in> > p_inData[t_R],
                              complex_wrapper<T_out> p_outData[t_R][t_L / t_R]) {
#pragma HLS INLINE

    const unsigned int log2_radix = (ssrFFTLog2<t_R>::val);

    complex_wrapper<T_in> digitReverseBuff[t_R][t_L / t_R];
#pragma HLS ARRAY_PARTITION variable = digitReverseBuff complete dim = 1
    //#pragma HLS data_pack variable = digitReverseBuff
    //#pragma HLS STREAM variable=digitReverseBuff off depth=4 dim=2

    cacheDataDR<t_L, t_R, T_in, T_in>(p_inData, digitReverseBuff);
    writeBackCacheDataDR<t_L, t_R, T_in, T_out>(digitReverseBuff, p_outData);
}
// SSR_FFT_VIVADO_END
template <int t_L, int t_R, typename T_in, typename T_out>
void digitReversedDataReOrder(hls::stream<complex_wrapper<T_in> > p_inData[t_R],
                              hls::stream<complex_wrapper<T_out> > p_outData[t_R]) {
    //#pragma HLS INLINE

    const unsigned int log2_radix = (ssrFFTLog2<t_R>::val);

    complex_wrapper<T_in> digitReverseBuff[t_R][t_L / t_R];
#pragma HLS ARRAY_PARTITION variable = digitReverseBuff complete dim = 1
    //#pragma HLS data_pack variable = digitReverseBuff
    //#pragma HLS STREAM variable = digitReverseBuff depth = 4

    cacheDataDR<t_L, t_R, T_in, T_in>(p_inData, digitReverseBuff);
    writeBackCacheDataDR<t_L, t_R, T_in, T_out>(digitReverseBuff, p_outData);
}

template <int t_L, int t_R, typename T_in, typename T_out>
void digitReversalSimulationModel(T_in p_inData[t_R][t_L / t_R], T_out p_outData[t_R][t_L / t_R]) {
    unsigned int ind1 = 0;
    unsigned int revind = 0;

    for (int i = 0; i < t_R; i++) {
        for (int j = 0; j < t_L / t_R; j++) {
            ind1 = j * (t_R) + i;
            revind = digitReversalFractionIsLSB<t_L, t_R>(ind1);
            p_outData[revind % t_R][revind / t_R] = p_inData[i][j];
        }
    }
}
} // end namespace fft
} // end namespace dsp
} // end namespace xf

#endif // !HLS_SSR_FFT_DATA_REORDER_H
