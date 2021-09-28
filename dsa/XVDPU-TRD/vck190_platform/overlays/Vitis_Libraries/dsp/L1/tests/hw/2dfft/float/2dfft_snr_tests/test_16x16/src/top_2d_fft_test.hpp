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
//================================== End Lic =================================================
#ifndef _TOP_2D_FFT_TEST_H_
#define _TOP_2D_FFT_TEST_H_
#ifndef __SYNTHESIS__
#include <iostream>
#endif
//#define DEBUG_DEMUX
//#define DEBUG_DEMUX
//#define DEBUG_FFT_STREAMING_KERNEL
//#define DEBUG_WIDE_TO_NARROW_CONVERTOR_
#include "vt_fft.hpp"
#include <iostream>
using namespace xf::dsp::fft;
typedef float T_innerData;
typedef complex_wrapper<T_innerData> T_elemType;
const int k_memWidthBits = 512;
const int k_memWidth = k_memWidthBits / (sizeof(complex_wrapper<T_innerData>) * 8);
const int k_fftKernelRadix = 4;
const int k_numOfKernels = k_memWidth / (k_fftKernelRadix);
const int k_fftKernelSize = 16;
typedef float T_innerFloat;
typedef complex_wrapper<T_innerFloat> T_compleFloat;
const int k_numRows = k_fftKernelSize;
const int k_numCols = k_fftKernelSize;
const int k_rowInstanceIDOffset = 40000;
const int k_colInstanceIDOffset = 80000;
const int k_totalWideSamples = k_fftKernelSize * k_fftKernelSize / k_memWidth;
struct FFTParams : ssr_fft_default_params {
    static const int N = k_fftKernelSize;
    static const int R = k_fftKernelRadix;
    static const scaling_mode_enum scaling_mode = SSR_FFT_SCALE;
    static const transform_direction_enum transform_direction = FORWARD_TRANSFORM;
};
struct FFTParams2 : ssr_fft_default_params {
    static const int N = k_fftKernelSize;
    static const int R = k_fftKernelRadix;
    static const scaling_mode_enum scaling_mode = SSR_FFT_SCALE;
    static const transform_direction_enum transform_direction = FORWARD_TRANSFORM;
};

typedef FFTIOTypes<FFTParams, T_elemType>::T_outType T_outType_row;
typedef FFTIOTypes<FFTParams2, T_outType_row>::T_outType T_outType;

typedef WideTypeDefs<k_memWidth, T_elemType>::WideIFType MemWideIFTypeIn;
typedef WideTypeDefs<k_memWidth, T_elemType>::WideIFStreamType MemWideIFStreamTypeIn;

typedef WideTypeDefs<k_memWidth, T_outType>::WideIFType MemWideIFTypeOut;
typedef WideTypeDefs<k_memWidth, T_outType>::WideIFStreamType MemWideIFStreamTypeOut;

void top_fft2d(MemWideIFStreamTypeIn& p_inStream, MemWideIFStreamTypeOut& p_outStream);
#endif
