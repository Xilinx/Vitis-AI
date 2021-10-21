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

/**
 * @file hls_ssr_fft_wide_type_utilities.hpp
 * @brief header provides wide type utilities for conversion between stream types with different widths
 *
 */

#ifndef __HLS_SSR_FFT_WIDE_TYPE_UTILITIES_H__
#define __HLS_SSR_FFT_WIDE_TYPE_UTILITIES_H__

#include <hls_stream.h>

#ifndef __SYNTHESIS__
#include <assert.h>
#endif

#include "vitis_fft/hls_ssr_fft_kernel_types.hpp"
#include "vitis_fft/hls_ssr_fft_utilities.hpp"

namespace xf {
namespace dsp {
namespace fft {

template <unsigned int t_wideStreamWidth,
          unsigned int t_narrowStreamWidth,
          unsigned int t_numOfWideSamples,
          typename T_elemType>
void wideToNarrowConverter(
    typename WideTypeDefs<t_wideStreamWidth, T_elemType>::WideIFStreamType& p_wideStreamIn,
    typename WideTypeDefs<t_narrowStreamWidth, T_elemType>::WideIFStreamType& p_narrowStreamOut) {
#pragma HLS INLINE off
    //#pragma HLS DATA_PACK variable = p_narrowStreamOut
    //#pragma HLS DATA_PACK variable = p_wideStreamIn

    const int k_wide2NarrowRatio = t_wideStreamWidth / t_narrowStreamWidth;
    const int k_ratioLog2 = ssrFFTLog2<k_wide2NarrowRatio>::val;
    const int k_ratioLog2Pow2 = ssrFFTPow<2, k_ratioLog2>::val;

    typedef typename WideTypeDefs<t_wideStreamWidth, T_elemType>::WideIFType T_wideSampleType;
    typedef typename WideTypeDefs<t_narrowStreamWidth, T_elemType>::WideIFType T_narrowSampleType;
#ifndef __SYNTHESIS__
    // assert that ratio is greater than 1
    assert(k_wide2NarrowRatio > 1);
    // assert that ratio is power of 2
    assert(k_wide2NarrowRatio == k_ratioLog2Pow2);
#endif
    T_wideSampleType wideSample;
    //#pragma HLS ARRAY_RESHAPE variable = wideSample.superSample complete dim = 1
    T_narrowSampleType narrowSample;
    //#pragma HLS ARRAY_RESHAPE variable = narrowSample.superSample complete dim = 1

    unsigned int wideReadIndex = 0;
wideToNarrowConverter_LOOP:
    for (int i = 0; i < t_numOfWideSamples * k_wide2NarrowRatio; i++) {
#pragma HLS PIPELINE II = 1 rewind
        if (i % k_wide2NarrowRatio == 0) wideSample = p_wideStreamIn.read();
#ifdef DEBUG_WIDE_TO_NARROW_CONVERTOR_
#ifndef __SYNTHESIS__
        if (i % k_wide2NarrowRatio == 0) std::cout << "Wide Sampl Read at input :  " << wideSample << std::endl;
#endif
#endif
        for (int ns = 0; ns < t_narrowStreamWidth; ns++) {
            narrowSample.superSample[ns] = wideSample.superSample[ns + wideReadIndex];
        }
        p_narrowStreamOut.write(narrowSample);
        wideReadIndex = (wideReadIndex + t_narrowStreamWidth) % t_wideStreamWidth;
    }
}

template <unsigned int t_narrowStreamWidth,
          unsigned int t_wideStreamWidth,
          unsigned int t_numOfWideSamples,
          typename T_elemType>
void narrowToWideConverter(typename WideTypeDefs<t_narrowStreamWidth, T_elemType>::WideIFStreamType& p_narrowStreamIn,
                           typename WideTypeDefs<t_wideStreamWidth, T_elemType>::WideIFStreamType& p_wideStreamOut) {
#pragma HLS INLINE off
    //#pragma HLS DATA_PACK variable = p_wideStreamOut
    //#pragma HLS DATA_PACK variable = p_narrowStreamIn
    const int k_wide2NarrowRatio = t_wideStreamWidth / t_narrowStreamWidth;
    const int k_ratioLog2 = ssrFFTLog2<k_wide2NarrowRatio>::val;
    const int k_ratioLog2Pow2 = ssrFFTPow<2, k_ratioLog2>::val;

    typedef typename WideTypeDefs<t_wideStreamWidth, T_elemType>::WideIFType T_wideSampleType;
    typedef typename WideTypeDefs<t_narrowStreamWidth, T_elemType>::WideIFType T_narrowSampleType;

#ifndef __SYNTHESIS__
    // assert that ratio is greater than 1
    assert(k_wide2NarrowRatio > 1);
    // assert that ratio is power of 2
    assert(k_wide2NarrowRatio == k_ratioLog2Pow2);
#endif
    T_wideSampleType wideSample;
    //#pragma HLS ARRAY_RESHAPE variable = wideSample.superSample complete dim = 1
    T_narrowSampleType narrowSample;
    //#pragma HLS ARRAY_RESHAPE variable = narrowSample.superSample complete dim = 1
    unsigned int wideWriteIndex = 0;
narroToWideConverter_LOOP:
    for (int i = 0; i < t_numOfWideSamples * k_wide2NarrowRatio; i++) {
#pragma HLS PIPELINE II = 1 rewind
        narrowSample = p_narrowStreamIn.read();
        for (int ns = 0; ns < t_narrowStreamWidth; ns++) {
            wideSample[ns + wideWriteIndex] = narrowSample[ns];
        }
        if (i % k_wide2NarrowRatio == 1) p_wideStreamOut.write(wideSample);

        wideWriteIndex = (wideWriteIndex + t_narrowStreamWidth) % t_wideStreamWidth;
    }
}

} // end namespace fft
} // end namespace dsp
} // end namespace xf

#endif //__HLS_SSR_FFT_WIDE_TYPE_UTILITIES_H__
