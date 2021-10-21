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
 * @file hls_ssr_fft_mux.hpp
 * @brief header, describes kernel array interface block for data muxing
 *
 */

#ifndef __HLS_SSR_FFT_MUX_H__
#define __HLS_SSR_FFT_MUX_H__
#include <hls_stream.h>

#include "vitis_fft/hls_ssr_fft_kernel_types.hpp"

namespace xf {
namespace dsp {
namespace fft {

template <unsigned int t_numOfKernels,
          unsigned int t_numRows,
          unsigned int t_numCols,
          unsigned int t_memReadWidth,
          /*t_memReadWidth= No of words of type T_elemType
           * in one wide word read at interface boundary*/
          typename T_elemType>
void muxWideStreaming(
    typename WideTypeDefs<t_memReadWidth, T_elemType>::WideIFStreamType p_inWideStream[t_numOfKernels],
    typename WideTypeDefs<t_memReadWidth, T_elemType>::WideIFStreamType& p_outWideStream) {
#pragma HLS INLINE off
#pragma HLS DATA_PACK variable = p_inWideStream
#pragma HLS DATA_PACK variable = p_outWideStream
#pragma HLS ARRAY_PARTITION variable = p_inWideStream complete dim = 1

    static const unsigned int k_rowIterations = t_numRows / t_memReadWidth;
    static const unsigned int k_colIterations = t_numCols;
    static const unsigned int k_totalIterations = k_rowIterations * k_colIterations;
    unsigned int outStreamPtr = 0;

MUX_STREAMING_LOOP:
    for (int iter = 0; iter < k_totalIterations; iter++) {
#pragma HLS PIPELINE II = 1 rewind

        typename WideTypeDefs<t_memReadWidth, T_elemType>::WideIFType wideIFSample =
            p_inWideStream[outStreamPtr].read();
        p_outWideStream.write(wideIFSample);
#ifdef DEBUG_MUX
        if (outStreamPtr == 1) std::cout << "Mux Stream 0 for kernel 0 : " << wideIFSample << std::endl;
#endif
        outStreamPtr = (outStreamPtr + 1) % t_numOfKernels;
    }
}

} // end namespace fft
} // end namespace dsp
} // end namespace xf

#endif //__HLS_SSR_FFT_MUX_H__
