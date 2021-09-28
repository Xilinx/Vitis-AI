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
 * @file hls_ssr_fft_demux.hpp
 * @brief header, describes OCM interface block for data demuxing
 *
 */

#ifndef __HLS_SSR_FFT_DEMUX_H__
#define __HLS_SSR_FFT_DEMUX_H__
#include <hls_stream.h>
#include "vitis_fft/hls_ssr_fft_kernel_types.hpp"

namespace xf {
namespace dsp {
namespace fft {

template <unsigned int t_numKernels,
          unsigned int t_numRows,
          unsigned int t_numCols,
          unsigned int t_memWidth,
          /*t_memWidth= No of words of type T_elemType
           * in one wide word read at interface boundary*/
          typename T_elemType>
void demuxWideStreaming(typename WideTypeDefs<t_memWidth, T_elemType>::WideIFStreamType& p_inWideStream,
                        typename WideTypeDefs<t_memWidth, T_elemType>::WideIFStreamType p_outWideStream[t_numKernels]) {
#pragma HLS INLINE off
    //#pragma HLS DATA_PACK variable = p_inWideStream
    //#pragma HLS DATA_PACK variable = p_outWideStream
    //#pragma HLS ARRAY_PARTITION variable = p_outWideStream complete dim = 1

    static const unsigned int k_rowIterations = t_numRows / t_memWidth;
    static const unsigned int k_colIterations = t_numCols;
    static const unsigned int k_totalIterations = k_rowIterations * k_colIterations;
    unsigned int outStreamPtr = 0;

DEMUX_STREAMING_LOOP:
    for (int iter = 0; iter < k_totalIterations; iter++) {
#pragma HLS PIPELINE II = 1 rewind

        typename WideTypeDefs<t_memWidth, T_elemType>::WideIFType wideIFSample = p_inWideStream.read();
#ifdef DEBUG_DEMUX
#ifndef __SYNTHESIS__
        if (outStreamPtr == 0)
            std::cout << "Demux Stream 0 for kernel" << outStreamPtr << " : " << wideIFSample << std::endl;
#endif
#endif
        p_outWideStream[outStreamPtr].write(wideIFSample);
        outStreamPtr = (outStreamPtr + 1) % t_numKernels;
    }
}

} // end namespace fft
} // end namespace dsp
} // end namespace xf
#endif
