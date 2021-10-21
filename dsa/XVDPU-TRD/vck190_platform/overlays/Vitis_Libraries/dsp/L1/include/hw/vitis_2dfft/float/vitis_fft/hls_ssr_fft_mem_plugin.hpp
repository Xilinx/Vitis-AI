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
 * @file hls_ssr_fft_mem_plugin.hpp
 * @brief header, describes DDR interface plugin for SS FFT kernel
 *
 */

#ifndef __HLS_SSR_FFT_MEM_PLUGIN_H__
#define __HLS_SSR_FFT_MEM_PLUGIN_H__

#include <hls_stream.h>

#include "vitis_fft/hls_ssr_fft_kernel_types.hpp"

namespace xf {
namespace dsp {
namespace fft {

template <unsigned int t_numBatches,
          unsigned int t_numRows,
          unsigned int t_numCols,
          unsigned int t_memReadWidth,
          typename T_elemType>
void memReadPlugin(typename WideTypeDefs<t_memReadWidth, T_elemType>::WideIFType* p_memIn,
                   typename WideTypeDefs<t_memReadWidth, T_elemType>::WideIFStreamType& p_OutWideStream) {
    const unsigned int k_numOfWideSample = (t_numBatches * t_numRows / t_memReadWidth * t_numCols);
MEM_READ_PLUGIN_LOOP:
    for (int s = 0; s < k_numOfWideSample; ++s) {
#pragma HLS PIPELINE II = 1
        p_OutWideStream.write(p_memIn[s]);
    }
}

template <unsigned int t_numBatches,
          unsigned int t_numRows,
          unsigned int t_numCols,
          unsigned int t_memReadWidth,
          typename T_elemType>
void memWritePlugin(typename WideTypeDefs<t_memReadWidth, T_elemType>::WideIFStreamType& p_InWideStream,
                    typename WideTypeDefs<t_memReadWidth, T_elemType>::WideIFType* p_memOut)

{
    const unsigned int k_numOfWideSample = (t_numBatches * t_numRows / t_memReadWidth * t_numCols);
MEM_WRITE_PLUGIN_LOOP:
    for (int s = 0; s < k_numOfWideSample; ++s) {
#pragma HLS PIPELINE II = 1
        p_memOut[s] = p_InWideStream.read();
    }
}

} // end namespace fft
} // end namespace dsp
} // end namespace xf

#endif
