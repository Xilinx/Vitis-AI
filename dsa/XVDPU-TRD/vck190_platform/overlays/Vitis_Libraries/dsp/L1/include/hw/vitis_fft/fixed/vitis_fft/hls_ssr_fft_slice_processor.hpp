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
 * @file hls_ssr_fft_kernel_array.h
 * @brief header, describes DDR interface plugin for SS FFT kernel
 *
 */

#define __HLS_SSR_FFT_LOCAL_LIB_DEVLOPMENT_PROJECT__

#ifndef __HLS_SSR_FFT_KERNEL_ARRAY_H__
#define __HLS_SSR_FFT_KERNEL_ARRAY_H__

#include <hls_stream.h>

#include "vitis_fft/hls_ssr_fft_kernel_types.hpp"
#include "vitis_fft/hls_ssr_fft_streaming_kernel.hpp"
#include "vitis_fft/hls_ssr_fft_enums.hpp"
#include "vitis_fft/hls_ssr_fft_types.hpp"
#include "vitis_fft/hls_ssr_fft_demux.hpp"
#include "vitis_fft/hls_ssr_fft_mux.hpp"

namespace xf {
namespace dsp {
namespace fft {

template <unsigned int t_memWidth, // ocm width in no. of complex elements
          unsigned int t_numRows,
          unsigned int t_numCols,
          typename t_ssrFFTParams, // 1-d kernel parameter structure
          unsigned int t_instanceIDOffset,
          typename T_elemType // complex element type
          >
class FFTMemWideSliceProcessor {
   public:
    // Input side IF and stream types
    typedef typename WideTypeDefs<t_memWidth, T_elemType>::WideIFType MemWideIFTypeIn;
    typedef typename WideTypeDefs<t_memWidth, T_elemType>::WideIFStreamType MemWideIFStreamTypeIn;
    // Output side IF and stream types
    typedef typename FFTIOTypes<t_ssrFFTParams, T_elemType>::T_outType T_outType;
    typedef typename WideTypeDefs<t_memWidth, T_outType>::WideIFType MemWideIFTypeOut;
    typedef typename WideTypeDefs<t_memWidth, T_outType>::WideIFStreamType MemWideIFStreamTypeOut;

    static const unsigned int k_numOfKernels = t_memWidth / t_ssrFFTParams::R;
    void sliceProcessor(MemWideIFStreamTypeIn& p_wideMemWideStreamIn, MemWideIFStreamTypeOut& p_wideMemWideStreamOut) {
#pragma HLS INLINE
#pragma HLS DATAFLOW
        MemWideIFStreamTypeIn demuxStreamArrayOut[k_numOfKernels];
        MemWideIFStreamTypeOut sliceProcesorStreamArrayOut[k_numOfKernels];

        // std::cout <<"Demux input Fifo Size Before call : " << p_wideMemWideStreamIn.size()<<std::endl;

        demuxWideStreaming<k_numOfKernels, t_numRows, t_numCols, t_memWidth, T_elemType>(p_wideMemWideStreamIn,
                                                                                         demuxStreamArrayOut);
        /*std::cout <<"Demux input Fifo Size After call : " << p_wideMemWideStreamIn.size()<<std::endl;
        /std::cout << "Demux output array sizes ...." << std::endl;
        for (int gh = 0; gh < k_numOfKernels; ++gh) {
          std::cout<<(demuxStreamArrayOut[gh]).size()<<std::endl;
        }*/
        // exit(1);
        FFTMemWideKernel1DArray<k_numOfKernels, t_memWidth, t_numRows, t_numCols, t_ssrFFTParams, t_instanceIDOffset,
                                T_elemType>::genMemWideFFTKernel1DArray(demuxStreamArrayOut,
                                                                        sliceProcesorStreamArrayOut);

        muxWideStreaming<k_numOfKernels, t_numRows, t_numCols, t_memWidth, T_outType>(sliceProcesorStreamArrayOut,
                                                                                      p_wideMemWideStreamOut);
    }
};

} // end namespace fft
} // end namespace dsp
} // end namespace xf

#endif
