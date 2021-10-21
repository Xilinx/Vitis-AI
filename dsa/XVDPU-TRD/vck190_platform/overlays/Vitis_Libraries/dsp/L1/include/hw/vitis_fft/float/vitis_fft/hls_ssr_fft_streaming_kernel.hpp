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
#ifndef __HLS_SSR_FFT_STREAMING_KERNEL_H__
#define __HLS_SSR_FFT_STREAMING_KERNEL_H__
#ifndef __SYNTHESIS__
#include <assert.h>
#endif
#include <complex>
#include <hls_stream.h>

#include "vitis_fft/hls_ssr_fft.hpp"
#include "vitis_fft/hls_ssr_fft_kernel_types.hpp"
#include "vitis_fft/hls_ssr_fft_types.hpp"
#include "vitis_fft/hls_ssr_fft_enums.hpp"
#include "vitis_fft/hls_ssr_fft_fork_merge_utils.hpp"
#include "vitis_fft/hls_ssr_fft_wide_type_utilities.hpp"

namespace xf {
namespace dsp {
namespace fft {

template <typename ssr_fft_param_struct, int t_instanceID, typename T_in>
class FFTStreamingKernelClass {
   public:
    typedef typename FFTIOTypes<ssr_fft_param_struct, T_in>::T_outType T_outType;
    typedef typename WideTypeDefs<ssr_fft_param_struct::R, T_outType>::WideIFStreamType WideIFStreamTypeOut;
    typedef typename WideTypeDefs<ssr_fft_param_struct::R, T_in>::WideIFStreamType WideIFStreamTypeIn;

    void fftStreamingKernel(WideIFStreamTypeIn& p_wideStreamIn, WideIFStreamTypeOut& p_wideStreamOut) {
#pragma HLS INLINE
#pragma HLS DATAFLOW
#pragma HLS DATA_PACK variable = p_wideStreamIn
#pragma HLS DATA_PACK variable = p_wideStreamOut
        T_in p_inDataArray[ssr_fft_param_struct::R][ssr_fft_param_struct::N / ssr_fft_param_struct::R];
        T_outType p_outDataArray[ssr_fft_param_struct::R][ssr_fft_param_struct::N / ssr_fft_param_struct::R];
        convertSuperStreamToArray<-1, t_instanceID, ssr_fft_param_struct::N, ssr_fft_param_struct::R>(p_wideStreamIn,
                                                                                                      p_inDataArray);
#ifdef DEBUG_FFT_STREAMING_KERNEL
#ifndef __SYNTHESIS__
        std::cout << "The input data for kernel : " << t_instanceID << std::endl;
        for (int t = 0; t < (ssr_fft_param_struct::N / ssr_fft_param_struct::R); ++t) {
            for (int r = 0; r < ssr_fft_param_struct::R; ++r) {
                std::cout << p_inDataArray[r][t] << "   ";
            }
            std::cout << std::endl;
        }
#endif
#endif
        fft<ssr_fft_param_struct, t_instanceID, T_in>(p_inDataArray, p_outDataArray);
#ifdef DEBUG_FFT_STREAMING_KERNEL
#ifndef __SYNTHESIS__
        std::cout << "The output data for kernel : " << t_instanceID << std::endl;
        for (int t = 0; t < (ssr_fft_param_struct::N / ssr_fft_param_struct::R); ++t) {
            for (int r = 0; r < ssr_fft_param_struct::R; ++r) {
                std::cout << p_outDataArray[r][t] << "   ";
            }
            std::cout << std::endl;
        }
#endif
#endif
        convertArrayToSuperStream<-1, t_instanceID, ssr_fft_param_struct::N, ssr_fft_param_struct::R, T_outType>(
            p_outDataArray, p_wideStreamOut);
    }
};

template <unsigned int t_kernelIndex,
          unsigned int t_numOfKernels,
          typename ssr_fft_param_struct,
          unsigned int t_instanceIDOffset,
          typename T_in>
class FFTStreamingKernel1DArray {
   public:
    typedef typename FFTIOTypes<ssr_fft_param_struct, T_in>::T_outType T_outType;
    typedef typename WideTypeDefs<ssr_fft_param_struct::R, T_outType>::WideIFStreamType WideIFStreamTypeOut;
    typedef typename WideTypeDefs<ssr_fft_param_struct::R, T_in>::WideIFStreamType WideIFStreamTypeIn;
    static const unsigned int this_instace_id = t_instanceIDOffset + t_kernelIndex;
    static void generateFFTKernel(WideIFStreamTypeIn p_inWideStreamArray[t_numOfKernels],
                                  WideIFStreamTypeOut p_outWideStreamArray[t_numOfKernels]) {
#pragma HLS INLINE
#pragma HLS DATAFLOW
        FFTStreamingKernelClass<ssr_fft_param_struct, this_instace_id, T_in> obj;
        obj.fftStreamingKernel(p_inWideStreamArray[t_kernelIndex - 1], p_outWideStreamArray[t_kernelIndex - 1]);
        FFTStreamingKernel1DArray<t_kernelIndex - 1, t_numOfKernels, ssr_fft_param_struct, t_instanceIDOffset,
                                  T_in>::generateFFTKernel(p_inWideStreamArray, p_outWideStreamArray);
    }
};

template <unsigned int t_numOfKernels, typename ssr_fft_param_struct, unsigned int t_instanceIDOffset, typename T_in>
class FFTStreamingKernel1DArray<1, t_numOfKernels, ssr_fft_param_struct, t_instanceIDOffset, T_in> {
   public:
    typedef typename FFTIOTypes<ssr_fft_param_struct, T_in>::T_outType T_outType;
    typedef typename WideTypeDefs<ssr_fft_param_struct::R, T_outType>::WideIFStreamType WideIFStreamTypeOut;
    typedef typename WideTypeDefs<ssr_fft_param_struct::R, T_in>::WideIFStreamType WideIFStreamTypeIn;
    static const unsigned int kernelIndex = 1;
    static const unsigned int this_instace_id = t_instanceIDOffset + kernelIndex;
    static void generateFFTKernel(WideIFStreamTypeIn p_inWideStreamArray[t_numOfKernels],
                                  WideIFStreamTypeOut p_outWideStreamArray[t_numOfKernels]) {
#pragma HLS INLINE
        FFTStreamingKernelClass<ssr_fft_param_struct, this_instace_id, T_in> obj;
        obj.fftStreamingKernel(p_inWideStreamArray[kernelIndex - 1], p_outWideStreamArray[kernelIndex - 1]);
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
 * Wide Streaming Kernle FFT Array Class : This class will generate array
 * of SSR FFT kernel which is 1-d and it adds adapters to the input interface
 * to make its connection with MEMORY WIDE INTERFACE
 */

template <unsigned int t_kernelIndex,
          unsigned int t_memWidth, // in elements complex<type>
          unsigned int t_numRows,  // in elements complex<type>
          unsigned int t_numCols,  //
          typename ssr_fft_param_struct,
          unsigned int t_instanceIDOffset,
          typename T_in>
class FFTMemWideKernel1DArray {
   public:
    static const unsigned int k_numOfKernels = t_memWidth / ssr_fft_param_struct::R;
    static const unsigned int k_totalWideSamplePerImage = (t_numCols / t_memWidth) * t_numRows;
    static const unsigned int k_totalWideSamplePerKernel = (k_totalWideSamplePerImage / k_numOfKernels);
    typedef typename FFTIOTypes<ssr_fft_param_struct, T_in>::T_outType T_outType;
    typedef typename WideTypeDefs<ssr_fft_param_struct::R, T_outType>::WideIFStreamType KernelWideIFStreamTypeOut;
    typedef typename WideTypeDefs<ssr_fft_param_struct::R, T_in>::WideIFStreamType KernelWideIFStreamTypeIn;
    typedef typename WideTypeDefs<t_memWidth, T_outType>::WideIFStreamType MemWideIFStreamTypeOut;
    typedef typename WideTypeDefs<t_memWidth, T_in>::WideIFStreamType MemWideIFStreamTypeIn;
    static const unsigned int this_instace_id = t_instanceIDOffset + t_kernelIndex;
    static void genMemWideFFTKernel1DArray(MemWideIFStreamTypeIn p_inMemWideStreamArray[k_numOfKernels],
                                           MemWideIFStreamTypeOut p_outMemWideStreamArray[k_numOfKernels]) {
#pragma HLS INLINE
#pragma HLS DATAFLOW

#ifndef __SYNTHESIS__
        assert((t_memWidth % ssr_fft_param_struct::R) == 0); // Memory bandwidth should be enough for kernels
        assert(t_kernelIndex <= k_numOfKernels);
        assert(t_kernelIndex > 0);
        assert(t_instanceIDOffset >
               k_numOfKernels); // Make sure no duplicate kernel IDs between row and col processors.
#endif
        // streams for connecting input/output data stream to kernel
        KernelWideIFStreamTypeIn ssrWideStream4kernelIn;
        KernelWideIFStreamTypeOut ssrWideStream4kernelOut;

#pragma HLS DATA_PACK variable = ssrWideStream4kernelIn
#pragma HLS DATA_PACK variable = ssrWideStream4kernelOut
        // create instance of wide2ssr utility function
        wideToNarrowConverter<t_memWidth, ssr_fft_param_struct::R, k_totalWideSamplePerKernel, T_in>(
            p_inMemWideStreamArray[t_kernelIndex - 1], ssrWideStream4kernelIn);

        FFTStreamingKernelClass<ssr_fft_param_struct, this_instace_id, T_in> obj;
        for (int fftn = 0; fftn < t_numRows / k_numOfKernels; ++fftn) {
            obj.fftStreamingKernel(ssrWideStream4kernelIn, ssrWideStream4kernelOut);
        }
        // utility function for converting ssrWide stream to memWide stream
        narrowToWideConverter<ssr_fft_param_struct::R, t_memWidth, k_totalWideSamplePerKernel, T_outType>(
            ssrWideStream4kernelOut, p_outMemWideStreamArray[t_kernelIndex - 1]);

        FFTMemWideKernel1DArray<t_kernelIndex - 1, t_memWidth, t_numRows, t_numCols, ssr_fft_param_struct,
                                t_instanceIDOffset, T_in>::genMemWideFFTKernel1DArray(p_inMemWideStreamArray,
                                                                                      p_outMemWideStreamArray);
    }
};

// Specialization to terminate recursion
template <
    // unsigned int t_kernelIndex=1
    unsigned int t_memWidth, // in elements complex<type>
    unsigned int t_numRows,  // in elements complex<type>
    unsigned int t_numCols,  //
    typename ssr_fft_param_struct,
    unsigned int t_instanceIDOffset,
    typename T_in>
class FFTMemWideKernel1DArray<1, t_memWidth, t_numRows, t_numCols, ssr_fft_param_struct, t_instanceIDOffset, T_in> {
   public:
    static const unsigned int t_kernelIndex = 1;
    static const unsigned int k_numOfKernels = t_memWidth / ssr_fft_param_struct::R;
    static const unsigned int k_totalWideSamplePerImage = (t_numCols / t_memWidth) * t_numRows;
    static const unsigned int k_totalWideSamplePerKernel = (k_totalWideSamplePerImage / k_numOfKernels);
    typedef typename FFTIOTypes<ssr_fft_param_struct, T_in>::T_outType T_outType;
    typedef typename WideTypeDefs<ssr_fft_param_struct::R, T_outType>::WideIFStreamType KernelWideIFStreamTypeOut;
    typedef typename WideTypeDefs<ssr_fft_param_struct::R, T_in>::WideIFStreamType KernelWideIFStreamTypeIn;
    typedef typename WideTypeDefs<t_memWidth, T_outType>::WideIFStreamType MemWideIFStreamTypeOut;
    typedef typename WideTypeDefs<t_memWidth, T_in>::WideIFStreamType MemWideIFStreamTypeIn;
    static const unsigned int this_instace_id = t_instanceIDOffset + t_kernelIndex;
    static void genMemWideFFTKernel1DArray(MemWideIFStreamTypeIn p_inMemWideStreamArray[k_numOfKernels],
                                           MemWideIFStreamTypeOut p_outMemWideStreamArray[k_numOfKernels]) {
#pragma HLS INLINE
#pragma HLS DATAFLOW

#ifndef __SYNTHESIS__
        assert((t_memWidth % ssr_fft_param_struct::R) == 0); // Memory bandwidth should be enough for kernels
        assert(t_kernelIndex <= k_numOfKernels);
        assert(t_kernelIndex > 0);
        assert(t_instanceIDOffset >
               k_numOfKernels); // Make sure no duplicate kernel IDs between row and col processors.
#endif
        // streams for connecting input/output data stream to kernel
        KernelWideIFStreamTypeIn ssrWideStream4kernelIn;
        KernelWideIFStreamTypeOut ssrWideStream4kernelOut;
#pragma HLS DATA_PACK variable = ssrWideStream4kernelIn
#pragma HLS DATA_PACK variable = ssrWideStream4kernelOut
        // create instance of wide2ssr utility function
        wideToNarrowConverter<t_memWidth, ssr_fft_param_struct::R, k_totalWideSamplePerKernel, T_in>(
            p_inMemWideStreamArray[t_kernelIndex - 1], ssrWideStream4kernelIn);

        FFTStreamingKernelClass<ssr_fft_param_struct, this_instace_id, T_in> obj;
        for (int fftn = 0; fftn < t_numRows / k_numOfKernels; ++fftn) {
            obj.fftStreamingKernel(ssrWideStream4kernelIn, ssrWideStream4kernelOut);
        }
        // utility function for converting ssrWide stream to memWide stream
        narrowToWideConverter<ssr_fft_param_struct::R, t_memWidth, k_totalWideSamplePerKernel, T_outType>(
            ssrWideStream4kernelOut, p_outMemWideStreamArray[t_kernelIndex - 1]);
    }
};

} // end namespace fft
} // end namespace dsp
} // end namespace xf
#endif
