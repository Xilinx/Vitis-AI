/*
 * Copyright 2021 Xilinx, Inc.
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

#ifndef FFT_KERNEL_H_
#define FFT_KERNEL_H_

#ifndef __SYNTHESIS__
#include <assert.h>
#endif
#include <hls_stream.h>
#include <ap_int.h>

#include "vitis_fft/hls_ssr_fft_2d.hpp"

namespace xf {
namespace dsp {
namespace fft {

template <unsigned int t_memWidthBits,
          unsigned int t_rows,
          unsigned int t_cols,
          unsigned int t_images,
          unsigned int t_wordWidth,
          unsigned int t_memWidth,
          typename T_elemType,
          typename T_streamDataType>
void readImages(ap_uint<t_memWidthBits> p_fftInData[t_rows * t_cols * t_images / t_memWidth],
                hls::stream<T_streamDataType>& p_stream,
                int n_images) {
    T_streamDataType l_inMemWideSample;
    for (int n = 0; n < n_images; n++) {
        for (int r = 0; r < t_rows; ++r) {
            for (int c = 0; c < t_cols / t_memWidth; ++c) {
#pragma HLS pipeline II = 1
                ap_uint<t_memWidthBits> superSample =
                    p_fftInData[n * t_rows * t_cols / t_memWidth + r * t_cols / t_memWidth + c];
                for (int w = 0; w < t_memWidth; ++w) {
#pragma HLS unroll
                    T_elemType singleSample;
                    singleSample.real(superSample.range(t_wordWidth - 1 + 64 * w, 64 * w));
                    singleSample.imag(superSample.range(t_wordWidth + 31 + 64 * w, 32 + 64 * w));
                    l_inMemWideSample[w] = singleSample;
                }
                p_stream.write(l_inMemWideSample);
            }
        }
    }
}

template <unsigned int t_memWidthBits,
          unsigned int t_rows,
          unsigned int t_cols,
          unsigned int t_images,
          unsigned int t_wordWidth,
          unsigned int t_memWidth,
          typename T_elemType,
          typename T_streamDataType>
void writeImages(hls::stream<T_streamDataType>& p_inStream,
                 ap_uint<t_memWidthBits> p_fftOutData[t_rows * t_cols * t_images / t_memWidth],
                 int n_images) {
    T_streamDataType l_inMemWideSample;
    for (int n = 0; n < n_images; n++) {
        for (int r = 0; r < t_rows; ++r) {
            for (int c = 0; c < t_cols / t_memWidth; ++c) {
#pragma HLS pipeline II = 1
                ap_uint<t_memWidthBits> singleSuperSample;
                l_inMemWideSample = p_inStream.read();
                for (int w = 0; w < t_memWidth; ++w) {
#pragma HLS unroll
                    singleSuperSample.range(t_wordWidth - 1 + 64 * w, 64 * w) = l_inMemWideSample[w].real();
                    singleSuperSample.range(t_wordWidth + 31 + 64 * w, 32 + 64 * w) = l_inMemWideSample[w].imag();
                }
                p_fftOutData[n * t_rows * t_cols / t_memWidth + r * t_cols / t_memWidth + c] = singleSuperSample;
            }
        }
    }
}

template <unsigned int t_memWidthBits,
          unsigned int t_memWidth,
          unsigned int t_numRows,
          unsigned int t_numCols,
          unsigned int t_numImages,
          unsigned int t_wordWidthBits,
          unsigned int t_growWidthBits,
          unsigned int t_numKernels,
          typename t_ssrFFTParamsRowProc, // 1-d row kernels parameter structure
          typename t_ssrFFTParamsColProc, // 1-d col kernels parameter structure
          unsigned int t_rowInstanceIDOffset,
          unsigned int t_colInstanceIDOffset,
          typename T_elemType, // complex input type
          typename T_outType   // complex output type
          >
void fft2dKernel(ap_uint<t_memWidthBits> p_fftInData[t_numRows * t_numCols * t_numImages / t_memWidth],
                 ap_uint<t_memWidthBits> p_fftOutData[t_numRows * t_numCols * t_numImages / t_memWidth],
                 int n_images) {
    enum { FIFO_SIZE = 2 * t_memWidth };
#pragma HLS dataflow

#ifndef __SYNTHESIS__
    assert(t_wordWidthBits < 32); // if bit width of each element is wider than 32, then float datatype should be used
#endif

    hls::stream<SuperSampleContainer<t_memWidth, T_elemType> > fftInStrm;
#pragma HLS stream variable = fftInStrm depth = FIFO_SIZE
    hls::stream<SuperSampleContainer<t_memWidth, T_outType> > fftOutStrm;
#pragma HLS stream variable = fftOutStrm depth = FIFO_SIZE

    readImages<t_memWidthBits, t_numRows, t_numCols, t_numImages, t_wordWidthBits, t_memWidth, T_elemType,
               SuperSampleContainer<t_memWidth, T_elemType> >(p_fftInData, fftInStrm, n_images);
    FFT2d<t_memWidth, t_numRows, t_numCols, t_numKernels, t_ssrFFTParamsRowProc, t_ssrFFTParamsColProc,
          t_rowInstanceIDOffset, t_colInstanceIDOffset, T_elemType>
        obj_fft2d;
    for (int n = 0; n < n_images; n++) {
        obj_fft2d.fft2dProc(fftInStrm, fftOutStrm);
    }
    writeImages<t_memWidthBits, t_numRows, t_numCols, t_numImages, t_growWidthBits, t_memWidth, T_elemType,
                SuperSampleContainer<t_memWidth, T_outType> >(fftOutStrm, p_fftOutData, n_images);
}

} // namespace fft
} // namespace dsp
} // namespace xf

#endif // !FFT_KERNEL_H_
