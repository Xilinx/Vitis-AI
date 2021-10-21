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
 * @file hls_ssr_fft_2d.hpp
 * @brief XF FFT 2D kernel top level module
 *
 * This file is part of XF FFT Library.
 */

#ifndef __HLS_SSR_FFT_FFT_2D__
#define __HLS_SSR_FFT_FFT_2D__

#ifndef __SYNTHESIS__
#include <assert.h>
#include <iostream>
#endif

#include "vitis_fft/fft_complex.hpp"
#include "vitis_fft/hls_ssr_fft_slice_processor.hpp"
#include "vitis_fft/hls_ssr_fft_matrix_commutors.hpp"
#define _HLS_SSR_FFT_2D_FIFO_SIZE__ 8

namespace xf {
namespace dsp {
namespace fft {

/**
 * @brief FFT2d class for 2d FFT kernel, provides DFT/Inverse DFT of 2-d data
 *
 * @tparam t_memWidth number of complex<float>==64bits streaming into the kernel in parallel
 * @tparam t_numRows number of rows in complex 2-d input matrix
 * @tparam t_numCols number of columns in complex 2-d input matrix
 * @param t_ssrFFTParamsRowProc gives parameters for 1-d fft kernel used on rows
 * @param t_ssrFFTParamsColProc gives parameters for 1-d fft kernel used on columns
 * @param t_colInstanceIDOffset, uniquefy row vs col kernel,  significantly different
 * @param t_rowInstanceIDOffset, uniquefy row vs col kernel,  significantly different
 * @tparam T_elemType data type of the individual matrix elements
 *
 */

template <unsigned int t_memWidth,
          unsigned int t_numRows,
          unsigned int t_numCols,
          unsigned int t_numKernels,
          typename t_ssrFFTParamsRowProc, // 1-d row kernels parameter structure
          typename t_ssrFFTParamsColProc, // 1-d col kernels parameter structure
          unsigned int t_rowInstanceIDOffset,
          unsigned int t_colInstanceIDOffset,
          typename T_elemType // complex element type
          >
struct FFT2d {
    // To do : Write trait type for fixed point which can calculate col-processor
    // output type given input type and different scaling and rounding params

    typedef T_elemType T_elemType_row;
    // typedef T_elemType T_elemType_col;
    typedef typename FFTIOTypes<t_ssrFFTParamsRowProc, T_elemType_row>::T_outType T_elemType_col;
    // typedef typename FFTIOTypes<t_ssrFFTParamsRowProc, T_elemType_row_out>::T_outType T_elemType_col_out;

    typedef typename FFTIOTypes<t_ssrFFTParamsRowProc, T_elemType_row>::T_outType T_outType_row;
    typedef typename FFTIOTypes<t_ssrFFTParamsColProc, T_outType_row>::T_outType T_outType_col;

    typedef typename WideTypeDefs<t_memWidth, T_elemType_row>::WideIFType MemWideIFTypeIn_row;
    typedef typename WideTypeDefs<t_memWidth, T_elemType_col>::WideIFType MemWideIFTypeIn_col;

    typedef typename WideTypeDefs<t_memWidth, T_elemType_row>::WideIFStreamType MemWideIFStreamTypeIn_row;
    typedef typename WideTypeDefs<t_memWidth, T_elemType_col>::WideIFStreamType MemWideIFStreamTypeIn_Col;

    typedef typename WideTypeDefs<t_memWidth, T_outType_row>::WideIFType MemWideIFTypeOut_row;
    typedef typename WideTypeDefs<t_memWidth, T_outType_row>::WideIFStreamType MemWideIFStreamTypeOut_row;

    typedef typename WideTypeDefs<t_memWidth, T_outType_col>::WideIFType MemWideIFTypeOut_col;
    typedef typename WideTypeDefs<t_memWidth, T_outType_col>::WideIFStreamType MemWideIFStreamTypeOut_col;

    void fft2dProc(MemWideIFStreamTypeIn_row& p_memWideStreamIn, MemWideIFStreamTypeOut_col& p_memWideStreamOut) {
//#pragma HLS INLINE
#ifndef __SYNTHESIS__
        assert((t_ssrFFTParamsRowProc::R) == (t_ssrFFTParamsColProc::R));
        assert(t_ssrFFTParamsRowProc::R > 2);
        assert(t_ssrFFTParamsRowProc::R * t_numKernels == t_memWidth);
        assert(t_numRows == t_numCols);
        assert((t_rowInstanceIDOffset - t_colInstanceIDOffset) > t_numKernels);
#endif

        // block wise tranposer output stream
        MemWideIFStreamTypeIn_row l_transBlkMatrixStream;
//#pragma HLS DATA_PACK variable = l_transBlkMatrixStream
#pragma HLS STREAM variable = l_transBlkMatrixStream depth = 2 // dim = 1
#pragma HLS RESOURCE variable = l_transBlkMatrixStream core = FIFO_LUTRAM
        // row processor output stream
        MemWideIFStreamTypeOut_row l_rowProcOutStream;
//#pragma HLS DATA_PACK variable = l_rowProcOutStream
#pragma HLS STREAM variable = l_rowProcOutStream depth = 2 // dim = 1
#pragma HLS RESOURCE variable = l_rowProcOutStream core = FIFO_LUTRAM

        // inverse blk wise tranposer output stream
        MemWideIFStreamTypeOut_row l_invTranspBlkMatrixStream;
//#pragma HLS DATA_PACK variable = l_invTranspBlkMatrixStream
#pragma HLS STREAM variable = l_invTranspBlkMatrixStream depth = 2 // dim = 1
#pragma HLS RESOURCE variable = l_invTranspBlkMatrixStream core = FIFO_LUTRAM

        MemWideIFStreamTypeOut_row l_transpMatrixStream;
//#pragma HLS DATA_PACK variable = l_transpMatrixStream
#pragma HLS STREAM variable = l_transpMatrixStream depth = 2 // dim = 1
#pragma HLS RESOURCE variable = l_transpMatrixStream core = FIFO_LUTRAM

        MemWideIFStreamTypeOut_row l_transpBlkMatrixStream2;
//#pragma HLS DATA_PACK variable = l_transpBlkMatrixStream2
#pragma HLS STREAM variable = l_transpBlkMatrixStream2 depth = 2 // dim = 1
#pragma HLS RESOURCE variable = l_transpBlkMatrixStream2 core = FIFO_LUTRAM

        MemWideIFStreamTypeOut_col l_colProcOutStream;
//#pragma HLS DATA_PACK variable = l_colProcOutStream
#pragma HLS STREAM variable = l_colProcOutStream depth = 2 // dim = 1
#pragma HLS RESOURCE variable = l_colProcOutStream core = FIFO_LUTRAM

        MemWideIFStreamTypeOut_col l_invTranspBlkMatrixStream2;
//#pragma HLS DATA_PACK variable = l_invTranspBlkMatrixStream2
#pragma HLS STREAM variable = l_invTranspBlkMatrixStream2 depth = 2 // dim = 1
#pragma HLS RESOURCE variable = l_invTranspBlkMatrixStream2 core = FIFO_LUTRAM

        FFTMemWideSliceProcessor<t_memWidth, t_numRows, t_numCols, t_numKernels, t_ssrFFTParamsRowProc,
                                 t_rowInstanceIDOffset, T_elemType_row>
            l_rowProcObj;
        FFTMemWideSliceProcessor<t_memWidth, t_numRows, t_numCols, t_numKernels, t_ssrFFTParamsColProc,
                                 t_colInstanceIDOffset, T_elemType_col>
            l_colProcObj;

#pragma HLS DATAFLOW // disable_start_propagation
        //#pragma HLS interface ap_ctrl_none port = return

        // Perform transpose on blocks of data size : t_numKernel x t_numCols
        // Transpose is done block-wise on whole matrix
        transpWideBlksInMatrix<t_numRows, t_numCols, t_numKernels, t_memWidth, T_elemType_row>(p_memWideStreamIn,
                                                                                               l_transBlkMatrixStream);

        // perform row wise fft
        l_rowProcObj.sliceProcessor(l_transBlkMatrixStream, l_rowProcOutStream);

        // perform inverse transpose on sub-matrices
        invTranspWideBlksInMatrix<t_numRows, t_numCols, t_numKernels, t_memWidth, T_outType_row>(
            l_rowProcOutStream, l_invTranspBlkMatrixStream);

        // perform full matrix tranpose
        transpMemBlocks<1, t_numRows, t_numCols, t_memWidth, T_outType_row>(l_invTranspBlkMatrixStream,
                                                                            l_transpMatrixStream);

        // perform blockwise transpose
        transpWideBlksInMatrix<t_numRows, t_numCols, t_numKernels, t_memWidth, T_outType_row>(l_transpMatrixStream,
                                                                                              l_transpBlkMatrixStream2);

        // perform col wise fft
        l_colProcObj.sliceProcessor(l_transpBlkMatrixStream2, l_colProcOutStream);

        // perform inverse block transpose
        invTranspWideBlksInMatrix<t_numRows, t_numCols, t_numKernels, t_memWidth, T_outType_col>(
            l_colProcOutStream, l_invTranspBlkMatrixStream2);

        // block inverser full matrix transpose
        transpMemBlocks<1, t_numRows, t_numCols, t_memWidth, T_outType_col>(l_invTranspBlkMatrixStream2,
                                                                            p_memWideStreamOut);
    }
};

/**
 * @brief FFT2d class for 2d FFT kernel, provides DFT/Inverse DFT of 2-d data
 *
 * @tparam t_memWidth number of complex<float>==64bits streaming into the kernel in parallel
 * @tparam t_numRows number of rows in complex 2-d input matrix
 * @tparam t_numCols number of columns in complex 2-d input matrix
 * @param t_ssrFFTParamsRowProc gives parameters for 1-d fft kernel used on rows
 * @param t_ssrFFTParamsColProc gives parameters for 1-d fft kernel used on columns
 * @param t_colInstanceIDOffset, uniquefy row vs col kernel,  significantly different
 * @param t_rowInstanceIDOffset, uniquefy row vs col kernel,  significantly different
 * @tparam T_elemType data type of the individual matrix elements
 *
 */

template <unsigned int t_memWidth,
          unsigned int t_numRows,
          unsigned int t_numCols,
          typename t_ssrFFTParamsRowProc, // 1-d row kernels parameter structure
          typename t_ssrFFTParamsColProc, // 1-d col kernels parameter structure
          unsigned int t_rowInstanceIDOffset,
          unsigned int t_colInstanceIDOffset,
          typename T_elemType // complex element type
          >
struct FFT2d<t_memWidth,
             t_numRows,
             t_numCols,
             1,
             t_ssrFFTParamsRowProc,
             t_ssrFFTParamsColProc,
             t_rowInstanceIDOffset,
             t_colInstanceIDOffset,
             T_elemType> {
    // To do : Write trait type for fixed point which can calculate col-processor
    // output type given input type and different scaling and rounding params

    typedef T_elemType T_elemType_row;
    typedef T_elemType T_elemType_col;

    typedef typename FFTIOTypes<t_ssrFFTParamsRowProc, T_elemType_row>::T_outType T_outType_row;
    typedef typename FFTIOTypes<t_ssrFFTParamsColProc, T_elemType_col>::T_outType T_outType_col;

    typedef typename WideTypeDefs<t_memWidth, T_elemType_row>::WideIFType MemWideIFTypeIn_row;
    typedef typename WideTypeDefs<t_memWidth, T_elemType_col>::WideIFType MemWideIFTypeIn_col;

    typedef typename WideTypeDefs<t_memWidth, T_elemType_row>::WideIFStreamType MemWideIFStreamTypeIn_row;
    typedef typename WideTypeDefs<t_memWidth, T_elemType_col>::WideIFStreamType MemWideIFStreamTypeIn_Col;

    typedef typename WideTypeDefs<t_memWidth, T_outType_row>::WideIFType MemWideIFTypeOut_row;
    typedef typename WideTypeDefs<t_memWidth, T_outType_row>::WideIFStreamType MemWideIFStreamTypeOut_row;

    typedef typename WideTypeDefs<t_memWidth, T_outType_col>::WideIFType MemWideIFTypeOut_col;
    typedef typename WideTypeDefs<t_memWidth, T_outType_col>::WideIFStreamType MemWideIFStreamTypeOut_col;

    void fft2dProc(MemWideIFStreamTypeIn_row& p_memWideStreamIn, MemWideIFStreamTypeOut_col& p_memWideStreamOut) {
#pragma HLS INLINE
#ifndef __SYNTHESIS__
        assert((t_ssrFFTParamsRowProc::R) == (t_ssrFFTParamsColProc::R));
        assert(t_ssrFFTParamsRowProc::R > 2);
        assert(t_ssrFFTParamsRowProc::R * 1 == t_memWidth);
        assert(t_numRows == t_numCols);
        assert((t_rowInstanceIDOffset - t_colInstanceIDOffset) > 1);
#endif

        // row processor output stream
        MemWideIFStreamTypeOut_row l_rowProcOutStream;
#pragma HLS DATA_PACK variable = l_rowProcOutStream
#pragma HLS STREAM variable = l_rowProcOutStream depth = 2 dim = 1
#pragma HLS RESOURCE variable = l_rowProcOutStream core = FIFO_LUTRAM

        MemWideIFStreamTypeOut_row l_transpMatrixStream;
#pragma HLS DATA_PACK variable = l_transpMatrixStream
#pragma HLS STREAM variable = l_transpMatrixStream depth = 2 dim = 1
#pragma HLS RESOURCE variable = l_transpMatrixStream core = FIFO_LUTRAM

        MemWideIFStreamTypeOut_col l_colProcOutStream;
#pragma HLS DATA_PACK variable = l_colProcOutStream
#pragma HLS STREAM variable = l_colProcOutStream depth = 2 dim = 1
#pragma HLS RESOURCE variable = l_colProcOutStream core = FIFO_LUTRAM

        FFTMemWideSliceProcessor<t_memWidth, t_numRows, t_numCols, 1, t_ssrFFTParamsRowProc, t_rowInstanceIDOffset,
                                 T_elemType_row>
            l_rowProcObj;
        FFTMemWideSliceProcessor<t_memWidth, t_numRows, t_numCols, 1, t_ssrFFTParamsColProc, t_colInstanceIDOffset,
                                 T_elemType_col>
            l_colProcObj;

#pragma HLS DATAFLOW disable_start_propagation
#pragma HLS interface ap_ctrl_none port = return

        // perform row wise fft
        l_rowProcObj.sliceProcessor(p_memWideStreamIn, l_rowProcOutStream);

        // perform full matrix transpose
        transpMemBlocks<1, t_numRows, t_numCols, t_memWidth, T_outType_row>(l_rowProcOutStream, l_transpMatrixStream);

        // perform col wise fft
        l_colProcObj.sliceProcessor(l_transpMatrixStream, l_colProcOutStream);

        // full matrix inverse transpose
        transpMemBlocks<1, t_numRows, t_numCols, t_memWidth, T_outType_col>(l_colProcOutStream, p_memWideStreamOut);
    }
};

template <unsigned int t_memWidth,
          unsigned int t_numRows,
          unsigned int t_numCols,
          unsigned int t_numKernels,
          typename t_ssrFFTParamsRowProc, // 1-d row kernels parameter structure
          typename t_ssrFFTParamsColProc, // 1-d col kernels parameter structure
          unsigned int t_rowInstanceIDOffset,
          unsigned int t_colInstanceIDOffset,
          typename T_elemType // complex element type
          >
void fft2d(typename WideTypeDefs<t_memWidth, T_elemType>::WideIFStreamType& p_memWideStreamIn,
           typename WideTypeDefs<
               t_memWidth,
               typename FFTIOTypes<t_ssrFFTParamsColProc,
                                   typename FFTIOTypes<t_ssrFFTParamsRowProc, T_elemType>::T_outType>::T_outType

               >::WideIFStreamType& p_memWideStreamOut)

{
    //#pragma HLS INLINE
    FFT2d<t_memWidth, t_numRows, t_numCols, t_numKernels, t_ssrFFTParamsRowProc, t_ssrFFTParamsColProc,
          t_rowInstanceIDOffset, t_colInstanceIDOffset, T_elemType>
        obj_fft2d;
    obj_fft2d.fft2dProc(p_memWideStreamIn, p_memWideStreamOut);
}

} // end namespace fft
} // end namespace dsp
} // end namespace xf

#endif //__HLS_SSR_FFT_FFT_2D__
