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
 * @file hls_ssr_fft_matrix_commutors.hpp
 * @brief XF FFT L1 modules for matrix data commutations
 *
 * This file is part of XF FFT Library.
 */

#ifndef __HLS_SSR_FFT_MATRIX_COMMUTOR_H__
#define __HLS_SSR_FFT_MATRIX_COMMUTOR_H__
#include <hls_stream.h>
#include "vitis_fft/hls_ssr_fft_kernel_types.hpp"
#ifndef __SYNTHESIS__
#include <assert.h>
#endif

namespace xf {
namespace dsp {
namespace fft {

/**
 * @brief transpWideBlksInMatrix transposes matrix blocks with wide elements
 * @tparam T_elemType data type of the matrix entries, inner type for Wide Type
 * @tparam t_memWidth number of entries in one memory word
 * @tparam t_numRows number of rows in the block
 * @tparam t_numCols number of cols in the block
 * @param p_inWideStream input stream of memory words
 * @param p_outWideStream output transposed stream of memory words
 */
template <unsigned int t_numRows,
          unsigned int t_numCols,
          unsigned int t_kernels,
          unsigned int t_memWidth,
          typename T_elemType>
void transpWideBlksInMatrix(typename WideTypeDefs<t_memWidth, T_elemType>::WideIFStreamType& p_inWideStream,
                            typename WideTypeDefs<t_memWidth, T_elemType>::WideIFStreamType& p_outWideStream) {
//#pragma HLS INLINE
#ifndef __SYNTHESIS__
    assert(t_numCols % t_memWidth == 0);
#endif
    typedef typename WideTypeDefs<t_memWidth, T_elemType>::WideIFStreamType WideIFStreamType;
    typedef typename WideTypeDefs<t_memWidth, T_elemType>::WideIFType WideSampleType;
    const int k_numWideBlks = t_numRows / t_kernels;
    const int k_numWideColWords = t_numCols / t_memWidth;
    const int k_numWideRowWords = t_numRows / t_memWidth;
    const int k_numWideRowsInBlk = t_kernels;
    const int k_numWideColsInBlk = k_numWideColWords;
    // Wide sample buffer manifests as pipo of wide
    // elements and stores a matrix block not full
    // matrix
    WideSampleType l_memWidePipoBuff[k_numWideRowsInBlk * k_numWideColsInBlk];
#pragma HLS RESOURCE variable = l_memWidePipoBuff core = XPM_MEMORY uram
    //#pragma HLS DATA_PACK variable = l_memWidePipoBuff

    //#ifndef __SYNTHESIS__
    for (unsigned int blk = 0; blk < k_numWideBlks; ++blk) {
    //#pragma HLS DATAFLOW disable_start_propagation
    //#endif
    // read in row major order///////////////////////////////////////
    transpWideBlksInMatrix_writePIPO:
        for (unsigned int r = 0; r < k_numWideRowsInBlk; r++) {
            for (unsigned int c = 0; c < k_numWideColsInBlk; c++) {
#pragma HLS PIPELINE II = 1 rewind
                WideSampleType l_wideSample = p_inWideStream.read();
                //#pragma HLS DATA_PACK variable = l_wideSample
                l_memWidePipoBuff[r * k_numWideColsInBlk + c] = l_wideSample;
            }
        }
    ////////////////////////////////////////////////////////////////

    // Write in column major order///////////////////////////////////////
    transpWideBlksInMatrix_readPIPO:

        for (unsigned int c = 0; c < k_numWideColsInBlk; ++c) {
            for (unsigned int r = 0; r < k_numWideRowsInBlk; r++) {
#pragma HLS PIPELINE II = 1 rewind
                WideSampleType l_wideSample = l_memWidePipoBuff[r * k_numWideColsInBlk + c];
                //#pragma HLS DATA_PACK variable = l_wideSample
                p_outWideStream.write(l_wideSample);
            }
        }

        //#ifndef __SYNTHESIS__
    }
    //#endif
}

/**
 * @brief InvTranspWideBlksInMatrix  Inverse transposes matrix blocks with wide elements
 * @tparam T_elemType data type of the matrix entries, inner type for Wide Type
 * @tparam t_memWidth number of entries in one memory word
 * @tparam t_numRows number of rows in the block
 * @tparam t_numCols number of cols in the block
 * @param p_inWideStream input stream of memory words
 * @param p_outWideStream output transposed stream of memory words
 */
template <unsigned int t_numRows,
          unsigned int t_numCols,
          unsigned int t_kernels,
          unsigned int t_memWidth,
          typename T_elemType>
void invTranspWideBlksInMatrix(typename WideTypeDefs<t_memWidth, T_elemType>::WideIFStreamType& p_inWideStream,
                               typename WideTypeDefs<t_memWidth, T_elemType>::WideIFStreamType& p_outWideStream) {
//#pragma HLS INLINE

#ifndef __SYNTHESIS__
    assert(t_numCols % t_memWidth == 0);
#endif
    typedef typename WideTypeDefs<t_memWidth, T_elemType>::WideIFStreamType WideIFStreamType;
    typedef typename WideTypeDefs<t_memWidth, T_elemType>::WideIFType WideSampleType;
    const int k_numWideBlks = t_numRows / t_kernels;
    const int k_numWideColWords = t_numCols / t_memWidth;
    const int k_numWideRowWords = t_numRows / t_memWidth;
    const int k_numWideRowsInBlk = t_kernels;
    const int k_numWideColsInBlk = k_numWideColWords;
    // Wide sample buffer manifests as pipo of wide
    // elements and stores a matrix block not full
    // matrix
    WideSampleType l_memWidePipoBuff[k_numWideRowsInBlk * k_numWideColsInBlk];
#pragma HLS RESOURCE variable = l_memWidePipoBuff core = XPM_MEMORY uram
    //#pragma HLS DATA_PACK variable = l_memWidePipoBuff
    //#ifndef __SYNTHESIS__
    for (unsigned int blk = 0; blk < k_numWideBlks; ++blk) {
    //#pragma HLS DATAFLOW disable_start_propagation
    //#pragma HLS DATAFLOW
    //#endif
    // Read/Store in column major order/////////////////////////////////
    invTranspWideBlksInMatrix_writePIPO:
        for (unsigned int c = 0; c < k_numWideColsInBlk; ++c) {
            for (unsigned int r = 0; r < k_numWideRowsInBlk; r++) {
#pragma HLS PIPELINE II = 1 rewind
                WideSampleType l_wideSample = p_inWideStream.read();
                //#pragma HLS DATA_PACK variable = l_wideSample
                l_memWidePipoBuff[r * k_numWideColsInBlk + c] = l_wideSample;
            }
        }
    // write in row major order///////////////////////////////////////
    invTranspWideBlksInMatrix_readPIPO:
        for (unsigned int r = 0; r < k_numWideRowsInBlk; r++) {
            for (unsigned int c = 0; c < k_numWideColsInBlk; c++) {
#pragma HLS PIPELINE II = 1 rewind
                WideSampleType l_wideSample;
                //#pragma HLS DATA_PACK variable = l_wideSample
                l_wideSample = l_memWidePipoBuff[r * k_numWideColsInBlk + c];
                p_outWideStream.write(l_wideSample);
            }
        }
        ////////////////////////////////////////////////////////////////

        //#ifndef __SYNTHESIS__
    }
    //#endif
}

/**
 * @brief transpMemWordBlocks transposes matrix blocks with wide elements
 *
 * @tparam T_elemType data type of the matrix entries, inner type for Wide Type
 * @tparam t_memWidth number of entries in one memory word
 * @tparam t_numRows number of rows in the block
 * @tparam t_numCols number of cols in the block
 *
 * @param t_numBlocks number of blocks
 * @param p_inWideStream input stream of memory words
 * @param p_outWideStream ouput transposed stream of memory words
 */
template <unsigned int t_numBlocks,
          unsigned int t_numRows,
          unsigned int t_numCols,
          unsigned int t_memWidth,
          typename T_elemType>
void transpMemWordBlocks(typename WideTypeDefs<t_memWidth, T_elemType>::WideIFStreamType& p_inWideStream,
                         typename WideTypeDefs<t_memWidth, T_elemType>::WideIFStreamType& p_outWideStream) {
#pragma HLS INLINE
#ifndef __SYNTHESIS__
    assert(t_numCols % t_memWidth == 0);
#endif
    typedef typename WideTypeDefs<t_memWidth, T_elemType>::WideIFStreamType WideIFStreamType;
    typedef typename WideTypeDefs<t_memWidth, T_elemType>::WideIFType WideSampleType;
    static const unsigned int t_ColWords = t_numCols / t_memWidth;
    static const unsigned int t_Size = t_ColWords * t_numRows;

    WideSampleType l_buf[t_Size];
#pragma HLS RESOURCE variable = l_buf core = XPM_MEMORY uram

#pragma HLS DATA_PACK variable = l_buf
#ifndef __SYNTHESIS__
    for (unsigned int b = 0; b < t_numBlocks; ++b) {
#pragma HLS DATAFLOW disable_start_propagation
//#pragma HLS DATAFLOW
#endif
    transpMemWordBlocks_writePIPO:
        for (unsigned int i = 0; i < t_numRows; ++i) {
            for (unsigned int j = 0; j < t_ColWords; ++j) {
#pragma HLS PIPELINE II = 1 rewind
                WideSampleType l_val = p_inWideStream.read();
                l_buf[j * t_numRows + i] = l_val;
            }
        }
    transpMemWordBlocks_readPIPO:
        for (unsigned int i = 0; i < t_Size; ++i) {
#pragma HLS PIPELINE II = 1 rewind
            WideSampleType l_val = l_buf[i];
            p_outWideStream.write(l_val);
        }
#ifndef __SYNTHESIS__
    }
#endif
}
// Element Wide Transpose block accepts MemWide Stream
/********************************************************************
 * @brief transpMemBlocks read data from device memory and transpose the memory block
 *
 * @tparam T_elemType data type of the matrix entries
 * @tparam t_memWidth number of entries in one memory word
 * @tparam t_numRows number of rows in the block
 * @tparam t_numCols number of cols in the block
 *
 * @param t_numBlocks number of blocks
 * @param p_inWideStream input stream of memory words
 * @param p_outWideStream ouput transposed stream of memory words
 ********************************************************************/
template <unsigned int t_numBlocks,
          unsigned int t_numRows,
          unsigned int t_numCols,
          unsigned int t_memWidth,
          typename T_elemType>
void transpMemBlocks(typename WideTypeDefs<t_memWidth, T_elemType>::WideIFStreamType& p_inWideStream,
                     typename WideTypeDefs<t_memWidth, T_elemType>::WideIFStreamType& p_outWideStream) {
//#pragma HLS INLINE
#ifndef __SYNTHESIS__
    assert(t_numCols % t_memWidth == 0);
    assert(t_numRows % t_memWidth == 0);
#endif
    typedef typename WideTypeDefs<t_memWidth, T_elemType>::WideIFStreamType WideIFStreamType;
    typedef typename WideTypeDefs<t_memWidth, T_elemType>::WideIFType WideSampleType;

    static const unsigned int k_ColWords = t_numCols / t_memWidth;
    static const unsigned int k_RowWords = t_numRows / t_memWidth;

    T_elemType l_buf[t_numRows * k_ColWords][t_memWidth];

#pragma HLS RESOURCE variable = l_buf core = XPM_MEMORY uram
#pragma HLS ARRAY_PARTITION variable = l_buf complete dim = 2

    //#ifndef __SYNTHESIS__
    for (unsigned int b = 0; b < t_numBlocks; ++b) {
    //#pragma HLS DATAFLOW disable_start_propagation

    //#pragma HLS DATAFLOW
    //#endif
    transpMemBlocks_writePIPO:
        for (unsigned int i = 0; i < t_numRows; ++i) {
            for (unsigned int j = 0; j < k_ColWords; ++j) {
#pragma HLS PIPELINE II = 1 rewind
                WideSampleType l_val = p_inWideStream.read();
#pragma HLS ARRAY_PARTITION variable = l_val complete
                for (unsigned int k = 0; k < t_memWidth; ++k) {
                    l_buf[i * k_ColWords + j][k] = l_val[(t_memWidth - i + k) % t_memWidth];
                }
            }
        }
    transpMemBlocks_readPIPO:
        for (unsigned int i = 0; i < t_numCols; ++i) {
            for (unsigned int j = 0; j < k_RowWords; ++j) {
#pragma HLS PIPELINE II = 1 rewind
                WideSampleType l_val;
#pragma HLS ARRAY_PARTITION variable = l_val complete
                WideSampleType l_out;
#pragma HLS ARRAY_PARTITION variable = l_out complete
                for (unsigned int k = 0; k < t_memWidth; ++k) {
                    l_val[k] = l_buf[j * t_numCols + i / t_memWidth +
                                     ((k + t_memWidth - i % t_memWidth) % t_memWidth) * k_ColWords][k];
                }
                for (unsigned int k = 0; k < t_memWidth; ++k) {
                    l_out[k] = l_val[(k + i) % t_memWidth];
                }
                p_outWideStream.write(l_out);
            }
        }
        //#ifndef __SYNTHESIS__
    }
    //#endif
}

} // end namespace fft
} // end namespace dsp
} // end namespace xf

#endif //__HLS_SSR_FFT_MATRIX_COMMUTOR_H__
