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
 * @file trmMatMoverB2.hpp
 * @brief data movers for triangular matrices and corresponding vectors.
 *
 * This file is part of Vitis BLAS Library.
 */

#ifndef XF_BLAS_TRMMATMOVERB2_HPP
#define XF_BLAS_TRMMATMOVERB2_HPP

#include "hls_stream.h"
#include "ap_int.h"
#include "ap_shift_reg.h"

namespace xf {

namespace blas {
/**
 * @brief trmUp2Stream function that read the super-triangular matrix from memory to stream
 *
 * @tparam t_DataType the data type of the matrix entries
 * @tparam t_ParEntries the number of parallelly processed entries in the matrix
 *
 * @param p_n number of rows/cols in a symmetric matrix
 * @param p_a memory location of a p_n x p_n symmetric matrix
 * @param p_out the streams of matrix entries
 */
template <typename t_DataType, unsigned int t_ParEntries>
void trmUp2Stream(unsigned int p_n,
                  t_DataType* p_a,
                  hls::stream<typename WideType<t_DataType, t_ParEntries>::t_TypeInt>& p_out) {
    unsigned int l_blocks = p_n / t_ParEntries;
    unsigned int l_blocksMinus1 = l_blocks - 1;
    unsigned int i = 0;
    unsigned int j = 0;
    while (i < p_n) {
#pragma HLS PIPELINE REWIND
        WideType<t_DataType, t_ParEntries> l_val;
#pragma HLS ARRAY_PARTITION variable = l_val complete
        for (unsigned int k = 0; k < t_ParEntries; ++k) {
            l_val[k] = p_a[(i * l_blocks + j) * t_ParEntries + k];
        }
        p_out.write(l_val);
        if (j == l_blocksMinus1) {
            i++;
            j = i / t_ParEntries;
        } else {
            j++;
        }
    }
}

/**
 * @brief trmLo2Stream function that read the sub-tridiagonal matrix with from memory to stream
 *
 * @tparam t_DataType the data type of the matrix entries
 * @tparam t_ParEntries the number of parallelly processed entries in the matrix
 *
 * @param p_n number of rows/cols in a symmetric matrix
 * @param p_a memory location of a p_n x p_n symmetric matrix
 * @param p_out the streams of matrix entries
 */
template <typename t_DataType, unsigned int t_ParEntries>
void trmLo2Stream(unsigned int p_n,
                  t_DataType* p_a,
                  hls::stream<typename WideType<t_DataType, t_ParEntries>::t_TypeInt>& p_out) {
    uint16_t l_blocks = p_n / t_ParEntries;
    unsigned int i = 0;
    unsigned int j = 0;
    while (i < p_n) {
#pragma HLS PIPELINE REWIND
        WideType<t_DataType, t_ParEntries> l_val;
#pragma HLS ARRAY_PARTITION variable = l_val complete
        for (unsigned int k = 0; k < t_ParEntries; ++k) {
            l_val[k] = p_a[(i * l_blocks + j) * t_ParEntries + k];
        }
        p_out.write(l_val);
        if (j == i / t_ParEntries) {
            j = 0;
            i++;
        } else {
            j++;
        }
    }
}

/**
 * @brief tpmUp2Stream function that read the packed super-triangular matrix from memory to stream
 *
 * @tparam t_DataType the data type of the matrix entries
 * @tparam t_ParEntries the number of parallelly processed entries in the matrix
 *
 * @param p_n number of rows/cols in a symmetric matrix
 * @param p_a memory location of a p_n x p_n symmetric matrix
 * @param p_out the streams of matrix entries
 */
template <typename t_DataType, unsigned int t_ParEntries>
void tpmUp2Stream(unsigned int p_n,
                  t_DataType* p_a,
                  hls::stream<typename WideType<t_DataType, t_ParEntries>::t_TypeInt>& p_out) {
    uint16_t l_blocks = p_n / t_ParEntries;
    t_DataType* l_rowAddr = p_a;
    for (unsigned int i = 0; i < p_n; ++i) {
        unsigned int l_rowBlockId = i / t_ParEntries;
        for (unsigned int j = l_rowBlockId; j < l_blocks; ++j) {
#pragma HLS PIPELINE
            t_DataType* l_blockAddr = l_rowAddr + (j - l_rowBlockId) * t_ParEntries;
            WideType<t_DataType, t_ParEntries> l_val;
#pragma HLS ARRAY_PARTITION variable = l_val complete
            for (unsigned int k = 0; k < t_ParEntries; ++k) {
                l_val[k] = l_blockAddr[k];
            }
            p_out.write(l_val);
        }
        l_rowAddr += p_n - (l_rowBlockId * t_ParEntries);
    }
}

/**
 * @brief tpmLo2Stream function that read the packed sub-symmetric matrix with from memory to stream
 *
 * @tparam t_DataType the data type of the matrix entries
 * @tparam t_ParEntries the number of parallelly processed entries in the matrix
 *
 * @param p_n number of rows/cols in a symmetric matrix
 * @param p_a memory location of a p_n x p_n symmetric matrix
 * @param p_out the streams of matrix entries
 */
template <typename t_DataType, unsigned int t_ParEntries>
void tpmLo2Stream(unsigned int p_n,
                  t_DataType* p_a,
                  hls::stream<typename WideType<t_DataType, t_ParEntries>::t_TypeInt>& p_out) {
    uint16_t l_blocks = p_n / t_ParEntries;
    t_DataType* l_rowAddr = p_a;
    for (unsigned int i = 0; i < p_n; ++i) {
        unsigned int l_rowBlockId = i / t_ParEntries + 1;
        for (unsigned int j = 0; j < l_rowBlockId; ++j) {
#pragma HLS PIPELINE
            t_DataType* l_blockAddr = l_rowAddr + j * t_ParEntries;
            WideType<t_DataType, t_ParEntries> l_val;
#pragma HLS ARRAY_PARTITION variable = l_val complete
            for (unsigned int k = 0; k < t_ParEntries; ++k) {
                l_val[k] = l_blockAddr[k];
            }
            p_out.write(l_val);
        }
        l_rowAddr += l_rowBlockId * t_ParEntries;
    }
}

/**
 * @brief vec2TrmUpStream function that moves vector from memory to stream that matches the trmUp2Stream/tpmUp2Stream
 * outputs
 *
 * @tparam t_DataType the data type of the matrix entries
 * @tparam t_ParEntries number of parallelly processed entries in the matrix
 *
 * @param p_n number of rows/cols in a square matrix
 * @param p_x vector input
 * @param p_out output stream
 */
template <typename t_DataType, unsigned int t_ParEntries>
void vec2TrmUpStream(unsigned int p_n,
                     t_DataType* p_x,
                     hls::stream<typename WideType<t_DataType, t_ParEntries>::t_TypeInt>& p_out) {
    unsigned int l_blocks = p_n / t_ParEntries;
    unsigned int l_blocksMinus1 = l_blocks - 1;
    unsigned int i = 0;
    unsigned int j = 0;
    while (i < p_n) {
#pragma HLS PIPELINE REWIND
        WideType<t_DataType, t_ParEntries> l_val;
#pragma HLS ARRAY_PARTITION variable = l_val complete
        for (unsigned int k = 0; k < t_ParEntries; ++k) {
            l_val[k] = p_x[j * t_ParEntries + k];
        }
        p_out.write(l_val);
        if (j == l_blocksMinus1) {
            i++;
            j = i / t_ParEntries;
        } else {
            j++;
        }
    }
}

/**
 * @brief vec2TrmLoStream function that moves vector from memory to stream that matches the trmLo2Stream/tpmLo2Stream
 * outputs
 *
 * @tparam t_DataType the data type of the matrix entries
 * @tparam t_ParEntries number of parallelly processed entries in the matrix
 *
 * @param p_n number of rows/cols in a square matrix
 * @param p_x vector input
 * @param p_out output stream
 */
template <typename t_DataType, unsigned int t_ParEntries>
void vec2TrmLoStream(unsigned int p_n,
                     t_DataType* p_x,
                     hls::stream<typename WideType<t_DataType, t_ParEntries>::t_TypeInt>& p_out) {
    unsigned int i = 0;
    unsigned int j = 0;
    while (i < p_n) {
#pragma HLS PIPELINE REWIND
        WideType<t_DataType, t_ParEntries> l_val;
#pragma HLS ARRAY_PARTITION variable = l_val complete
        for (unsigned int k = 0; k < t_ParEntries; ++k) {
            l_val[k] = p_x[j * t_ParEntries + k];
        }
        p_out.write(l_val);
        if (j == i / t_ParEntries) {
            j = 0;
            i++;
        } else {
            j++;
        }
    }
}

} // namespace blas

} // namespace xf
#endif
