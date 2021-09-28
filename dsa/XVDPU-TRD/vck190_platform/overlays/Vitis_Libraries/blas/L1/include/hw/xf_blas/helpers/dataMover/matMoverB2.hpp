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
 * @file matMoverB2.hpp
 * @brief common datamovers for matrices and vectors used in BLAS L2 routines.
 *
 * This file is part of Vitis BLAS Library.
 */

#ifndef XF_BLAS_MATMOVERB2_HPP
#define XF_BLAS_MATMOVERB2_HPP

#include "hls_stream.h"
#include "ap_int.h"
#include "ap_shift_reg.h"

namespace xf {

namespace blas {

/**
 * @brief gem2Stream function that moves row-major matrix from memory to stream
 *
 * @tparam t_DataType the data type of the matrix entries
 * @tparam t_ParEntries number of parallelly processed entries in the matrix
 *
 * @param p_m number of rows in a matrix
 * @param p_n number of cols in a matrix
 * @param p_in a p_m x p_n matrix with on-chip row-major storage
 * @param p_out output stream
 */
template <typename t_DataType, unsigned int t_ParEntries>
void gem2Stream(unsigned int p_m,
                unsigned int p_n,
                t_DataType* p_in,
                hls::stream<typename WideType<t_DataType, t_ParEntries>::t_TypeInt>& p_out) {
#ifndef __SYNTHESIS__
    assert((p_n % t_ParEntries) == 0);
#endif
    unsigned int l_parBlocks = p_m * p_n / t_ParEntries;
    for (unsigned int i = 0; i < l_parBlocks; ++i) {
#pragma HLS PIPELINE
        BitConv<t_DataType> l_bitConv;
        WideType<t_DataType, t_ParEntries> l_val;
        for (unsigned int j = 0; j < t_ParEntries; ++j) {
            l_val[j] = p_in[i * t_ParEntries + j];
        }
        p_out.write(l_val);
    }
} // end gem2Stream

/**
 * @brief vec2GemStream function that moves vector from memory to stream that matches the gem2Stream outputs
 *
 * @tparam t_DataType the data type of the matrix entries
 * @tparam t_ParEntries number of parallelly processed entries in the matrix
 *
 * @param p_m number of rows in a matrix
 * @param p_n number of cols in a matrix
 * @param p_in vector input
 * @param p_out output stream
 */
template <typename t_DataType, unsigned int t_ParEntries>
void vec2GemStream(unsigned int p_m,
                   unsigned int p_n,
                   t_DataType* p_in,
                   hls::stream<typename WideType<t_DataType, t_ParEntries>::t_TypeInt>& p_out) {
#ifndef __SYNTHESIS__
    assert((p_n % t_ParEntries) == 0);
#endif
    unsigned int l_parBlocks = p_n / t_ParEntries;
    for (unsigned int l = 0; l < p_m; ++l) {
        for (unsigned int i = 0; i < l_parBlocks; ++i) {
#pragma HLS PIPELINE
            BitConv<t_DataType> l_bitConv;
            WideType<t_DataType, t_ParEntries> l_val;
            for (unsigned int j = 0; j < t_ParEntries; ++j) {
                l_val[j] = p_in[i * t_ParEntries + j];
            }
            p_out.write(l_val);
        }
    }
} // end vec2GemStream
} // namespace blas

} // namespace xf
#endif
