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

#ifndef XF_BLAS_SYMV_HPP
#define XF_BLAS_SYMV_HPP

#ifndef __cplusplus
#error "BLAS Library only works with C++."
#endif

#include "ap_int.h"
#include "hls_stream.h"
#include "xf_blas/helpers.hpp"
#include "scal.hpp"
#include "axpy.hpp"

namespace xf {

namespace blas {

template <typename t_DataType, unsigned int t_LogParEntries, typename t_IndexType = unsigned int>
void symv(const unsigned int p_n,
          hls::stream<typename WideType<t_DataType, 1 << t_LogParEntries>::t_TypeInt>& p_M,
          hls::stream<typename WideType<t_DataType, 1 << t_LogParEntries>::t_TypeInt>& p_x,
          hls::stream<typename WideType<t_DataType, 1 << t_LogParEntries>::t_TypeInt>& p_y) {
#ifndef __SYNTHESIS__
    assert(p_n % (1 << t_LogParEntries) == 0);
#endif
    const unsigned int l_parEntries = 1 << t_LogParEntries;
    const unsigned int l_numBlocks = p_n >> t_LogParEntries;
    for (t_IndexType i = 0; i < l_numBlocks; i++) {
        WideType<t_DataType, 1 << t_LogParEntries> l_y(0);
        for (t_IndexType j = 0; j < l_numBlocks; j++) {
            for (t_IndexType k = 0; k < l_parEntries; k++) {
#pragma HLS PIPELINE
                t_DataType l_dot[1 << t_LogParEntries];
#pragma HLS ARRAY_PARTITION variable = l_dot complete dim = 1
                WideType<t_DataType, 1 << t_LogParEntries> l_M = p_M.read();
                WideType<t_DataType, 1 << t_LogParEntries> l_x = p_x.read();
                for (t_IndexType l = 0; l < l_parEntries; l++) {
                    l_dot[l] = l_M[l] * l_x[l];
                }
                l_y[k] += BinarySum<t_DataType, 1 << t_LogParEntries>::sum(l_dot);
            }
        }
        p_y.write(l_y);
    }
}

/**
 * @brief symv function that returns the result vector of the multiplication of a
 * symmetric matrix and a vector y = alpha * M * x + beta * y
 *
 * @tparam t_DataType the data type of the vector entries
 * @tparam t_LogParEntries log2 of the number of parallelly processed entries in the input vector
 * @tparam t_IndexType the datatype of the index
 *
 * @param p_n the dimention of input matrix p_M, as well as the number of entries in the input vector p_x, p_n %
 * l_ParEntries == 0
 * @param p_alpha, scalar alpha
 * @param p_M the input stream of packed Matrix entries
 * @param p_x the input stream of packed vector entries
 * @param p_beta, scalar beta
 * @param p_y the output vector
 */
template <typename t_DataType, unsigned int t_LogParEntries, typename t_IndexType = unsigned int>
void symv(const unsigned int p_n,
          const t_DataType p_alpha,
          hls::stream<typename WideType<t_DataType, (1 << t_LogParEntries)>::t_TypeInt>& p_M,
          hls::stream<typename WideType<t_DataType, (1 << t_LogParEntries)>::t_TypeInt>& p_x,
          const t_DataType p_beta,
          hls::stream<typename WideType<t_DataType, (1 << t_LogParEntries)>::t_TypeInt>& p_y,
          hls::stream<typename WideType<t_DataType, (1 << t_LogParEntries)>::t_TypeInt>& p_yr) {
#ifndef __SYNTHESIS__
    assert(p_n % (1 << t_LogParEntries) == 0);
#endif
    const unsigned int l_numIter = p_n >> t_LogParEntries;
    hls::stream<typename WideType<t_DataType, 1 << t_LogParEntries>::t_TypeInt> l_x, l_y;
#pragma HLS DATAFLOW
    symv<t_DataType, t_LogParEntries, t_IndexType>(p_n, p_M, p_x, l_x);
    scal<t_DataType, 1 << t_LogParEntries, t_IndexType>(p_n, p_beta, p_y, l_y);
    axpy<t_DataType, 1 << t_LogParEntries, t_IndexType>(p_n, p_alpha, l_x, l_y, p_yr);
}

} // end namespace blas

} // end namespace xf

#endif
