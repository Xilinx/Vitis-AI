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
 * @file nrm2.hpp
 * @brief BLAS Level 1 asum template function implementation.
 *
 * This file is part of Vitis BLAS Library.
 */

#ifndef XF_BLAS_NRM2_HPP
#define XF_BLAS_NRM2_HPP

#ifndef __cplusplus
#error "BLAS Library only works with C++."
#endif

#include "ap_int.h"
#include "hls_math.h"
#include "hls_stream.h"
#include "xf_blas/helpers.hpp"

namespace xf {

namespace blas {

namespace {
template <typename t_DataType, unsigned int t_ParEntries, typename t_IndexType = unsigned int>
void square(unsigned int p_n,
            hls::stream<typename WideType<t_DataType, t_ParEntries>::t_TypeInt>& p_x,
            hls::stream<typename WideType<t_DataType, t_ParEntries>::t_TypeInt>& p_res) {
#ifndef __SYNTHESIS__
    assert(p_n % t_ParEntries == 0);
#endif
    t_IndexType l_numParEntries = p_n / t_ParEntries;
    for (t_IndexType i = 0; i < l_numParEntries; ++i) {
#pragma HLS PIPELINE
        WideType<t_DataType, t_ParEntries> l_valX;
        WideType<t_DataType, t_ParEntries> l_valRes;
        l_valX = p_x.read();
        for (unsigned int j = 0; j < t_ParEntries; ++j) {
            l_valRes[j] = l_valX[j] * l_valX[j];
        }
        p_res.write(l_valRes);
    }
}
template <typename t_DataType, unsigned int t_LogParEntries, typename t_IndexType = unsigned int>
void nrm2Square(unsigned int p_n,
                hls::stream<typename WideType<t_DataType, 1 << t_LogParEntries>::t_TypeInt>& p_x,
                t_DataType& p_res) {
#ifndef __SYNTHESIS__
    assert(p_n % (1 << t_LogParEntries) == 0);
#endif
#pragma HLS DATAFLOW
    hls::stream<typename WideType<t_DataType, 1 << t_LogParEntries>::t_TypeInt> l_mulStr;
    square<t_DataType, 1 << t_LogParEntries, t_IndexType>(p_n, p_x, l_mulStr);
    sum<t_DataType, t_LogParEntries, t_IndexType>(p_n, l_mulStr, p_res);
}
} // namespace

/**
 * @brief nrm2 function that returns the Euclidean norm of the vector x.
 *
 * @tparam t_DataType the data type of the vector entries
 * @tparam t_LogParEntries log2 of the number of parallelly processed entries in the input vector
 * @tparam t_IndexType the datatype of the index
 *
 * @param p_n the number of entries in the input vector p_x, p_n % (1<<l_LogParEntries) == 0
 * @param p_x the input stream of packed vector entries
 * @param p_res the nrm2  of x
 */

template <typename t_DataType, unsigned int t_LogParEntries, typename t_IndexType = unsigned int>
void nrm2(unsigned int p_n,
          hls::stream<typename WideType<t_DataType, (1 << t_LogParEntries)>::t_TypeInt>& p_x,
          t_DataType& p_res) {
#ifndef __SYNTHESIS__
    assert(p_n % (1 << t_LogParEntries) == 0);
#endif
    t_DataType l_resSquare;
    nrm2Square<t_DataType, t_LogParEntries, t_IndexType>(p_n, p_x, l_resSquare);
    p_res = hls::sqrt(l_resSquare);
}

} // end namespace blas

} // end namespace xf

#endif
