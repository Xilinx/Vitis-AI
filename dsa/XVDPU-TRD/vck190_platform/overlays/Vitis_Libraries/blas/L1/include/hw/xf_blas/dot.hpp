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
 * @file dot.hpp
 * @brief BLAS Level 1 dot template function implementation.
 *
 * This file is part of Vitis BLAS Library.
 */

#ifndef XF_BLAS_DOT_HPP
#define XF_BLAS_DOT_HPP

#ifndef __cplusplus
#error "BLAS Library only works with C++."
#endif

#include "ap_int.h"
#include "hls_stream.h"
#include "xf_blas/helpers.hpp"

namespace xf {

namespace blas {

/**
 * @brief dot function that returns the dot product of vector x and y.
 *
 * @tparam t_DataType the data type of the vector entries
 * @tparam t_LogParEntries log2 of the number of parallelly processed entries in the input vector
 * @tparam t_IndexType the datatype of the index
 *
 * @param p_n the number of entries in the input vector p_x, p_n % l_ParEntries == 0
 * @param p_x the input stream of packed vector entries
 * @param p_res the dot product of x and y
 */

template <typename t_DataType, unsigned int t_LogParEntries, typename t_IndexType = unsigned int>
void dot(unsigned int p_n,
         hls::stream<typename WideType<t_DataType, (1 << t_LogParEntries)>::t_TypeInt>& p_x,
         hls::stream<typename WideType<t_DataType, (1 << t_LogParEntries)>::t_TypeInt>& p_y,
         t_DataType& p_res) {
#ifndef __SYNTHESIS__
    assert(p_n % (1 << t_LogParEntries) == 0);
#endif

    hls::stream<typename WideType<t_DataType, 1>::t_TypeInt> l_dot;
#pragma HLS DATAFLOW
    DotHelper<t_DataType, t_LogParEntries, t_IndexType>::dot(p_n, 1, p_x, p_y, l_dot);
    WideType<t_DataType, 1> l_v = l_dot.read();
    p_res = l_v[0];
}

} // end namespace blas

} // end namespace xf

#endif
