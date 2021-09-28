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
 * @file amin.hpp
 * @brief BLAS Level 1 amin template function implementation.
 *
 * This file is part of Vitis BLAS Library.
 */

#ifndef XF_BLAS_AMIN_HPP
#define XF_BLAS_AMIN_HPP

#ifndef __cplusplus
#error "BLAS Library only works with C++."
#endif

#include "ap_int.h"
#include "hls_stream.h"
#include "xf_blas/helpers.hpp"

namespace xf {

namespace blas {

/**
 * @brief amin function that returns the position of the vector element that has the minimum magnitude.
 *
 * @tparam t_DataType the data type of the vector entries
 * @tparam t_LogParEntries log2 of the number of parallelly processed entries in the input vector
 * @tparam t_IndexType the datatype of the index
 *
 * @param p_n the number of entries in the input vector p_x, p_n % l_ParEntries == 0
 * @param p_x the input stream of packed vector entries
 * @param p_result the resulting index, which is 0 if p_n <= 0
 */

template <typename t_DataType, unsigned int t_LogParEntries, typename t_IndexType>
void amin(unsigned int p_n,
          hls::stream<typename WideType<t_DataType, (1 << t_LogParEntries)>::t_TypeInt>& p_x,
          t_IndexType& p_result) {
#ifndef __SYNTHESIS__
    assert(p_n % (1 << t_LogParEntries) == 0);
#endif
    hls::stream<typename WideType<t_DataType, 1 << t_LogParEntries>::t_TypeInt> l_abs;
#pragma HLS stream variable = l_abs depth = 2
#pragma HLS DATAFLOW

    abs<t_DataType, 1 << t_LogParEntries, t_IndexType>(p_n, p_x, l_abs);
    min<t_DataType, t_LogParEntries, t_IndexType>(p_n, l_abs, p_result);
}

} // end namespace blas

} // end namespace xf

#endif
