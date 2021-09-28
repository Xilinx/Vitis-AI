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
 * @file axpy.hpp
 * @brief BLAS Level 1 axpy template function implementation.
 *
 * This file is part of Vitis BLAS Library.
 */

#ifndef XF_BLAS_AXPY_HPP
#define XF_BLAS_AXPY_HPP

#ifndef __cplusplus
#error "BLAS Library only works with C++."
#endif

#include "ap_int.h"
#include "hls_math.h"
#include "hls_stream.h"
#include "xf_blas/helpers.hpp"

namespace xf {

namespace blas {

/**
 * @brief axpy function that compute Y = alpha*X + Y.
 *
 * @tparam t_DataType the data type of the vector entries
 * @tparam t_LogParEntries log2 of the number of parallelly processed entries in the input vector
 * @tparam t_IndexType the datatype of the index
 *
 * @param p_n the number of entries in the input vector p_x, p_n % t_ParEntries == 0
 * @param p_x the input stream of packed entries of vector X
 * @param p_y the input stream of packed entries of vector Y
 * @param p_r the output stream of packed entries of result vector Y
 */
template <typename t_DataType, unsigned int t_ParEntries, typename t_IndexType = unsigned int>
void axpy(unsigned int p_n,
          const t_DataType p_alpha,
          hls::stream<typename WideType<t_DataType, t_ParEntries>::t_TypeInt>& p_x,
          hls::stream<typename WideType<t_DataType, t_ParEntries>::t_TypeInt>& p_y,
          hls::stream<typename WideType<t_DataType, t_ParEntries>::t_TypeInt>& p_r) {
#ifndef __SYNTHESIS__
    assert(p_n % t_ParEntries == 0);
#endif
    const unsigned int l_numElem = p_n / t_ParEntries;
    for (t_IndexType i = 0; i < l_numElem; i++) {
#pragma HLS PIPELINE
        WideType<t_DataType, t_ParEntries> l_x = p_x.read();
        WideType<t_DataType, t_ParEntries> l_y = p_y.read();
        WideType<t_DataType, t_ParEntries> l_r;
        for (t_IndexType j = 0; j < t_ParEntries; j++) {
#pragma HLS UNROLL
            t_DataType l_realX = l_x[j];
            t_DataType l_realY = l_y[j];
            t_DataType l_result = p_alpha * l_realX + l_realY;
            l_r[j] = l_result;
        }
        p_r.write(l_r);
    }
}

} // end namespace blas

} // end namespace xf
#endif
