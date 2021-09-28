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
 * @file abs.hpp
 * @brief BLAS Level 1 abs template function implementation.
 *
 * This file is part of Vitis BLAS Library.
 */

#ifndef XF_BLAS_ABS_HPP
#define XF_BLAS_ABS_HPP

#ifndef __cplusplus
#error "BLAS Library only works with C++."
#endif

#include "hls_math.h"
#include "hls_stream.h"

namespace xf {

namespace blas {

/**
 * @brief abs function that returns the magnitude of each vector element.
 *
 * @tparam t_DataType the data type of the vector entries
 * @tparam t_ParEntries the number of parallelly processed entries in the input vector
 * @tparam t_IndexType the datatype of the index
 *
 * @param p_n the number of entries in the input vector p_x, p_n % l_ParEntries == 0
 * @param p_x the input stream of packed vector entries
 * @param p_abs the output stream of packed vector entries
 */

template <typename t_DataType,
          unsigned int t_ParEntries,
          typename t_IndexType = unsigned int,
          typename t_AbsDataType = t_DataType,
          unsigned int t_AbsDataWidth = sizeof(t_AbsDataType) << 3>
void abs(unsigned int p_n,
         hls::stream<typename WideType<t_DataType, t_ParEntries>::t_TypeInt>& p_x,
         hls::stream<typename WideType<t_AbsDataType, t_ParEntries, t_AbsDataWidth>::t_TypeInt>& p_abs) {
#ifndef __SYNTHESIS__
    assert(p_n % t_ParEntries == 0);
#endif
    unsigned int l_numElems = p_n / t_ParEntries;
    for (t_IndexType i = 0; i < l_numElems; i++) {
#pragma HLS PIPELINE
        WideType<t_DataType, t_ParEntries> l_x = p_x.read();
        WideType<t_AbsDataType, t_ParEntries> l_abs;
        for (t_IndexType j = 0; j < t_ParEntries; j++) {
#pragma HLS UNROLL
            l_abs[j] = hls::abs(l_x[j]);
        }
        p_abs.write(l_abs);
    }
}
} // namespace blas

} // namespace xf

#endif
