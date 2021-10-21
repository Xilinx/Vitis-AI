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
 * @file sum.hpp
 * @brief BLAS Level 1 sum template function implementation.
 *
 * This file is part of Vitis BLAS Library.
 */

#ifndef XF_BLAS_PADDING_HPP
#define XF_BLAS_PADDING_HPP

#ifndef __cplusplus
#error "BLAS Library only works with C++."
#endif

#include "hls_math.h"
#include "hls_stream.h"

namespace xf {

namespace blas {

/**
 * @brief padding function that pads the input vector to a size multiple of a given value
 *
 * @tparam t_DataType the data type of the vector entries
 * @tparam t_Pads, pads the vector size to be multiple of t_Pads
 * @tparam t_IndexType the datatype of the index
 *
 * @param p_n the number of entries in the input vector p_x, p_n % l_ParEntries == 0
 * @param p_data the input stream of vector entries
 * @param p_mulIters number of repeat
 * @param p_pad the output stream
 */

template <typename t_DataType, unsigned int t_Pads, typename t_IndexType = unsigned int>
void padding(unsigned int p_n,
             hls::stream<t_DataType>& p_data,
             hls::stream<t_DataType>& p_pad,
             unsigned int p_mulIters = 1) {
    const unsigned int l_numIter = (p_n + t_Pads - 1) / t_Pads;
    const unsigned int l_totalNum = l_numIter * t_Pads;
    for (unsigned int r = 0; r < p_mulIters; r++) {
        for (t_IndexType i = 0; i < l_totalNum; i++) {
#pragma HLS PIPELINE
            t_DataType l_v = i < p_n ? p_data.read() : 0;
            p_pad.write(l_v);
        }
    }
}

} // end namespace blas

} // end namespace xf

#endif
