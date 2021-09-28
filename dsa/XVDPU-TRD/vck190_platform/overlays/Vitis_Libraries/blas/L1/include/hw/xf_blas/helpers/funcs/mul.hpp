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
 * @file mul.hpp
 * @brief BLAS Level 1 sum template function implementation.
 *
 * This file is part of Vitis BLAS Library.
 */

#ifndef XF_BLAS_MUL_HPP
#define XF_BLAS_MUL_HPP

#ifndef __cplusplus
#error "BLAS Library only works with C++."
#endif

#include "hls_stream.h"

namespace xf {

namespace blas {

template <typename t_DataType,
          unsigned int t_ParEntries,
          typename t_IndexType = unsigned int,
          typename t_MulDataType = t_DataType>
void mul(unsigned int p_n,
         hls::stream<typename WideType<t_DataType, t_ParEntries>::t_TypeInt>& p_x,
         hls::stream<typename WideType<t_DataType, t_ParEntries>::t_TypeInt>& p_y,
         hls::stream<typename WideType<t_MulDataType, t_ParEntries>::t_TypeInt>& p_res,
         hls::stream<typename WideType<t_DataType, t_ParEntries>::t_TypeInt>& p_y_c,
         unsigned int p_mulIters = 1) {
#ifndef __SYNTHESIS__
    assert(p_n % t_ParEntries == 0);
#endif
    t_IndexType l_numParEntries = p_n / t_ParEntries;
    for (int r = 0; r < p_mulIters; r++)
        for (t_IndexType i = 0; i < l_numParEntries; ++i) {
#pragma HLS PIPELINE
            WideType<t_DataType, t_ParEntries> l_valX;
            WideType<t_DataType, t_ParEntries> l_valY;
            WideType<t_MulDataType, t_ParEntries> l_valRes;
            l_valX = p_x.read();
            l_valY = p_y.read();
            for (unsigned int j = 0; j < t_ParEntries; ++j) {
                l_valRes[j] = l_valX[j] * l_valY[j];
            }
            p_res.write(l_valRes);
            p_y_c.write(l_valY);
        }
}

template <typename t_DataType,
          unsigned int t_ParEntries,
          typename t_IndexType = unsigned int,
          typename t_MulDataType = t_DataType>
void mul(unsigned int p_n,
         hls::stream<typename WideType<t_DataType, t_ParEntries>::t_TypeInt>& p_x,
         hls::stream<typename WideType<t_DataType, t_ParEntries>::t_TypeInt>& p_y,
         hls::stream<typename WideType<t_MulDataType, t_ParEntries>::t_TypeInt>& p_res,
         unsigned int p_mulIters = 1) {
#ifndef __SYNTHESIS__
    assert(p_n % t_ParEntries == 0);
#endif
    t_IndexType l_numParEntries = p_n / t_ParEntries;
    for (int r = 0; r < p_mulIters; r++)
        for (t_IndexType i = 0; i < l_numParEntries; ++i) {
#pragma HLS PIPELINE
            WideType<t_DataType, t_ParEntries> l_valX;
            WideType<t_DataType, t_ParEntries> l_valY;
            WideType<t_MulDataType, t_ParEntries> l_valRes;
            l_valX = p_x.read();
            l_valY = p_y.read();
            for (unsigned int j = 0; j < t_ParEntries; ++j) {
                l_valRes[j] = l_valX[j] * l_valY[j];
            }
            p_res.write(l_valRes);
        }
}

} // end namespace blas

} // end namespace xf
#endif
