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

#ifndef XF_BLAS_SUM_HPP
#define XF_BLAS_SUM_HPP

#ifndef __cplusplus
#error "BLAS Library only works with C++."
#endif

#include "hls_math.h"
#include "hls_stream.h"

namespace xf {

namespace blas {

namespace {
template <typename t_DataType,
          unsigned int t_LogParEntries,
          typename t_IndexType = unsigned int,
          typename t_SumDataType = t_DataType>
void preProcess(unsigned int p_numElems,
                hls::stream<typename WideType<t_DataType, 1 << t_LogParEntries>::t_TypeInt>& p_x,
                hls::stream<t_SumDataType>& p_data,
                unsigned int p_mulIters = 1

                ) {
    const unsigned int l_ParEntries = 1 << t_LogParEntries;
    for (unsigned int r = 0; r < p_mulIters; r++)
        for (t_IndexType i = 0; i < p_numElems; i++) {
#pragma HLS PIPELINE
            WideType<t_DataType, 1 << t_LogParEntries> l_x = p_x.read();
            t_SumDataType l_sum;
            l_sum = BinarySum<t_DataType, l_ParEntries, t_SumDataType>::sum(l_x.getValAddr());
            p_data.write(l_sum);
        }
}

template <typename t_DataType, unsigned int t_LogDelays, typename t_IndexType = unsigned int>
void postProcess(unsigned int p_numElems,
                 hls::stream<t_DataType>& p_pad,
                 hls::stream<typename WideType<t_DataType, 1>::t_TypeInt>& p_sum,
                 unsigned int p_mulIters = 1

                 ) {
    const unsigned int l_Delays = 1 << t_LogDelays;
    const unsigned int l_numIter = (p_numElems + l_Delays - 1) >> t_LogDelays;
    for (unsigned int r = 0; r < p_mulIters; r++) {
        WideType<t_DataType, 1> l_finalSum = 0;
        for (t_IndexType i = 0; i < l_numIter; i++) {
#pragma HLS PIPELINE II = l_Delays
            WideType<t_DataType, l_Delays> l_input;
            for (t_IndexType j = 0; j < l_Delays; j++) {
#pragma HLS UNROLL
                l_input.shift(p_pad.read());
            }
            l_finalSum[0] += BinarySum<t_DataType, l_Delays>::sum(l_input.getValAddr());
            if (i == l_numIter - 1) p_sum.write(l_finalSum);
        }
    }
}
} // namespace

/**
 * @brief sum function that returns the sum of all the vector elements.
 *
 * @tparam t_DataType the data type of the vector entries
 * @tparam t_LogParEntries log2 of the number of parallelly processed entries in the input vector
 * @tparam t_IndexType the datatype of the index
 *
 * @param p_n the number of entries in the input vector p_x, p_n % l_ParEntries == 0
 * @param p_x the input stream of packed vector entries
 * @param p_mulIters number of repeat
 * @param p_sum the sum, which is 0 if p_n <= 0
 */

template <typename t_DataType,
          unsigned int t_LogParEntries,
          typename t_IndexType = unsigned int,
          typename t_SumDataType = t_DataType>
void sum(unsigned int p_n,
         hls::stream<typename WideType<t_DataType, 1 << t_LogParEntries>::t_TypeInt>& p_x,
         hls::stream<typename WideType<t_SumDataType, 1>::t_TypeInt>& p_sum,
         unsigned int p_mulIters) {
#ifndef __SYNTHESIS__
    assert(p_n % (1 << t_LogParEntries) == 0);
#endif
    const unsigned int l_LogDelays = AdderDelay<t_DataType>::m_logDelays;
#pragma HLS DATAFLOW
    hls::stream<t_SumDataType> l_data, l_pad;
#pragma HLS stream variable = l_data depth = 2
#pragma HLS stream variable = l_pad depth = 2
    unsigned int l_numElem = p_n >> t_LogParEntries;
    preProcess<t_DataType, t_LogParEntries, t_IndexType, t_SumDataType>(l_numElem, p_x, l_data, p_mulIters);
    padding<t_SumDataType, 1 << l_LogDelays, t_IndexType>(l_numElem, l_data, l_pad, p_mulIters);
    postProcess<t_SumDataType, l_LogDelays, t_IndexType>(l_numElem, l_pad, p_sum, p_mulIters);
}

/**
 * @brief sum function that returns the sum of all the vector elements.
 *
 * @tparam t_DataType the data type of the vector entries
 * @tparam t_LogParEntries log2 of the number of parallelly processed entries in the input vector
 * @tparam t_IndexType the datatype of the index
 *
 * @param p_n the number of entries in the input vector p_x, p_n % l_ParEntries == 0
 * @param p_x the input stream of packed vector entries
 * @param p_sum the sum, which is 0 if p_n <= 0
 */

template <typename t_DataType,
          unsigned int t_LogParEntries,
          typename t_IndexType = unsigned int,
          typename t_SumDataType = t_DataType>
void sum(unsigned int p_n,
         hls::stream<typename WideType<t_DataType, 1 << t_LogParEntries>::t_TypeInt>& p_x,
         t_DataType& p_sum) {
#pragma HLS DATAFLOW
    hls::stream<typename WideType<t_DataType, 1>::t_TypeInt> l_s;
    sum<t_DataType, t_LogParEntries, t_IndexType>(p_n, p_x, l_s, 1);
    WideType<t_DataType, 1> l_v = l_s.read();
    p_sum = l_v[0];
}

} // end namespace blas

} // end namespace xf

#endif
