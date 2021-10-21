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
 * @file maxmin.hpp
 * @brief BLAS Level 1 max and min template function implementation.
 *
 * This file is part of Vitis BLAS Library.
 */

#ifndef XF_BLAS_MAXMIN_HPP
#define XF_BLAS_MAXMIN_HPP

#ifndef __cplusplus
#error "BLAS Library only works with C++."
#endif

#include "hls_math.h"
#include "hls_stream.h"

namespace xf {

namespace blas {

namespace {
template <bool t_Max, typename t_DataType, typename t_IndexType>
void Compare(t_DataType p_v0, t_IndexType p_i0, t_DataType p_v1, t_IndexType p_i1, t_DataType& p_v, t_IndexType& p_i) {
    if (t_Max) {
        if (p_v1 > p_v0) {
            p_v = p_v1;
            p_i = p_i1;
        } else {
            p_v = p_v0;
            p_i = p_i0;
        }
    } else {
        if (p_v1 < p_v0) {
            p_v = p_v1;
            p_i = p_i1;
        } else {
            p_v = p_v0;
            p_i = p_i0;
        }
    }
}
template <typename t_DataType, unsigned int t_Entries, typename t_IndexType, bool t_Max>
class BinaryCmp {
   public:
    static const void cmp(t_DataType p_x[t_Entries], t_DataType& p_value, t_IndexType& p_index) {
        const unsigned int l_halfEntries = t_Entries >> 1;
        t_DataType l_msbValue, l_lsbValue;
        t_IndexType l_msbIndex, l_lsbIndex;
        BinaryCmp<t_DataType, l_halfEntries, t_IndexType, t_Max>::cmp(p_x, l_lsbValue, l_lsbIndex);
        BinaryCmp<t_DataType, l_halfEntries, t_IndexType, t_Max>::cmp(p_x + l_halfEntries, l_msbValue, l_msbIndex);
        Compare<t_Max, t_DataType, t_IndexType>(l_lsbValue, l_lsbIndex, l_msbValue, l_msbIndex + l_halfEntries, p_value,
                                                p_index);
    }
};
template <typename t_DataType, typename t_IndexType, bool t_Max>
class BinaryCmp<t_DataType, 1, t_IndexType, t_Max> {
   public:
    static const void cmp(t_DataType p_x[1], t_DataType& p_value, t_IndexType& p_index) {
        p_index = 0;
        p_value = p_x[p_index];
    }
};

template <typename t_DataType, unsigned int t_LogParEntries, typename t_IndexType, bool t_Max>
void preProcess(unsigned int p_numElement,
                hls::stream<t_DataType>& p_valueStream,
                hls::stream<t_IndexType>& p_indexStream,
                hls::stream<typename WideType<t_DataType, 1 << t_LogParEntries>::t_TypeInt>& p_x) {
    const unsigned int l_ParEntries = 1 << t_LogParEntries;
    for (t_IndexType i = 0; i < p_numElement; i++) {
#pragma HLS PIPELINE
        WideType<t_DataType, 1 << t_LogParEntries> l_elem = p_x.read();
#pragma HLS ARRAY_PARTITION variable = l_elem complete dim = 1
        t_IndexType l_pos;
        t_DataType l_value;
        BinaryCmp<t_DataType, l_ParEntries, t_IndexType, t_Max>::cmp(l_elem.getValAddr(), l_value, l_pos);
        p_valueStream.write(l_value);
        p_indexStream.write((i << t_LogParEntries) + l_pos);
    }
}

template <typename t_DataType, unsigned int t_LogNumEntries, typename t_IndexType, bool t_Max>
void postProcess(unsigned int p_numElement,
                 hls::stream<t_DataType>& p_valueStream,
                 hls::stream<t_IndexType>& p_indexStream,
                 t_IndexType& p_result) {
    const unsigned int l_numEntries = 1 << t_LogNumEntries;
    t_DataType l_maxmin;
    t_IndexType l_index = 0;
    const unsigned int l_numIter = p_numElement >> t_LogNumEntries;
    for (t_IndexType i = 0; i < l_numIter; i++) {
#pragma HLS PIPELINE II = l_numEntries
        t_DataType l_v[l_numEntries];
#pragma HLS ARRAY_PARTITION variable = l_v complete dim = 1
        t_IndexType l_i[l_numEntries];
        for (t_IndexType j = 0; j < l_numEntries; j++) {
            l_v[j] = p_valueStream.read();
            l_i[j] = p_indexStream.read();
        }
        if (i == 0) {
            l_maxmin = l_v[0];
            l_index = l_i[0];
        }
        t_IndexType l_pos;
        t_DataType l_value;
        BinaryCmp<t_DataType, l_numEntries, t_IndexType, t_Max>::cmp(l_v, l_value, l_pos);
        Compare<t_Max>(l_maxmin, l_index, l_value, l_i[l_pos], l_maxmin, l_index);
    }
    const unsigned int l_numRem = p_numElement - (l_numIter << t_LogNumEntries);
    for (t_IndexType i = 0; i < l_numRem; i++) {
//          #pragma HLS PIPELINE
#pragma HLS loop_tripcount max = l_numEntries
        t_DataType l_v;
        t_IndexType l_i;
        l_v = p_valueStream.read();
        l_i = p_indexStream.read();
        Compare<t_Max>(l_maxmin, l_index, l_v, l_i, l_maxmin, l_index);
    }
    p_result = l_index;
}

template <typename t_DataType,
          unsigned int t_LogParEntries,
          typename t_IndexType,
          bool t_Max>
void MaxMinHelper(unsigned int p_n, // number of element in the stream
                  hls::stream<typename WideType<t_DataType, 1 << t_LogParEntries>::t_TypeInt>& p_x,
                  t_IndexType& p_result) {
#pragma HLS DATAFLOW
    hls::stream<t_DataType> l_valueStream;
#pragma HLS stream variable = l_valueStream depth = 2
    hls::stream<t_IndexType> l_indexStream;
#pragma HLS stream variable = l_indexStream depth = 2

    preProcess<t_DataType, t_LogParEntries, t_IndexType, t_Max>(p_n, l_valueStream, l_indexStream, p_x);
    postProcess<t_DataType, 1, t_IndexType, t_Max>(p_n, l_valueStream, l_indexStream, p_result);
}

} // namespace

/**
 * @brief max function that returns the position of the vector element that has the maximum magnitude.
 *
 * @tparam t_DataType the data type of the vector entries
 * @tparam t_LogParEntries log2 of the number of parallelly processed entries in the input vector
 * @tparam t_IndexType the datatype of the index
 *
 * @param p_n the number of stided entries entries in the input vector p_x, p_n % l_ParEntries == 0
 * @param p_x the input stream of packed vector entries
 * @param p_result the resulting index, which is 0 if p_n <= 0
 */

template <typename t_DataType, unsigned int t_LogParEntries, typename t_IndexType>
void max(unsigned int p_n,
         hls::stream<typename WideType<t_DataType, 1 << t_LogParEntries>::t_TypeInt>& p_x,
         t_IndexType& p_result) {
#ifndef __SYNTHESIS__
    assert(p_n % (1 << t_LogParEntries) == 0);
#endif
    unsigned int l_numElem = p_n >> t_LogParEntries;
    MaxMinHelper<t_DataType, t_LogParEntries, t_IndexType, true>(l_numElem, p_x, p_result);
}

/**
 * @brief min function that returns the position of the vector element that has the minimum magnitude.
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
void min(unsigned int p_n,
         hls::stream<typename WideType<t_DataType, 1 << t_LogParEntries>::t_TypeInt>& p_x,
         t_IndexType& p_result) {
#ifndef __SYNTHESIS__
    assert(p_n % (1 << t_LogParEntries) == 0);
#endif
    unsigned int l_numElem = p_n >> t_LogParEntries;
    MaxMinHelper<t_DataType, t_LogParEntries, t_IndexType, false>(l_numElem, p_x, p_result);
}

} // end namespace blas

} // end namespace xf

#endif
