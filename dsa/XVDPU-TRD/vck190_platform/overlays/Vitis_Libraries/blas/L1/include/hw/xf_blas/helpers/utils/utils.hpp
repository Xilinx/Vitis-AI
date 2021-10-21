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
 * @file utils.hpp
 * @brief common datatypes for L1 modules.
 *
 * This file is part of Vitis BLAS Library.
 */

#ifndef XF_BLAS_UTILS_HPP
#define XF_BLAS_UTILS_HPP

namespace xf {

namespace blas {
// Helper macros for renaming kernel
#define PASTER(x, y) x##y
#define EVALUATOR(x, y) PASTER(x, y)

constexpr size_t mylog2(size_t n) {
    return ((n < 2) ? 0 : 1 + mylog2(n / 2));
}

template <typename t_DataType, unsigned int t_Entries, typename t_SumType = t_DataType>
class BinarySum {
   public:
    static const t_SumType sum(t_DataType p_x[t_Entries]) {
        const unsigned int l_halfEntries = t_Entries >> 1;
        return BinarySum<t_DataType, l_halfEntries, t_SumType>::sum(p_x) +
               BinarySum<t_DataType, l_halfEntries, t_SumType>::sum(p_x + l_halfEntries);
    }
};
template <typename t_DataType, typename t_SumType>
class BinarySum<t_DataType, 1, t_SumType> {
   public:
    static const t_SumType sum(t_DataType p_x[1]) {
#pragma HLS INLINE
        return p_x[0];
    }
};

template <typename t_StreamType, unsigned int t_Entries>
class Merge {
   public:
    static void merge(t_StreamType& p_left, t_StreamType& p_right, t_StreamType& p_str, int p_n) {
        const unsigned int l_halfEntries = t_Entries >> 1;
        for (int i = 0; i < p_n; i++) {
            for (int j = 0; j < l_halfEntries; j++) p_str.write(p_left.read());
            for (int j = 0; j < l_halfEntries; j++) p_str.write(p_right.read());
        }
    }
    static void merge(t_StreamType p_strs[t_Entries], t_StreamType& p_str, int p_n) {
        const unsigned int l_halfEntries = t_Entries >> 1;
        t_StreamType l_left, l_right;
#pragma HLS stream variable = l_left depth = t_Entries / 2
#pragma HLS stream variable = l_right depth = t_Entries / 2

#pragma HLS DATAFLOW
        Merge<t_StreamType, l_halfEntries>::merge(p_strs, l_left, p_n);
        Merge<t_StreamType, l_halfEntries>::merge(p_strs + l_halfEntries, l_right, p_n);
        merge(l_left, l_right, p_str, p_n);
    }
};

template <typename t_StreamType>
class Merge<t_StreamType, 2> {
   public:
    static void merge(t_StreamType p_strs[2], t_StreamType& p_str, int p_n) {
        for (int i = 0; i < p_n; i++) {
            p_str.write(p_strs[0].read());
            p_str.write(p_strs[1].read());
        }
    }
};

template <typename t_DataType>
class AdderDelay {
   public:
    static const unsigned int m_logDelays = 0;
    static const unsigned int m_Delays = 1 << m_logDelays;
};

template <>
class AdderDelay<double> {
   public:
    static const unsigned int m_logDelays = 3;
    static const unsigned int m_Delays = 1 << m_logDelays;
};

template <>
class AdderDelay<float> {
   public:
    static const unsigned int m_logDelays = 2;
    static const unsigned int m_Delays = 1 << m_logDelays;
};
} // namespace blas

} // namespace xf
#endif
