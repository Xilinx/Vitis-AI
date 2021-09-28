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

// File Name : hls_ssr_fft_super_sample.hpp
#ifndef HLS_SSR_FFT_SUPER_SAMPLE_H_
#define HLS_SSR_FFT_SUPER_SAMPLE_H_
#include <iostream>
template <typename T_T>
struct tagged_sample {
    T_T sample;
    bool valid;
};
template <int t_R, typename T_elemType>
struct SuperSampleContainer {
    T_elemType superSample[t_R];
    T_elemType& getVal(unsigned int i) { return (superSample[i]); }
    T_elemType& operator[](unsigned int p_Idx) { return (superSample[p_Idx]); }
    T_elemType* getValAddr() { return (&superSample[0]); }
    SuperSampleContainer() {
#pragma HLS inline
#pragma HLS array_partition variable = superSample complete
    }
    SuperSampleContainer(T_elemType p_initScalar) {
        for (int i = 0; i < t_R; ++i) {
            getVal(i) = p_initScalar;
        }
    }
    T_elemType shift(T_elemType p_ValIn) {
#pragma HLS inline
#pragma HLS data_pack variable = p_ValIn
        T_elemType l_valOut = superSample[t_R - 1];
    WIDE_TYPE_SHIFT:
        for (int i = t_R - 1; i > 0; --i) {
            T_elemType l_val = superSample[i - 1];
#pragma HLS data_pack variable = l_val
            superSample[i] = l_val;
        }
        superSample[0] = p_ValIn;
        return (l_valOut);
    }
    T_elemType shift() {
#pragma HLS inline
        T_elemType l_valOut = superSample[t_R - 1];
    WIDE_TYPE_SHIFT:
        for (int i = t_R - 1; i > 0; --i) {
            T_elemType l_val = superSample[i - 1];
#pragma HLS data_pack variable = l_val
            superSample[i] = l_val;
        }
        return (l_valOut);
    }
    T_elemType unshift() {
#pragma HLS inline
        T_elemType l_valOut = superSample[0];
    WIDE_TYPE_SHIFT:
        for (int i = 0; i < t_R - 1; ++i) {
            T_elemType l_val = superSample[i + 1];
#pragma HLS data_pack variable = l_val
            superSample[i] = l_val;
        }
        return (l_valOut);
    }
    static const SuperSampleContainer zero() {
        SuperSampleContainer l_zero;
#pragma HLS data_pack variable = l_zero
        for (int i = 0; i < t_R; ++i) {
            l_zero[i] = 0;
        }
        return (l_zero);
    }
    void print(std::ostream& os) {
        for (int i = 0; i < t_R; ++i) {
            os << getVal(i) << ", ";
        }
    }
    bool operator==(const SuperSampleContainer& rhs_in) const {
        for (int i = 0; i < t_R; ++i) {
            if (superSample[i] != rhs_in.superSample[i]) return false;
        }
        return true;
    }

    bool operator!=(const SuperSampleContainer& rhs_in) const {
        for (int i = 0; i < t_R; ++i) {
            if (superSample[i] != rhs_in.superSample[i]) return true;
        }
        return false;
    }
};
template <typename T2, int T1>
std::ostream& operator<<(std::ostream& os, SuperSampleContainer<T1, T2>& p_Val) {
    p_Val.print(os);
    return (os);
}
#endif // HLS_SSR_FFT_SUPER_SAMPLE_H_
