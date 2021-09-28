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

// File Name: hls_ssr_fft_adder_tree.hpp
#ifndef __HLS_SSR_FFT_ADDER_TREE_H__
#define __HLS_SSR_FFT_ADDER_TREE_H__
#include <complex>

#include "vitis_fft/hls_ssr_fft_traits.hpp"
#include "vitis_fft/hls_ssr_fft_enums.hpp"
#include "vitis_fft/fft_complex.hpp"

/*
 ========================================================================================
 -_-                                                                                   -_-
 -_-                                                                                   -_-
 -_-                                                                                   -_-
 -_-                                                                                   -_-
 -_-                                                                                   -_-
 -_- AdderTreeClass: defines a function called creatTreeLevel, which is used to        -_-
 -_- create adder tree, every tree level is defined by number of tree nodes at that    -_-
 -_- level, the recursion terminates when the tree level is created by two need.       -_-
 -_- Essentially ever tree level represents a set of N/2 binary adders doing addition  -_-
 -_- in parallel.                                                                      -_-
 -_-                                                                                   -_-
 -_-                                                                                   -_-
 -_-                                                                                   -_-
 -_-                                                                                   -_-
 -_-                                                                                   -_-
 -_-                                                                                   -_-
 ========================================================================================
 */

// Constant Defined For Debugging ///////////////////////////
//#define ADDER_TREE_PRINT_DEBUG_MESSAGES
/////////////////////////////////////////////////////////////
//=====================================================================================
// 2di : No  2 dimensional arrays in this block for reshaping and streaming used
//=====================================================================================

namespace xf {
namespace dsp {
namespace fft {

template <int t_numberOfTreeNodes>
struct AdderTreeClass {
    template <int t_treeLevel, bool isFirstStage, scaling_mode_enum t_scalingMode, typename T_in, typename T_out>
    void createTreeLevel(T_in p_data[t_numberOfTreeNodes], T_out& p_accumValue);
};
template <int t_numberOfTreeNodes>
template <int t_treeLevel, bool isFirstStage, scaling_mode_enum t_scalingMode, typename T_in, typename T_out>
void AdderTreeClass<t_numberOfTreeNodes>::createTreeLevel(T_in p_data[t_numberOfTreeNodes], T_out& p_accumValue) {
#pragma HLS INLINE
    typename ButterflyTraits<isFirstStage, t_scalingMode, T_in>::T_butterflyAccumType
        m_outputData[t_numberOfTreeNodes / 2];

#ifdef ADDER_TREE_PRINT_DEBUG_MESSAGES
    std::cout << "The Butter Fly  Type Used ::" << ButterflyTraits<t_scalingMode, T_in>::m_butterflyType << std::endl;
#endif

#pragma HLS ARRAY_PARTITION variable = m_outputData complete dim = 1

LOOP_TREE_LEVEL:
    for (int n = 0; n < t_numberOfTreeNodes / 2; n++) {
#pragma HLS UNROLL
        m_outputData[n] = p_data[2 * n] + p_data[2 * n + 1];
    }
    AdderTreeClass<t_numberOfTreeNodes / 2> AdderTreeClass_obj;

    AdderTreeClass_obj.template createTreeLevel<
        t_treeLevel + 1, isFirstStage, t_scalingMode,
        typename ButterflyTraits<isFirstStage, t_scalingMode, T_in>::T_butterflyAccumType, T_out>(m_outputData,
                                                                                                  p_accumValue);
}

template <>
template <int t_treeLevel, bool isFirstStage, scaling_mode_enum t_scaling_mode, typename T_in, typename T_out>
void AdderTreeClass<2>::createTreeLevel(T_in p_data[2], T_out& p_accumValue) {
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable = p_data complete dim = 1

    p_accumValue = p_data[1] + p_data[0];
}

} // end namespace fft
} // end namespace dsp
} // namespace xf

#endif //__HLS_SSR_FFT_ADDER_TREE_H__
