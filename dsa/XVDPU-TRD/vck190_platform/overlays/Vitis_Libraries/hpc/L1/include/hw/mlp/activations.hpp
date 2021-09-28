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

#include "hls_stream.h"
#include "xf_blas.hpp"

using namespace xf::blas;

#ifndef XF_HPC_MLP_ACTIVATIONS_HPP
#define XF_HPC_MLP_ACTIVATIONS_HPP

/**
 * @file activations.hpp
 * @brief activation functions used in MLP
 * streaming modules for 2D-MLP2D kernels
 */

namespace xf {
namespace hpc {
namespace mlp {

// Activation functions
template <typename t_DataType>
t_DataType FcnScalePRelu(t_DataType x, int16_t p_PReluVal) {
    ap_int<16> l_PReluVal = p_PReluVal;
    ap_int<10> l_scaleVal;
    ap_int<6> l_alpha;
    l_scaleVal = l_PReluVal.range(15, 6);
    l_alpha = l_PReluVal.range(5, 0);
    t_DataType l_prePRelu = x;
#if BLAS_keepMacBits
    t_DataType l_postPRelu = (l_prePRelu < 0) ? (l_prePRelu * l_scaleVal.to_int()) >> l_alpha.to_int() : l_prePRelu;
#else
    t_DataType l_ceiling = static_cast<t_DataType>(l_alpha);
    t_DataType l_postPRelu = (l_prePRelu < 0)
                                 ? (l_prePRelu * l_scaleVal)
                                 : ((l_ceiling != 0) && (l_prePRelu >= l_ceiling)) ? l_ceiling : l_prePRelu;
#endif
    return l_postPRelu;
}

/** @brief relu (rectified linear unit) is a very common activation function in
 * deep neural network
 *
 * @param x is the input value
 */
template <typename t_DataType>
t_DataType relu(t_DataType x) {
    if (x > 0)
        return x;
    else
        return 0;
}

/** @brief sigmoid function is a very common activation function in MLP
 *
 * @param x is the input value
 */
template <typename t_DataType>
t_DataType sigmoid(t_DataType x) {
    t_DataType l_exp = hls::expf(-x);
    return 1.0f / (1.0f + l_exp);
}

/** @brief tansig function is used as an activation function in some MLPs
 *
 * @param x is the input value
 */
template <typename t_DataType>
t_DataType tansig(t_DataType x) {
    t_DataType l_exp = hls::expf(-2.0f * x);
    return 2.0f / (1.0f + l_exp) - 1.0f;
}

} // end namespace mlp
} // end namespace hpc
} // end namespace xf
#endif
