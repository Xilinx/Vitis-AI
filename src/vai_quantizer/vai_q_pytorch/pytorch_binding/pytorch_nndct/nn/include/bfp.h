/* Copyright 2023 Xilinx Inc.
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

#ifndef _NN_INCLUDE_BFP_H_
#define _NN_INCLUDE_BFP_H_

#include <torch/extension.h>

torch::Tensor& to_bfp(const torch::Tensor& tensor,
                      int64_t bit_width,
                      int64_t block_size,
                      torch::Tensor& out);

torch::Tensor& to_bfp_v2(const torch::Tensor& tensor,
                         int64_t bit_width,
                         int64_t block_size,
                         torch::Tensor& out);

torch::Tensor& to_bfp_prime_shared(const torch::Tensor& tensor,
                                   int64_t bit_width,
                                   int64_t block_size,
                                   int64_t sub_block_size,
                                   int64_t sub_block_shift_bits,
                                   int64_t rounding_mode,
                                   torch::Tensor& out);

#endif // _NN_INCLUDE_BFP_H_
