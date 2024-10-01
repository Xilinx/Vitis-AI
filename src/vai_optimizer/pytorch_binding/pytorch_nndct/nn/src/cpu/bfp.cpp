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

#include <c10/core/DeviceType.h>

#include "../../include/bfp.h"
#include "../../../../../include/cpu/bfp.h"

torch::Tensor& to_bfp(const torch::Tensor& tensor,
                      int64_t bit_width,
                      int64_t block_size,
                      torch::Tensor& out) {
  if (tensor.device().type() == c10::DeviceType::CUDA) {
    LOG(FATAL) << "BFP operation only supports CPU, but got device type: "
               << tensor.device().type();
  }
  CheckInputForBFP(tensor, bit_width, block_size);

  //torch::Tensor result = at::empty_like(tensor);
  const float* input = tensor.data_ptr<float>();
  float* output = out.data_ptr<float>();

  LaunchBFPCPUKernel(input, output, tensor.numel(), bit_width, block_size);
  return out;
}

torch::Tensor& to_bfp_v2(const torch::Tensor& tensor,
                         int64_t bit_width,
                         int64_t block_size,
                         torch::Tensor& out) {
  if (tensor.device().type() == c10::DeviceType::CUDA) {
    LOG(FATAL) << "BFP operation only supports CPU, but got device type: "
               << tensor.device().type();
  }
  CheckInputForBFPV2(tensor, bit_width, block_size);

  //torch::Tensor result = at::empty_like(tensor);
  const float* input = tensor.data_ptr<float>();
  float* output = out.data_ptr<float>();

  LaunchBFPCPUKernelV2(input, output, tensor.numel(), bit_width, block_size);
  return out;
}

torch::Tensor& to_bfp_prime_shared(const torch::Tensor& tensor,
                                   int64_t bit_width,
                                   int64_t block_size,
                                   int64_t sub_block_size,
                                   int64_t sub_block_shift_bits,
                                   int64_t rounding_mode,
                                   torch::Tensor& out) {
  if (tensor.device().type() == c10::DeviceType::CUDA) {
    LOG(FATAL) << "BFP operation only supports CPU, but got device type: "
               << tensor.device().type();
  }
  CheckInputForBFPPrime(tensor, bit_width, block_size, sub_block_size);

  const float* input = tensor.data_ptr<float>();
  float* output = out.data_ptr<float>();

  LaunchBFPPrimeCPUKernel(input, output, tensor.numel(), bit_width,
        block_size, sub_block_size, sub_block_shift_bits, rounding_mode);

  return out;
}

//TORCH_LIBRARY_IMPL(vai, CPU, m) {
//  m.impl("to_bfp_v2", to_bfp_cpu)
//}
