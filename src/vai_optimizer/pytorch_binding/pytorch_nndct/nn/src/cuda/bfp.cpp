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
#include "../../../../../include/cuda/bfp_kernel.h"
#include "../../../../../include/cpu/bfp.h"

torch::Tensor& to_bfp(const torch::Tensor& tensor,
                      int64_t bit_width,
                      int64_t block_size,
                      torch::Tensor& out) {
  CheckInputForBFP(tensor, bit_width, block_size);
  const float* input = tensor.data_ptr<float>();
  float* output = out.data_ptr<float>();

  LaunchBFPCUDAKernel(input, output, tensor.numel(), bit_width, block_size);
  return out;
}

torch::Tensor& to_bfp_v2(const torch::Tensor& tensor,
                         int64_t bit_width,
                         int64_t block_size,
                         torch::Tensor& out) {
  CheckInputForBFPV2(tensor, bit_width, block_size);

  const float* input = tensor.data_ptr<float>();
  float* output = out.data_ptr<float>();
  //int axis_size = tensor.size(tensor.dim() - 1);
  //LaunchBFPKernelV2(
  //    input, output, tensor.numel(), axis_size, bit_width, block_size);
  //return out;

  if (tensor.device().type() == c10::DeviceType::CUDA) {
    int axis_size = tensor.size(tensor.dim() - 1);
    LaunchBFPCUDAKernelV2(input, output, tensor.numel(), axis_size, bit_width,
        block_size);
  } else if (tensor.device().type() == c10::DeviceType::CPU) {
    LaunchBFPCPUKernelV2(input, output, tensor.numel(), bit_width, block_size);
  } else {
    LOG(FATAL) << "Unsupported device type: " << tensor.device().type();
  }
  return out;
}

torch::Tensor& to_bfp_prime_cuda(const torch::Tensor& tensor,
                                 int64_t bit_width,
                                 int64_t block_size,
                                 int64_t sub_block_size,
                                 int64_t sub_block_shift_bits,
                                 int64_t rounding_mode,
                                 torch::Tensor& out) {
  CheckInputForBFPPrime(tensor, bit_width, block_size, sub_block_size);

  const float* input = tensor.data_ptr<float>();
  float* output = out.data_ptr<float>();
  int axis_size = tensor.size(tensor.dim() - 1);
  LaunchBFPPrimeCUDAKernel(
      input, output, tensor.numel(), axis_size, bit_width, block_size,
      sub_block_size, sub_block_shift_bits, rounding_mode);
  return out;
}

torch::Tensor& to_bfp_prime_cpu(const torch::Tensor& tensor,
                                int64_t bit_width,
                                int64_t block_size,
                                int64_t sub_block_size,
                                int64_t sub_block_shift_bits,
                                int64_t rounding_mode,
                                torch::Tensor& out) {
  CheckInputForBFPPrime(tensor, bit_width, block_size, sub_block_size);

  const float* input = tensor.data_ptr<float>();
  float* output = out.data_ptr<float>();

  int num_blocks = tensor.numel() / block_size;

  for (int index = 0; index < num_blocks; index++) {
    BFPPrimeCPUKernel(
        input, output, tensor.numel(), index * block_size/*index*/, 1/*stride*/,
        bit_width, block_size, sub_block_size, sub_block_shift_bits,
        rounding_mode);
  }
  return out;
}

torch::Tensor& to_bfp_prime_shared(const torch::Tensor& tensor,
                                   int64_t bit_width,
                                   int64_t block_size,
                                   int64_t sub_block_size,
                                   int64_t sub_block_shift_bits,
                                   int64_t rounding_mode,
                                   torch::Tensor& out) {
  CheckInputForBFPPrime(tensor, bit_width, block_size, sub_block_size);

  const float* input = tensor.data_ptr<float>();
  float* output = out.data_ptr<float>();

  if (tensor.device().type() == c10::DeviceType::CUDA) {
    int axis_size = tensor.size(tensor.dim() - 1);
    LaunchBFPPrimeCUDAKernel(input, output, tensor.numel(), axis_size,
        bit_width, block_size, sub_block_size, sub_block_shift_bits,
        rounding_mode);
  } else if (tensor.device().type() == c10::DeviceType::CPU) {
    LaunchBFPPrimeCPUKernel(input, output, tensor.numel(), bit_width,
        block_size, sub_block_size, sub_block_shift_bits, rounding_mode);
  } else {
    LOG(FATAL) << "Unsupported device type: " << tensor.device().type();
  }
  return out;
}

//TORCH_LIBRARY_IMPL(vai, CUDA, m) {
//  m.impl("to_bfp_prime", to_bfp_prime_cuda);
//}
//
//TORCH_LIBRARY_IMPL(vai, CPU, m) {
//  m.impl("to_bfp_prime", to_bfp_prime_cpu);
//}
