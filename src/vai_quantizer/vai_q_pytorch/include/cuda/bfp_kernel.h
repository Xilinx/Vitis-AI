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

#ifndef _INCLUDE_CUDA_BFP_KERNEL_H_
#define _INCLUDE_CUDA_BFP_KERNEL_H_

void LaunchBFPCUDAKernel(const float* input,
                         float* output,
                         int n,
                         int bit_width,
                         int block_size);

void LaunchBFPCUDAKernelV2(const float* input,
                           float* output,
                           const int n,
                           const int axis_size,
                           const int bit_width,
                           const int block_size);

void LaunchBFPPrimeCUDAKernel(const float* input,
                              float* output,
                              const int n,
                              const int axis_size,
                              const int bit_width,
                              const int block_size,
                              const int sub_block_size,
                              const int sub_block_shift_bits,
                              const int rounding_mode);

#endif // _INCLUDE_CUDA_BFP_KERNEL_H_
