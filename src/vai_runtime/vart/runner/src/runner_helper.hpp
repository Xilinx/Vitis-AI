/*
 * Copyright 2022-2023 Advanced Micro Devices Inc.
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
#pragma once
#include <vector>

#include "vart/runner.hpp"
#include "vitis/ai/dpu_runner.hpp"

std::string to_string(
    const std::vector<vitis::ai::TensorBuffer*>& tensor_buffers);
std::string to_string(const std::vector<vart::TensorBuffer*>& tensor_buffers);
std::string to_string(const std::vector<xir::Tensor*>& tensors);
std::string to_string(
    const std::vector<const vitis::ai::TensorBuffer*>& tensor_buffers);
std::string to_string(
    const std::vector<const vart::TensorBuffer*>& tensor_buffers);
std::string to_string(const std::vector<const xir::Tensor*>& tensors);
std::string to_string(const vitis::ai::TensorBuffer* tensor_buffers);
std::string to_string(const vart::TensorBuffer* tensor_buffers);
std::string to_string(const xir::Tensor* tensors);
std::string to_string(const vitis::ai::Tensor* tensors);

namespace vitis {
namespace ai {
std::vector<std::unique_ptr<vitis::ai::TensorBuffer>>
alloc_cpu_flat_tensor_buffers(const std::vector<vitis::ai::Tensor*>& tensors);
}  // namespace ai
}  // namespace vitis

namespace vart {
std::vector<std::int32_t> get_index_zeros(const xir::Tensor* tensor);
std::vector<std::unique_ptr<vart::TensorBuffer>> alloc_cpu_flat_tensor_buffers(
    const std::vector<const xir::Tensor*>& tensors);
std::unique_ptr<vart::TensorBuffer> alloc_cpu_flat_tensor_buffer(
    const xir::Tensor* tensor);

class CpuFlatTensorBuffer : public TensorBuffer {
 public:
  explicit CpuFlatTensorBuffer(void* data, const xir::Tensor* tensor);
  virtual ~CpuFlatTensorBuffer() = default;

 public:
  virtual std::pair<uint64_t, size_t> data(
      const std::vector<int> idx = {}) override;

 protected:
  void* data_;
};

class CpuFlatTensorBufferOwned : public CpuFlatTensorBuffer {
 public:
  explicit CpuFlatTensorBufferOwned(const xir::Tensor* tensor);
  virtual ~CpuFlatTensorBufferOwned() = default;

 private:
  std::vector<char> buffer_;
};
}  // namespace vart
