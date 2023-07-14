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

#include <xir/tensor/tensor.hpp>

#include "vart/tensor_buffer.hpp"

namespace vart {

/// @brief BatchTensorBuffer does not own the underlying tensor buffers.
class BatchTensorBuffer : public vart::TensorBuffer {
 public:
  explicit BatchTensorBuffer(
      const std::vector<vart::TensorBuffer*>& tensor_buffers);
  virtual ~BatchTensorBuffer();

 public:
  virtual std::pair<uint64_t, size_t> data(
      const std::vector<int> idx = {}) override;
  TensorBuffer* get_tensor_buffer(size_t idx) { return tensor_buffers_[idx]; }

 private:
  std::vector<TensorBuffer*> tensor_buffers_;
  std::unique_ptr<xir::Tensor> tensor_;
};
}  // namespace vart
