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
class BatchTensorBufferView : public vart::TensorBuffer {
 public:
  explicit BatchTensorBufferView(vart::TensorBuffer* tensor_buffer,
                                 size_t batch_idx, size_t batch);
  virtual ~BatchTensorBufferView();

 public:
  void update_batch_index(vart::TensorBuffer* tensor_buffer_, size_t batch_idx);

 private:
  virtual std::pair<uint64_t, size_t> data(
      const std::vector<int> idx = {}) override;
  virtual location_t get_location() const override;
  virtual std::pair<uint64_t, size_t> data_phy(
      const std::vector<std::int32_t> idx) override;

  virtual void sync_for_read(uint64_t offset, size_t size) override;
  virtual void sync_for_write(uint64_t offset, size_t size) override;
  virtual void copy_from_host(size_t batch_idx, const void* buf, size_t size,
                              size_t offset) override;
  virtual void copy_to_host(size_t batch_idx, void* buf, size_t size,
                            size_t offset) override;

 private:
  std::pair<uint64_t, size_t> limit_size(
      const std::pair<uint64_t, size_t>& ret);
  size_t limit_size(size_t sz);

 private:
  vart::TensorBuffer* tensor_buffer_;
  size_t batch_index_;
  size_t batch_;
  /// the tensor buffer has to own a tensor which have different
  /// shape.
  std::unique_ptr<xir::Tensor> tensor_;
};
}  // namespace vart
