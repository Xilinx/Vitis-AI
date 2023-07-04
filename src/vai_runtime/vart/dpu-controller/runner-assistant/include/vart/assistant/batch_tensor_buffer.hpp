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
namespace assistant {
/// @brief BatchTensorBuffer does not own the underlying tensor buffers.
/// BatchTensorBuffer will manage the input tensor buffers and from the view of
/// consumers, it seems to be a tensor buffer combining the batches of all input
/// tensor buffers
/// note: the input tensor buffers should have the same shape except the batch
/// for example:
/// two input tensor buffers: [3,224,224,3] and [5,224,224,3]
/// BatchTensorBuffer object seems to be [8,224,224,3]
class BatchTensorBuffer : public vart::TensorBuffer {
 public:
  static std::unique_ptr<vart::TensorBuffer> create(
      const std::vector<vart::TensorBuffer*>& tensor_buffers);

 public:
  explicit BatchTensorBuffer(
      const std::vector<vart::TensorBuffer*>& tensor_buffers);
  virtual ~BatchTensorBuffer();

 public:
  virtual std::pair<uint64_t, size_t> data(
      const std::vector<int> idx = {}) override;

  virtual std::pair<uint64_t, size_t> data_phy(
      const std::vector<std::int32_t> idx) override;

  TensorBuffer* get_tensor_buffer(size_t idx) { return tensor_buffers_[idx]; }

  virtual vart::TensorBuffer::location_t get_location() const override;

  virtual void sync_for_read(uint64_t offset, size_t size) override;
  virtual void sync_for_write(uint64_t offset, size_t size) override;
  virtual void copy_from_host(size_t batch_idx, const void* buf, size_t size,
                              size_t offset) override;
  virtual void copy_to_host(size_t batch_idx, void* buf, size_t size,
                            size_t offset) override;

 private:
  std::pair<uint64_t, size_t> xdata(const std::vector<int>& idx, int is_phy);
  std::pair<int, int> get_tb_idx(size_t batch_idx);

 private:
  std::vector<TensorBuffer*> tensor_buffers_;
  std::unique_ptr<xir::Tensor> tensor_;
  vart::TensorBuffer::location_t location_;
};
}  // namespace assistant
}  // namespace vart
