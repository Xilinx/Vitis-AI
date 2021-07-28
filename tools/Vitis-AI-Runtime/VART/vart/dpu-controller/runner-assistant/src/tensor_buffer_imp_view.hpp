/*
 * Copyright 2019 Xilinx Inc.
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
#include <memory>
#include <string>
#include <vart/tensor_buffer.hpp>

namespace vart {
namespace dpu {
class TensorBufferExtImpView : public vart::TensorBuffer {
 public:
 public:
  explicit TensorBufferExtImpView(
      const xir::Tensor* tensor, size_t offset,
      std::shared_ptr<vart::TensorBuffer> backstore);

  TensorBufferExtImpView(const TensorBufferExtImpView&) = delete;
  TensorBufferExtImpView& operator=(const TensorBufferExtImpView& other) =
      delete;

  virtual ~TensorBufferExtImpView();

 private:
  virtual location_t get_location() const override;
  virtual std::pair<uint64_t, size_t> data_phy(
      const std::vector<std::int32_t> idx) override;
  virtual std::pair<std::uint64_t, std::size_t> data(
      const std::vector<std::int32_t> idx = {}) override;
  virtual void sync_for_read(uint64_t offset, size_t size) override;
  virtual void sync_for_write(uint64_t offset, size_t size) override;
  virtual void copy_from_host(size_t batch_idx, const void* buf, size_t size,
                              size_t offset) override;
  virtual void copy_to_host(size_t batch_idx, void* buf, size_t size,
                            size_t offset) override;
  std::pair<uint64_t, size_t> data_x(const std::vector<std::int32_t> idx,
                                     int phy);

 private:
  std::unique_ptr<xir::Tensor> tensor_;
  const size_t offset_;
  std::shared_ptr<vart::TensorBuffer> backstore_;
};
}  // namespace dpu
}  // namespace vart
