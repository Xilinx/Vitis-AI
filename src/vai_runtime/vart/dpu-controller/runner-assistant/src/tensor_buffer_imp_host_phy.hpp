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
#include <memory>
#include <vart/tensor_buffer.hpp>
#include <xir/buffer_object.hpp>
#include <xir/tensor/tensor.hpp>

namespace vart {
namespace dpu {
class TensorBufferExtImpHostPhy : public vart::TensorBufferExt {
 public:
  explicit TensorBufferExtImpHostPhy(
      const xir::Tensor* tensor, location_t location, size_t device_id,
      const std::string& cu_name, std::shared_ptr<std::vector<char>> content);
  virtual ~TensorBufferExtImpHostPhy();
  TensorBufferExtImpHostPhy(const TensorBufferExtImpHostPhy& other) = delete;
  TensorBufferExtImpHostPhy& operator=(const TensorBufferExtImpHostPhy& rhs) =
      delete;

 private:
  virtual location_t get_location() const override;
  virtual std::pair<std::uint64_t, std::size_t> data(
      const std::vector<std::int32_t> index) override;
  virtual std::pair<uint64_t, size_t> data_phy(
      const std::vector<std::int32_t> idx) override;
  virtual void sync_for_read(uint64_t offset, size_t size) override;
  virtual void sync_for_write(uint64_t offset, size_t size) override;
  virtual void copy_from_host(size_t batch_idx, const void* buf, size_t size,
                              size_t offset) override;
  virtual void copy_to_host(size_t batch_idx, void* buf, size_t size,
                            size_t offset) override;
  std::pair<uint64_t, size_t> data_x(const std::vector<std::int32_t> idx,
                                     int phy);
  virtual XclBo get_xcl_bo(int batch_index) const override;

 private:
  const location_t location_;
  std::unique_ptr<xir::Tensor> tensor_;
  std::vector<std::unique_ptr<xir::BufferObject>> buffer_objects_;
};
}  // namespace dpu
}  // namespace vart
