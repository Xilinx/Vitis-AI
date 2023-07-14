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
#include "vart/zero_copy_helper.hpp"

namespace vart {
namespace assistant {

class XrtBoTensorBuffer : public vart::TensorBuffer {
 public:
  static std::unique_ptr<vart::TensorBuffer> create(vart::xrt_bo_t bo,
                                                    const xir::Tensor* tensor);

  explicit XrtBoTensorBuffer(vart::xrt_bo_t bo, const xir::Tensor* tensor);

  XrtBoTensorBuffer(const XrtBoTensorBuffer&) = delete;
  XrtBoTensorBuffer& operator=(const XrtBoTensorBuffer& other) = delete;

  virtual ~XrtBoTensorBuffer() = default;

 private:
  virtual std::pair<std::uint64_t, std::size_t> data(
      const std::vector<std::int32_t> idx = {}) override;
  virtual vart::TensorBuffer::location_t get_location() const override;
  virtual std::pair<uint64_t, size_t> data_phy(
      const std::vector<std::int32_t> idx) override;

  virtual void sync_for_read(uint64_t offset, size_t size) override;
  virtual void sync_for_write(uint64_t offset, size_t size) override;
  virtual void copy_from_host(size_t batch_idx, const void* buf, size_t size,
                              size_t offset) override;
  virtual void copy_to_host(size_t batch_idx, void* buf, size_t size,
                            size_t offset) override;

 private:
  const vart::xrt_bo_t bo_;
  uint64_t phy_addr_;
  int ddr_addr_;
  size_t size_;
};

}  // namespace assistant
}  // namespace vart
