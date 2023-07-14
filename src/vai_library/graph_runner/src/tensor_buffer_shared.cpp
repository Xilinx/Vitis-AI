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
#include "./tensor_buffer_shared.hpp"

#include "vitis/ai/env_config.hpp"
#include "xir/tensor/tensor.hpp"
DEF_ENV_PARAM(DEBUG_GRAPH_RUNNER, "0")

namespace vart {

TensorBufferShared::TensorBufferShared(std::shared_ptr<vart::TensorBuffer> real,
                                       const xir::Tensor* tensor)
    : vart::TensorBuffer{xir::Tensor::clone(tensor).release()},
      tensor_{
          std::unique_ptr<xir::Tensor>(const_cast<xir::Tensor*>(get_tensor()))},
      real_(real) {
  LOG_IF(INFO, ENV_PARAM(DEBUG_GRAPH_RUNNER))
      << "TensorBufferShared create "
      << "@" << (void*)this << " tensor :" << tensor_->get_name();
}

TensorBufferShared::~TensorBufferShared() {
  LOG_IF(INFO, ENV_PARAM(DEBUG_GRAPH_RUNNER))
      << "TensorBufferShared destroyed "
      << "@" << (void*)this << " tensor :" << tensor_->get_name();
}

std::pair<std::uint64_t, std::size_t> TensorBufferShared::data(
    const std::vector<std::int32_t> idx) {
  return real_->data(idx);
}

TensorBuffer::location_t TensorBufferShared::get_location() const {
  return real_->get_location();
}

/** @brief return the physical addresses for zero copy. */
std::pair<uint64_t, size_t> TensorBufferShared::data_phy(
    const std::vector<std::int32_t> idx) {
  return real_->data_phy(idx);
}

void TensorBufferShared::sync_for_read(uint64_t offset, size_t size) {
  real_->sync_for_read(offset, size);
}

void TensorBufferShared::sync_for_write(uint64_t offset, size_t size) {
  real_->sync_for_write(offset, size);
}

void TensorBufferShared::copy_from_host(size_t batch_idx, const void* buf,
                                        size_t size, size_t offset) {
  real_->copy_from_host(batch_idx, buf, size, offset);
}

void TensorBufferShared::copy_to_host(size_t batch_idx, void* buf, size_t size,
                                      size_t offset) {
  real_->copy_to_host(batch_idx, buf, size, offset);
}

}  // namespace vart
