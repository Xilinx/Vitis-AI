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

#include "./tensor_buffer_imp_host_phy.hpp"

#include <UniLog/UniLog.hpp>
#include <sstream>
#include <xir/tensor/tensor.hpp>

#include "vitis/ai/dim_calc.hpp"
#include "vitis/ai/env_config.hpp"

DEF_ENV_PARAM(DEBUG_TENSOR_BUFFER_ALLOCATOR, "0");
namespace vart {
namespace dpu {
static size_t align(size_t a, size_t b) {
  if (a % b == 0) {
    return a;
  }
  return (a / b + 1) * b;
}

static std::vector<std::unique_ptr<xir::BufferObject>> create_bo(
    size_t num_of_buffer_objects, size_t size, size_t device_id,
    const std::string& cu_name) {
  size = align(size, 1024u);
  auto ret =
      std::vector<std::unique_ptr<xir::BufferObject>>(num_of_buffer_objects);
  for (auto i = 0u; i < num_of_buffer_objects; ++i) {
    ret[i] = xir::BufferObject::create(size, device_id, cu_name);
  }
  return ret;
}

TensorBufferExtImpHostPhy::TensorBufferExtImpHostPhy(
    const xir::Tensor* tensor, location_t location, size_t device_id,
    const std::string& cu_name, std::shared_ptr<std::vector<char>> content)
    : TensorBufferExt(xir::Tensor::clone(tensor).release()),
      location_{location},
      tensor_{
          std::unique_ptr<xir::Tensor>(const_cast<xir::Tensor*>(get_tensor()))},
      buffer_objects_(
          create_bo((size_t)tensor->get_shape()[0],
                    tensor->get_data_size() / tensor->get_shape()[0],  //
                    device_id, cu_name)) {
  LOG_IF(INFO, ENV_PARAM(DEBUG_TENSOR_BUFFER_ALLOCATOR))
      << "TensorBufferExtImpHostPhy "
      << "@" << (void*)this << " created";
  if (content != nullptr && !content->empty()) {
    LOG_IF(INFO, ENV_PARAM(DEBUG_TENSOR_BUFFER_ALLOCATOR))
        << " init phy tensor buffer with " << content->size() << " bytes";
    UNI_LOG_CHECK(buffer_objects_.size() == 1u, VART_TENSOR_INFO_ERROR)
        << " for constant buffer object, we do not support batch ";
    buffer_objects_[0]->copy_from_host(&(*content)[0], content->size(), 0u);
  }
}
TensorBufferExtImpHostPhy::~TensorBufferExtImpHostPhy() {
  LOG_IF(INFO, ENV_PARAM(DEBUG_TENSOR_BUFFER_ALLOCATOR))
      << "TensorBufferExtImpHostPhy "
      << "@" << (void*)this << " destroyed";
}
TensorBuffer::location_t TensorBufferExtImpHostPhy::get_location() const {
  return location_;
}

std::pair<uint64_t, size_t> TensorBufferExtImpHostPhy::data_x(
    const std::vector<std::int32_t> idx_orig, int phy) {
  auto dims = get_tensor()->get_shape();
  auto batch = dims[0];
  UNI_LOG_CHECK((size_t)batch == buffer_objects_.size(), VART_TENSOR_INFO_ERROR);
  dims[0] = 1;
  auto calc = vitis::ai::DimCalc(dims);
  auto idx = std::vector<int32_t>(idx_orig);
  auto batch_idx = idx[0];
  idx[0] = 0;
  auto offset = calc.offset(idx);
  UNI_LOG_CHECK((size_t)batch_idx < buffer_objects_.size(), VART_TENSOR_INFO_ERROR)
    << " this=" << this->to_string();
  // auto size = buffer_objects_[batch_idx]->size() - offset;
  auto size = get_tensor()->get_data_size() - offset;
  LOG_IF(INFO, ENV_PARAM(DEBUG_TENSOR_BUFFER_ALLOCATOR)) << "size: " << size;
  uint64_t ret = 0u;
  if (phy) {
    ret = ((uint64_t)buffer_objects_[batch_idx]->phy()) + offset;
  } else {
    ret = ((uint64_t)buffer_objects_[batch_idx]->data_r()) + offset;
  }
  return std::make_pair(ret, size);
}

std::pair<uint64_t, size_t> TensorBufferExtImpHostPhy::data_phy(
    const std::vector<std::int32_t> idx) {
  return data_x(idx, 1);
}

std::pair<std::uint64_t, std::size_t> TensorBufferExtImpHostPhy::data(
    const std::vector<std::int32_t> idx) {
  if (location_ < TensorBuffer::location_t::HOST_PHY) {
    return std::make_pair((uint64_t)0u, (size_t)0u);
  }
  return data_x(idx, 0);
}

void TensorBufferExtImpHostPhy::sync_for_read(uint64_t offset, size_t size) {
  auto num_of_buffer_objects = buffer_objects_.size();
  for (auto i = 0u; i < num_of_buffer_objects; ++i) {
    buffer_objects_[i]->sync_for_read(offset, size);
  }
}

void TensorBufferExtImpHostPhy::sync_for_write(uint64_t offset, size_t size) {
  auto num_of_buffer_objects = buffer_objects_.size();
  for (auto i = 0u; i < num_of_buffer_objects; ++i) {
    buffer_objects_[i]->sync_for_write(offset, size);
  }
}

void TensorBufferExtImpHostPhy::copy_from_host(size_t batch_idx,
                                               const void* buf, size_t size,
                                               size_t offset) {
  UNI_LOG_CHECK(batch_idx < buffer_objects_.size(), VART_TENSOR_INFO_ERROR);
  buffer_objects_[batch_idx]->copy_from_host(buf, size, offset);
}

void TensorBufferExtImpHostPhy::copy_to_host(size_t batch_idx, void* buf,
                                             size_t size, size_t offset) {
  UNI_LOG_CHECK(batch_idx < buffer_objects_.size(), VART_TENSOR_INFO_ERROR);
  buffer_objects_[batch_idx]->copy_to_host(buf, size, offset);
}

XclBo TensorBufferExtImpHostPhy::get_xcl_bo(int batch_index) const {
  auto ret = XclBo{nullptr, 0u};
  UNI_LOG_CHECK(batch_index < (int)buffer_objects_.size(), VART_TENSOR_INFO_ERROR);
  auto bo = buffer_objects_[batch_index]->get_xcl_bo();
  ret.xcl_handle = bo.xcl_handle;
  ret.bo_handle = bo.bo_handle;
  return ret;
}

}  // namespace dpu
}  // namespace vart
