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

#include "vart/batch_tensor_buffer_view.hpp"

#include "./tensor_mirror_attrs.hpp"
#include "vart/runner.hpp"
#include "vitis/ai/env_config.hpp"
DEF_ENV_PARAM(DEBUG_GRAPH_RUNNER, "0")
namespace vart {

static std::vector<int32_t> change_shape(const std::vector<int32_t>& shape,
                                         size_t batch) {
  auto ret = shape;
  CHECK(!shape.empty());
  ret[0] = (int)batch;
  return ret;
}

BatchTensorBufferView::BatchTensorBufferView(vart::TensorBuffer* tensor_buffer,
                                             size_t batch_index, size_t batch)
    : vart::TensorBuffer(
          TensorMirrorAttrs::create(
              tensor_buffer->get_tensor(),
              change_shape(tensor_buffer->get_tensor()->get_shape(), batch))
              .release()),
      tensor_buffer_{tensor_buffer},
      batch_index_{batch_index},
      batch_{batch},
      tensor_{std::unique_ptr<xir::Tensor>(
          const_cast<xir::Tensor*>(get_tensor()))} {
  CHECK(tensor_buffer_ != nullptr);
  LOG_IF(INFO, ENV_PARAM(DEBUG_GRAPH_RUNNER))
      << "BatchTensorBufferView create "
      << "@" << (void*)this << " tensor_buffer:" << tensor_buffer_->to_string()
      << " batch_index " << batch_index_ << " batch_size " << batch_;
}

BatchTensorBufferView ::~BatchTensorBufferView() {
  LOG_IF(INFO, ENV_PARAM(DEBUG_GRAPH_RUNNER))
      << "BatchTensorBufferView destroyed "
      << "@" << (void*)this << " tensor_buffer:" << tensor_buffer_->to_string()
      << " batch_index " << batch_index_ << " batch_size " << batch_;
}

std::pair<uint64_t, size_t> BatchTensorBufferView::data(
    const std::vector<int> idx1) {
  auto idx = idx1;
  idx[0] = idx[0] + batch_index_;
  return limit_size(tensor_buffer_->data(idx));
}

BatchTensorBufferView::location_t BatchTensorBufferView::get_location() const {
  return tensor_buffer_->get_location();
}

/** @brief return the physical addresses for zero copy. */
std::pair<uint64_t, size_t> BatchTensorBufferView::data_phy(
    const std::vector<std::int32_t> idx1) {
  auto idx = idx1;
  idx[0] = idx[0] + batch_index_;
  return limit_size(tensor_buffer_->data_phy(idx));
}

std::pair<uint64_t, size_t> BatchTensorBufferView::limit_size(
    const std::pair<uint64_t, size_t>& ret) {
  return std::make_pair(ret.first, limit_size(ret.second));
}

size_t BatchTensorBufferView::limit_size(const size_t sz) {
  auto max_size = get_tensor()->get_data_size();
  return std::min((size_t)max_size, sz);
}

void BatchTensorBufferView::sync_for_read(uint64_t offset, size_t size) {
  return tensor_buffer_->sync_for_read(offset, size);
}
void BatchTensorBufferView::sync_for_write(uint64_t offset, size_t size) {
  return tensor_buffer_->sync_for_write(offset, size);
}

void BatchTensorBufferView::copy_from_host(size_t batch_idx, const void* buf,
                                           size_t size, size_t offset) {
  batch_idx = batch_idx + batch_index_;
  return tensor_buffer_->copy_from_host(batch_idx, buf, size, offset);
}

void BatchTensorBufferView::copy_to_host(size_t batch_idx, void* buf,
                                         size_t size, size_t offset) {
  batch_idx = batch_idx + batch_index_;
  return tensor_buffer_->copy_to_host(batch_idx, buf, size, offset);
}

void BatchTensorBufferView::update_batch_index(
    vart::TensorBuffer* tensor_buffer, size_t batch_idx) {
  auto my_tensor = get_tensor();
  auto other_tensor = tensor_buffer->get_tensor();
  auto my_shape = my_tensor->get_shape();
  auto other_shape = other_tensor->get_shape();
  CHECK_EQ(my_shape.size(), other_shape.size());
  CHECK_EQ(my_tensor->get_name(), other_tensor->get_name());
  for (auto i = 1u; i < my_shape.size(); ++i) {
    CHECK_EQ(my_shape[i], other_shape[i]);
  }
  CHECK(my_tensor->get_data_type() == other_tensor->get_data_type());
  CHECK_LE((int)batch_idx + my_shape[0], other_shape[0]);
  tensor_buffer_ = tensor_buffer;
  batch_index_ = (int)batch_idx;
}

}  // namespace vart
