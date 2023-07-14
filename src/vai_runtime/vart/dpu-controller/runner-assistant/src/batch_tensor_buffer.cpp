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

#include "vart/assistant/batch_tensor_buffer.hpp"

#include <UniLog/UniLog.hpp>
#include <memory>

#include "vart/runner.hpp"
#include "vart/tensor_buffer.hpp"
#include "vart/tensor_mirror_attrs.hpp"

namespace vart {
namespace assistant {
std::unique_ptr<vart::TensorBuffer> BatchTensorBuffer::create(
    const std::vector<vart::TensorBuffer*>& tensor_buffers) {
  return std::make_unique<BatchTensorBuffer>(tensor_buffers);
}

static std::unique_ptr<xir::Tensor> create_tensor(
    const std::vector<vart::TensorBuffer*>& from) {
  auto ret = std::unique_ptr<xir::Tensor>();
  auto batch = 0;
  auto batch_dims = std::vector<int32_t>();
  auto name = std::string();
  auto data_type = xir::DataType{};
  const xir::Tensor* tensor = nullptr;
  for (auto& b : from) {
    auto t = b->get_tensor();
    if (tensor == nullptr) {
      tensor = t;
    }
    UNI_LOG_CHECK(t != nullptr, VART_XRT_NULL_PTR)
        << "cannot get tensor from the tensor buffer";
    auto dims = t->get_shape();
    UNI_LOG_CHECK(!dims.empty(), VART_TENSOR_INFO_ERROR)
        << "dims.size() " << dims.size();
    batch = batch + dims[0];
    if (batch_dims.empty()) {
      batch_dims = dims;
      name = t->get_name();
      data_type = t->get_data_type();
    } else {
      UNI_LOG_CHECK(batch_dims.size() == dims.size(), VART_TENSOR_INFO_ERROR);
      for (auto i = 1u; i < batch_dims.size(); ++i) {
        UNI_LOG_CHECK(batch_dims[i] == dims[i], VART_TENSOR_INFO_ERROR)
            << " all tensor buffer should have same shepe except the batch, "
               "i.e. the first dimension. i = "
            << i;
      }
      UNI_LOG_CHECK(name == t->get_name(), VART_TENSOR_INFO_ERROR)
          << "all tensor should have same name";
      UNI_LOG_CHECK((int)data_type.type == (int)t->get_data_type().type,
                    VART_TENSOR_INFO_ERROR)
          << "all tensor should have data_type";
      UNI_LOG_CHECK(data_type.bit_width == t->get_data_type().bit_width,
                    VART_TENSOR_INFO_ERROR)
          << "all tensor should have bit_width";
    }
  }
  batch_dims[0] = batch;
  ret = vart::TensorMirrorAttrs::create(tensor, batch_dims, data_type);
  return ret;
}

static vart::TensorBuffer::location_t my_get_location(
    const std::vector<vart::TensorBuffer*>& tensor_buffers) {
  UNI_LOG_CHECK(!tensor_buffers.empty(), VART_TENSOR_INFO_ERROR);
  auto ret = tensor_buffers[0]->get_location();
  for (auto i = 1u; i < tensor_buffers.size(); ++i) {
    UNI_LOG_CHECK(ret == tensor_buffers[i]->get_location(),
                  VART_TENSOR_INFO_ERROR)
        << "all tensor buffers must have the same location: tensor_buffers[i]="
        << tensor_buffers[i]->to_string()
        << "; the first tensor buffer=" << tensor_buffers.front()->to_string();
  }
  return ret;
}

BatchTensorBuffer::BatchTensorBuffer(
    const std::vector<vart::TensorBuffer*>& tensor_buffers)
    : vart::TensorBuffer(create_tensor(tensor_buffers).release()),
      tensor_buffers_(tensor_buffers),
      // capture the tensor again,
      tensor_(const_cast<xir::Tensor*>(get_tensor())),
      location_{my_get_location(tensor_buffers_)} {
  UNI_LOG_CHECK(!tensor_buffers_.empty(), VART_TENSOR_INFO_ERROR);
}

BatchTensorBuffer ::~BatchTensorBuffer() {}

std::pair<uint64_t, size_t> BatchTensorBuffer::xdata(
    const std::vector<int>& idx, int is_phy) {
  if (idx.empty()) {
    if (is_phy) {
      return tensor_buffers_[0]->data_phy(idx);
    } else {
      return tensor_buffers_[0]->data(idx);
    }
  }
  auto tb_pair = get_tb_idx(idx[0]);
  auto idx2 = idx;
  idx2[0] = tb_pair.second;
  auto ret = std::pair<uint64_t, size_t>();
  if (is_phy) {
    ret = tensor_buffers_[tb_pair.first]->data_phy(idx2);
  } else {
    ret = tensor_buffers_[tb_pair.first]->data(idx2);
  }
  return ret;
}
std::pair<uint64_t, size_t> BatchTensorBuffer::data(
    const std::vector<int> idx) {
  return xdata(idx, 0);
}

std::pair<uint64_t, size_t> BatchTensorBuffer::data_phy(
    const std::vector<std::int32_t> idx) {
  return xdata(idx, 1);
}

vart::TensorBuffer::location_t BatchTensorBuffer::get_location() const {
  return location_;
}

void BatchTensorBuffer::sync_for_read(uint64_t offset, size_t size) {
  for (auto tb : tensor_buffers_) {
    tb->sync_for_read(offset, size);
  }
  return;
}

void BatchTensorBuffer::sync_for_write(uint64_t offset, size_t size) {
  for (auto tb : tensor_buffers_) {
    tb->sync_for_write(offset, size);
  }
  return;
}

std::pair<int, int> BatchTensorBuffer::get_tb_idx(size_t batch_idx) {
  auto batch = 0;
  auto batch_idx_tb = batch_idx;
  for (auto tb_idx = 0; tb_idx < (int)tensor_buffers_.size(); tb_idx++) {
    auto dims = tensor_buffers_[tb_idx]->get_tensor()->get_shape();
    UNI_LOG_CHECK(!dims.empty(), VART_TENSOR_INFO_ERROR);
    batch = batch + dims[0];
    if ((int)batch_idx < batch) {
      return std::make_pair(tb_idx, batch_idx_tb);
    }
    batch_idx_tb -= dims[0];
  }
  CHECK(false) << "batch_idx is incorrect, batch_idx " << batch_idx
               << " batch_total " << batch;
  return std::make_pair(0, 0);
}
void BatchTensorBuffer::copy_from_host(size_t batch_idx, const void* buf,
                                       size_t size, size_t offset) {
  auto tb_idx = get_tb_idx(batch_idx);
  tensor_buffers_[tb_idx.first]->copy_from_host(tb_idx.second, buf, size,  //
                                                offset);
}

void BatchTensorBuffer::copy_to_host(size_t batch_idx, void* buf, size_t size,
                                     size_t offset) {
  auto tb_idx = get_tb_idx(batch_idx);
  tensor_buffers_[tb_idx.first]->copy_to_host(tb_idx.second, buf, size,  //
                                              offset);
}

}  // namespace assistant
}  // namespace vart
