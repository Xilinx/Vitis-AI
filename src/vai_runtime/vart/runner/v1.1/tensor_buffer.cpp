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
#include "vitis/ai/tensor_buffer.hpp"

namespace vitis {
namespace ai {

TensorBuffer::TensorBuffer(const Tensor* tensor) : tensor_{tensor} {}

const Tensor* TensorBuffer::get_tensor() const { return tensor_; }

CpuFlatTensorBuffer::CpuFlatTensorBuffer(void* data, const Tensor* tensor)
    : TensorBuffer{tensor}, data_{data} {}

std::pair<void*, size_t> CpuFlatTensorBuffer::data(const std::vector<int> idx) {
  if (idx.size() == 0) {
    return {data_,
            tensor_->get_element_num() * size_of(tensor_->get_data_type())};
  }
  auto dims = tensor_->get_dims();
  auto offset = 0;
  for (auto k = 0; k < tensor_->get_dim_num(); k++) {
    auto stride = 1;
    for (auto m = k + 1; m < tensor_->get_dim_num(); m++) {
      stride *= dims[m];
    }
    offset += idx[k] * stride;
  }

  auto dtype_size = size_of(tensor_->get_data_type());
  auto elem_num = tensor_->get_element_num();

  return std::make_pair((char*)data_ + offset * dtype_size,
                        (elem_num - offset) * dtype_size);
}

CpuFlatTensorBufferOwned::CpuFlatTensorBufferOwned(const Tensor* tensor)
    : CpuFlatTensorBuffer(nullptr, tensor),
      buffer_(tensor_->get_element_num() * size_of(tensor_->get_data_type())) {
  data_ = (void*)&buffer_[0];
}

}  // namespace ai
}  // namespace vitis
