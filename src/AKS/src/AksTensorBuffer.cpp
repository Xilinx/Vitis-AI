/*
 * Copyright 2019 Xilinx Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <aks/AksTensorBuffer.h>

namespace AKS {

// explicit constructor
AksTensorBuffer::AksTensorBuffer (std::unique_ptr<xir::Tensor> tensor)
  : vart::TensorBuffer {tensor.get()}
  , buffer_(tensor->get_data_size())
  , tsor_(std::move(tensor)) 
{ }

// swap for AksTensorBuffer
void swap (AksTensorBuffer& a, AksTensorBuffer& b) noexcept
{
  using std::swap;
  swap(a.buffer_, b.buffer_);
  swap(a.tsor_, b.tsor_);
  swap(a.tensor_, b.tensor_);
}

// Copy Constructor
AksTensorBuffer::AksTensorBuffer (const AksTensorBuffer& src)
  : vart::TensorBuffer {xir::Tensor::clone(src.tsor_.get()).release()}
  , buffer_(src.buffer_)
  , tsor_(std::unique_ptr<xir::Tensor>(const_cast<xir::Tensor*>(
          this->get_tensor()))) 
{ }

// Copy Assignment
AksTensorBuffer& AksTensorBuffer::operator= (const AksTensorBuffer& src)
{
  AksTensorBuffer temp (src);
  swap(*this, temp);
  return *this;
}

// Move Constructor
AksTensorBuffer::AksTensorBuffer (AksTensorBuffer&& src) noexcept
  : vart::TensorBuffer {src.get_tensor()}
  , buffer_(std::exchange(src.buffer_, {}))
  , tsor_(std::exchange(src.tsor_, nullptr)) 
{
  tensor_ = std::exchange(src.tensor_, nullptr);
}

// Move Assignment
AksTensorBuffer& AksTensorBuffer::operator= (AksTensorBuffer&& src) noexcept
{
  AksTensorBuffer temp (std::move(src));
  swap(*this, temp);
  return *this;
}

// data method
std::pair<uint64_t, size_t> AksTensorBuffer::data(
    const std::vector<std::int32_t> idx) 
{
  auto data_ = static_cast<void*>(&(buffer_[0]));
  uint32_t size = std::ceil(tensor_->get_data_type().bit_width / 8.f);
  if (idx.size() == 0) {
    return {reinterpret_cast<uint64_t>(data_),
      tensor_->get_element_num() * size};
  }

  auto dims = tensor_->get_shape();
  std::vector<std::int32_t> indices (idx);
  if (indices.size() != dims.size()) {
    for (int i = indices.size(); i < dims.size(); ++i)
      indices.push_back(0);
  }
  auto offset = 0;
  for (auto k = 0; k < dims.size(); k++) {
    auto stride = 1;
    for (auto m = k + 1; m < dims.size(); m++) {
      stride *= dims[m];
    }
    offset += indices[k] * stride;
  }
  auto elem_num = tensor_->get_element_num();

  return { (reinterpret_cast<uint64_t>(data_) + (offset * size)),
           ((elem_num - offset) * size) };
}

}//namespace AKS
