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

#include "./runner_helper.hpp"

#include <sstream>
#include <xir/tensor/tensor.hpp>

static std::string to_string(const std::pair<void*, size_t>& v) {
  std::ostringstream str;
  str << "@(" << v.first << "," << std::dec << v.second << ")";
  return str.str();
}

static std::string to_string(const std::pair<uint64_t, size_t>& v) {
  std::ostringstream str;
  str << "@(0x" << std::hex << v.first << "," << std::dec << v.second << ")";
  return str.str();
}

template <typename T>
std::string to_string(T begin, T end, char s = '[', char e = ']',
                      char sep = ',');

std::string to_string(const vitis::ai::TensorBuffer* tensor_buffer) {
  auto dims = tensor_buffer->get_tensor()->get_dims();
  auto idx = dims;
  std::fill(idx.begin(), idx.end(), 0u);
  auto batch_size = dims[0];
  auto data_size = std::vector<std::pair<void*, size_t>>(batch_size);
  for (auto batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
    idx[0] = batch_idx;
    data_size[batch_idx] =
        const_cast<vitis::ai::TensorBuffer*>(tensor_buffer)->data(idx);
  }
  std::ostringstream str;
  str << "TensorBuffer@" << (void*)tensor_buffer
      << "{data=" << to_string(data_size.begin(), data_size.end())
      << ", tensor=" << to_string(tensor_buffer->get_tensor()) << "}";
  return str.str();
}

std::string to_string(const vart::TensorBuffer* tensor_buffer) {
  auto dims = tensor_buffer->get_tensor()->get_shape();
  auto idx = dims;
  std::fill(idx.begin(), idx.end(), 0u);
  auto batch_size = dims[0];
  auto data_size = std::vector<std::pair<uint64_t, size_t>>(batch_size);
  for (auto batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
    idx[0] = batch_idx;
    data_size[batch_idx] =
        const_cast<vart::TensorBuffer*>(tensor_buffer)->data(idx);
  }
  std::ostringstream str;
  str << "TensorBuffer@" << (void*)tensor_buffer
      << "{data=" << to_string(data_size.begin(), data_size.end())
      << ", tensor=" << to_string(tensor_buffer->get_tensor()) << "}";
  return str.str();
}

std::string to_string(const xir::Tensor* tensor) {
  std::ostringstream str;
  auto dims = tensor->get_shape();
  str << "Tensor@" << (void*)tensor << "{"                //
      << "name=" << tensor->get_name()                    //
      << ",dims=" << to_string(dims.begin(), dims.end())  //
      << "}"                                              //
      ;
  return str.str();
}

std::string to_string(const vitis::ai::Tensor* tensor) {
  std::ostringstream str;
  auto dims = tensor->get_dims();
  str << "Tensor@" << (void*)tensor << "{"                //
      << "name=" << tensor->get_name()                    //
      << ",dims=" << to_string(dims.begin(), dims.end())  //
      << "}"                                              //
      ;
  return str.str();
}

std::string to_string(const std::vector<vart::TensorBuffer*>& tensor_buffers) {
  return to_string(tensor_buffers.begin(), tensor_buffers.end());
}

std::string to_string(
    const std::vector<vitis::ai::TensorBuffer*>& tensor_buffers) {
  return to_string(tensor_buffers.begin(), tensor_buffers.end());
}
std::string to_string(const std::vector<xir::Tensor*>& tensors) {
  return to_string(tensors.begin(), tensors.end());
}
std::string to_string(
    const std::vector<const vitis::ai::TensorBuffer*>& tensor_buffers) {
  return to_string(tensor_buffers.begin(), tensor_buffers.end());
}
std::string to_string(
    const std::vector<const vart::TensorBuffer*>& tensor_buffers) {
  return to_string(tensor_buffers.begin(), tensor_buffers.end());
}
std::string to_string(const std::vector<const xir::Tensor*>& tensors) {
  return to_string(tensors.begin(), tensors.end());
}
/*MSVC NOTE: msvc complain with a strange error code C2665, it is because
 * std::to_string is not imported.*/
/*GCC NOTE: some version of gcc import std::string AUTOMATICALLY, which pollute
 * the namespace, using std::string hopefully works for all platforms and
 * compiler versions. */
using std::to_string;

template <typename T>
std::string to_string(T begin, T end, char s, char e, char sep) {
  std::ostringstream str;
  str << s;
  int c = 0;
  for (auto it = begin; it != end; ++it) {
    if (c++ != 0) {
      str << sep;
    };
    str << to_string(*it);
  }
  str << e;
  return str.str();
}

namespace vitis {
namespace ai {
std::vector<std::unique_ptr<vitis::ai::TensorBuffer>>
alloc_cpu_flat_tensor_buffers(const std::vector<vitis::ai::Tensor*>& tensors) {
  auto ret =
      std::vector<std::unique_ptr<vitis::ai::TensorBuffer>>(tensors.size());
  for (auto i = 0u; i < tensors.size(); ++i) {
    ret[i] = std::unique_ptr<vitis::ai::TensorBuffer>(
        new vitis::ai::CpuFlatTensorBufferOwned(tensors[i]));
  }
  return ret;
}
}  // namespace ai
}  // namespace vitis

namespace vart {
std::vector<std::int32_t> get_index_zeros(const xir::Tensor* tensor) {
  auto ret = tensor->get_shape();
  std::fill(ret.begin(), ret.end(), 0);
  return ret;
}

CpuFlatTensorBuffer::CpuFlatTensorBuffer(void* data, const xir::Tensor* tensor)
    : TensorBuffer{tensor}, data_{data} {}

std::pair<uint64_t, size_t> CpuFlatTensorBuffer::data(
    const std::vector<int> idx) {
  if (idx.size() == 0) {
    return {reinterpret_cast<uint64_t>(data_), tensor_->get_data_size()};
  }
  auto dims = tensor_->get_shape();
  auto offset = 0;
  for (auto k = 0u; k < dims.size(); k++) {
    auto stride = 1;
    for (auto m = k + 1; m < dims.size(); m++) {
      stride *= dims[m];
    }
    offset += idx[k] * stride;
  }

  auto dtype_size = tensor_->get_data_type().bit_width / 8;
  auto elem_num = tensor_->get_element_num();

  return std::make_pair(reinterpret_cast<uint64_t>(data_) + offset * dtype_size,
                        (elem_num - offset) * dtype_size);
}

static size_t tensor_real_size(const xir::Tensor* tensor) {
  auto ret = tensor->get_data_size();
  if (tensor->has_attr("stride")) {
    auto strides = tensor->get_attr<std::vector<std::int32_t>>("stride");
    ret = strides.at(0);
  }
  return ret;
}
CpuFlatTensorBufferOwned::CpuFlatTensorBufferOwned(const xir::Tensor* tensor)
    : CpuFlatTensorBuffer(nullptr, tensor), buffer_(tensor_real_size(tensor_)) {
  data_ = (void*)&buffer_[0];
}

std::vector<std::unique_ptr<vart::TensorBuffer>> alloc_cpu_flat_tensor_buffers(
    const std::vector<const xir::Tensor*>& tensors) {
  auto ret = std::vector<std::unique_ptr<vart::TensorBuffer>>(tensors.size());
  for (auto i = 0u; i < tensors.size(); ++i) {
    ret[i] = std::unique_ptr<vart::TensorBuffer>(
        new CpuFlatTensorBufferOwned(tensors[i]));
  }
  return ret;
}

std::unique_ptr<vart::TensorBuffer> alloc_cpu_flat_tensor_buffer(
    const xir::Tensor* tensor) {
  auto ret =
      std::unique_ptr<vart::TensorBuffer>(new CpuFlatTensorBufferOwned(tensor));
  return ret;
}
}  // namespace vart
