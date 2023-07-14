/*
 * Copyright 2022-2023 Advanced Micro Devices Inc.
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

#ifndef __COMMON_H__
#define __COMMON_H__

#include <cmath>

/* header file for Vitis AI unified API */
#include <vart/runner.hpp>
#include <xir/tensor/tensor.hpp>
#include <xir/util/data_type.hpp>

std::pair<char*, size_t> read_binary_file(const std::string &file_name)
{
  CHECK(file_name.empty()==0);

  std::ifstream stream(file_name.c_str());
  stream.seekg(0, stream.end);
  size_t size = stream.tellg();
  stream.seekg(0, stream.beg);
  
  //LOG(INFO)<< file_name<< ", size " << size;
  
  char *file_ptr = new char[size];
  stream.read(file_ptr, size);

  return std::make_pair(file_ptr, size) ;
}

class CpuFlatTensorBuffer : public vart::TensorBuffer {
 public:
  explicit CpuFlatTensorBuffer(void* data, const xir::Tensor* tensor)
      : TensorBuffer{tensor}, data_{data} {}
  virtual ~CpuFlatTensorBuffer() = default;

 public:
  virtual std::pair<uint64_t, size_t> data(
      const std::vector<int> idx = {}) override {
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

 private:
  void* data_;
};

#endif
