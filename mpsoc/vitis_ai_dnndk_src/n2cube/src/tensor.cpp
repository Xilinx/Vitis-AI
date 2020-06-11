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

#include "vai/tensor.hpp"

#include <cassert>

namespace vitis {
namespace ai {

Tensor::Tensor(const std::string &name, const std::vector<std::int32_t> &dims,
               DataType data_type)
    : name_{name}, dims_{dims}, data_type_{data_type} {
  for (auto idx = 0; idx < static_cast<int>(dims_.size()); idx++) {
    assert(dims_[idx] >= 0);
  }
}

const std::string Tensor::get_name() const { return name_; }

const std::int32_t Tensor::get_dim_num() const { return dims_.size(); }

const std::int32_t Tensor::get_element_num() const {
  std::int32_t ret = 1;
  for (auto dim : dims_) {
    ret *= dim;
  }
  return ret;
}

const std::int32_t Tensor::get_dim_size(std::int32_t idx) const {
  assert(idx >= 0 && idx < static_cast<int>(dims_.size()));
  return dims_[idx];
}

const std::vector<std::int32_t> Tensor::get_dims() const { return dims_; }

const Tensor::DataType Tensor::get_data_type() const { return data_type_; }

std::size_t size_of(Tensor::DataType data_type) {
  switch (data_type) {
  case Tensor::DataType::INT8:
  case Tensor::DataType::UINT8:
    return 1;
  case Tensor::DataType::INT16:
  case Tensor::DataType::UINT16:
    return 2;
  case Tensor::DataType::INT32:
  case Tensor::DataType::UINT32:
  case Tensor::DataType::FLOAT:
    return 4;
  case Tensor::DataType::DOUBLE:
    return 8;
  case Tensor::DataType::UNKNOWN:
    return 0;
  }
  return 0;
}
}
}
