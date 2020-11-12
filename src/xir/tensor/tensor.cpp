/*
 * Copyright 2019 Xilinx Inc.
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

#include "xir/tensor/tensor.hpp"
#include "xir/tensor/tensor_imp.hpp"

namespace xir {

std::unique_ptr<Tensor> Tensor::create(const std::string& name,
                                       const std::vector<std::int32_t>& shape,
                                       const DataType& data_type) {
  return std::unique_ptr<Tensor>{
      static_cast<Tensor*>(new TensorImp(name, shape, data_type))};
}

// TODO:: legacy API
std::unique_ptr<Tensor> Tensor::create(const std::string& name,
                                       const std::vector<std::int32_t>& shape,
                                       const DataType::Type& data_type,
                                       const std::int32_t bit_width) {
  return create(name, shape, DataType{data_type, bit_width});
}

std::unique_ptr<Tensor> Tensor::clone(const Tensor* tensor) {
  auto ret = std::unique_ptr<Tensor>{static_cast<Tensor*>(new TensorImp(
      tensor->get_name(), tensor->get_shape(), tensor->get_data_type()))};
  ret->set_attrs(tensor->get_attrs());

  return ret;
}

std::unique_ptr<Tensor> Tensor::clone(const Tensor* tensor,
                                      const std::string& name) {
  auto ret = clone(tensor);
  ret->rename(name);
  return ret;
}

}  // namespace xir
