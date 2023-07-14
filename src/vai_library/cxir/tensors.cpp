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

#include "./util.hpp"
#include "xir/tensor/tensor_imp.hpp"
namespace xir {
class c_api {
 public:
  static xir_string_t tensor_get_name(xir_tensor_t tensor) {
    auto self = static_cast<xir::TensorImp*>(tensor);
    return conv_to_xir_string(self->name_);
  }
  static xir_attrs_t tensor_get_attrs(xir_tensor_t tensor) {
    auto self = static_cast<xir::TensorImp*>(tensor);
    return static_cast<xir_attrs_t>(self->attrs_.get());
  }
};
}  // namespace xir

extern "C" xir_tensor_t xir_tensor_create(xir_string_t name,
                                          const int32_t* dims,
                                          const int32_t dim_num,
                                          enum xir_tensor_data_type_t data_type,
                                          const int32_t bit_width) {
  auto vdims = std::vector<int32_t>(dims, dims + dim_num);
  auto n_data_type =
      xir::DataType{static_cast<xir::DataType::Type>(data_type), bit_width};
  auto tensor =
      xir::Tensor::create(conv_to_std_string(name), vdims, n_data_type);
  return static_cast<xir_tensor_t>(tensor.release());
}
extern "C" int xir_tensor_destroy(xir_tensor_t tensor) {
  auto t = static_cast<xir::Tensor*>(tensor);
  delete t;
  return 0;
}
extern "C" xir_string_t xir_tensor_get_name(xir_tensor_t tensor) {
  return xir::c_api::tensor_get_name(tensor);
}
extern "C" int32_t xir_tensor_get_bit_width(xir_tensor_t tensor) {
  return static_cast<xir::Tensor*>(tensor)->get_data_type().bit_width;
}
extern "C" int32_t xir_tensor_get_dim_size(xir_tensor_t tensor, int32_t idx) {
  return static_cast<xir::Tensor*>(tensor)->get_shape().at(idx);
}

extern "C" int32_t xir_tensor_get_dim_num(xir_tensor_t tensor) {
  return static_cast<xir::Tensor*>(tensor)->get_shape().size();
}

extern "C" enum xir_tensor_data_type_t xir_tensor_get_data_type(
    xir_tensor_t tensor) {
  return static_cast<xir_tensor_data_type_t>(
      static_cast<xir::Tensor*>(tensor)->get_data_type().type);
}
extern "C" int64_t xir_tensor_get_element_num(xir_tensor_t tensor) {
  return static_cast<xir::Tensor*>(tensor)->get_element_num();
}

extern "C" uint64_t xir_tensor_get_data_size(xir_tensor_t tensor) {
  return static_cast<xir::Tensor*>(tensor)->get_data_size();
}
extern "C" xir_attrs_t xir_tensor_get_attrs(xir_tensor_t tensor) {
  return xir::c_api::tensor_get_attrs(tensor);
}
extern "C" void xir_tensor_set_attrs(xir_tensor_t tensor, xir_attrs_t attrs) {
  static_cast<xir::Tensor*>(tensor)->set_attrs(
      std::unique_ptr<xir::Attrs>(static_cast<xir::Attrs*>(attrs)));
}
