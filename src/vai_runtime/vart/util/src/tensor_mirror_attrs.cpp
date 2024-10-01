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

#include "vart/tensor_mirror_attrs.hpp"

#include <sstream>
namespace vart {

std::unique_ptr<TensorMirrorAttrs> TensorMirrorAttrs::create(
    const xir::Tensor* other, const std::vector<std::int32_t> shape,
    const xir::DataType data_type) {
  return std::make_unique<TensorMirrorAttrs>(other, shape, data_type);
}

TensorMirrorAttrs::TensorMirrorAttrs(const xir::Tensor* other,
                                     const std::vector<std::int32_t> shape,
                                     const xir::DataType data_type)
    : other_{other},
      shape_{shape.empty() ? other->get_shape() : shape},
      data_type_{data_type.bit_width == 0 ? other->get_data_type()
                                          : data_type} {
  LOG_IF(INFO, false) << "shape_.size() " << shape_.size()
                      << " TensorMirrorAttrs@" << (void*)this << " "
                      << other->to_string();
}

TensorMirrorAttrs::~TensorMirrorAttrs() {
  LOG_IF(INFO, false) << "shape_.size() " << shape_.size()
                      << " ~TensorMirrorAttrs@" << (void*)this << " "
                      << other_->to_string();
}
const std::string TensorMirrorAttrs::get_name() const {
  return other_->get_name();
}

const xir::Op* TensorMirrorAttrs::get_producer() const {
  return other_->get_producer();
}

xir::Op* TensorMirrorAttrs::get_producer() {
  LOG(FATAL) << "not allowed";
  return nullptr;
}
const std::vector<std::int32_t> TensorMirrorAttrs::get_shape() const {
  return shape_;
}
const std::vector<std::int32_t> TensorMirrorAttrs::get_dims() const {
  return shape_;
}

const std::int32_t TensorMirrorAttrs::get_dim_num() const {
  return (int32_t)get_dims().size();
}

const std::int32_t TensorMirrorAttrs::get_dim_size(std::int32_t idx) const {
  return get_shape()[idx];
}

const std::int64_t TensorMirrorAttrs::get_element_num() const {
  int64_t r = 1;
  for (auto x : get_shape()) {
    r = r * x;
  }
  return r;
}

const xir::DataType& TensorMirrorAttrs::get_data_type() const {
  return data_type_;
}
const std::int32_t TensorMirrorAttrs::get_bit_width() const {
  return data_type_.bit_width;
}
const std::uint64_t TensorMirrorAttrs::get_data_size() const {
  return get_element_num() * get_bit_width() / 8;
}

std::unique_ptr<xir::Attrs> TensorMirrorAttrs::get_attrs() const {
  return other_->get_attrs();
}

TensorMirrorAttrs* TensorMirrorAttrs::set_attrs(
    std::unique_ptr<xir::Attrs> attrs) {
  LOG(FATAL) << "not allowed";
  return nullptr;
}
const bool TensorMirrorAttrs::has_attr(const std::string& key) const {
  return other_->has_attr(key);
}

const xir::any TensorMirrorAttrs::get_attr(const std::string& key) const {
  return other_->get_attr(key);
}

TensorMirrorAttrs* TensorMirrorAttrs::set_attr(const std::string& key,
                                               const xir::any& value) {
  LOG(FATAL) << "not allowed";
  return nullptr;
}

TensorMirrorAttrs* TensorMirrorAttrs::rename(const std::string& name) {
  LOG(FATAL) << "not allowed";
  return nullptr;
}

const std::string TensorMirrorAttrs::to_string(
    const std::string& delimiter, const std::string& left_bracket,
    const std::string& right_bracket) const {
  std::ostringstream str;
  str << "TensorMirrorAttrs{@"
      << "@" << (void*)this
      << other_->to_string(delimiter, left_bracket, right_bracket) << "}";
  return str.str();
}

}  // namespace vart
