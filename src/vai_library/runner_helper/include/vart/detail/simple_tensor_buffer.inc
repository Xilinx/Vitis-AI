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
#pragma once
#include <tuple>
#include <vart/tensor_buffer.hpp>
#include <xir/tensor/tensor.hpp>
namespace vart {
template <typename T>
struct type_matcher_t {
  static inline bool match(const xir::DataType& dtype) { return false; }
};

template <>
struct type_matcher_t<float> {
  static inline bool match(const xir::DataType& dtype) {
    return dtype.type == xir::DataType::FLOAT && dtype.bit_width == 32;
  }
};

template <>
struct type_matcher_t<int8_t> {
  static inline bool match(const xir::DataType& dtype) {
    auto is_xint = dtype.type == xir::DataType::XINT && dtype.bit_width == 8;
    auto is_int = dtype.type == xir::DataType::INT && dtype.bit_width == 8;
    return is_xint || is_int;
  }
};

template <>
struct type_matcher_t<uint8_t> {
  static inline bool match(const xir::DataType& dtype) {
    auto is_xuint = dtype.type == xir::DataType::XUINT && dtype.bit_width == 8;
    auto is_uint = dtype.type == xir::DataType::UINT && dtype.bit_width == 8;
    return is_xuint || is_uint;
  }
};

template <>
struct type_matcher_t<void> {
  static inline bool match(const xir::DataType& dtype) { return true; }
};

template <typename T>
simple_tensor_buffer_t<T> simple_tensor_buffer_t<T>::create(
    vart::TensorBuffer* t) {
  auto shape = t->get_tensor()->get_shape();
  auto idx = std::vector<int>(shape.size(), 0);
  uint64_t data = 0u;
  size_t size = 0u;
  std::tie(data, size) = t->data(idx);
  if (t->get_tensor()->get_data_size() != size) {
    std::tie(data, size) = t->data(idx);
  }
  CHECK_EQ(t->get_tensor()->get_data_size(), size)
      << "only support tensor buffer with continuous memory region:"
      << t->to_string();
  CHECK(type_matcher_t<T>::match(t->get_tensor()->get_data_type()))
      << "type mismatch: T=" << typeid(T).name()
      << " dtype=" << t->get_tensor()->get_data_type().to_string()
      << " tensor_buffer=" << t->to_string();
  return simple_tensor_buffer_t<T>{(T*)data, size, t->get_tensor()};
}
}  // namespace vart

// Local Variables:
// mode:c++
// coding: utf-8-unix
// End:
