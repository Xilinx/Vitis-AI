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
/*
 * Filename: simple_tensor_buffer.hpp
 *
 * Description: A utility class to simplify the usage of vart::TensorBuffer
 *
 */

#pragma once
#include <tuple>
#include <vart/tensor_buffer.hpp>
#include <xir/tensor/tensor.hpp>
namespace vart {

/**
 * @brief A utility class to simplify the usage of vart::TensorBuffer
 *
 * template parameter `T` can be one of `void`, `uint8_t`, `int8_t`,
 * `float`. Other types are not supported yet.
 *
 * A simple_tensor_buffer_t object does not own underlying memory.
 *
 * Sample code:
 * @code
   simple_tensor_buffer_t<int8_t> x =
   simple_tensor_buffer_t<int8_t>::create(a_vart_tensor_buffer);

   for(auto i = 0u; i < x.mem_size/sizeof(int8_t); ++i) {
     x.data[i] ++;
   }
   @endcode
 *
 */
template <typename T = void>
struct simple_tensor_buffer_t {
  static simple_tensor_buffer_t create(TensorBuffer* t);

 public:
  simple_tensor_buffer_t(const simple_tensor_buffer_t<void>& other)
      : data{(T*)other.data}, mem_size(other.mem_size), tensor(other.tensor) {}
  simple_tensor_buffer_t(T* d, size_t s, const xir::Tensor* t)
      : data{(T*)d}, mem_size(s), tensor(t) {}
  /** @brief data pointer to the underlying memory. It does not take the
   * ownership of the memory. */
  T* data;
  /** @brief the size of underlying memory in bytes */
  size_t mem_size;
  /** @brief the tensor object ownedy by the tensor buffer object. It
   * is necessary to use XIR API e.g. xir::Tensor::get_shape etc to
   * inspect the shape and data type of underlying memory */
  const xir::Tensor* tensor;
};
}  // namespace vart

#include "./detail/simple_tensor_buffer.inc"
