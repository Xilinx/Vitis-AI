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

#include <memory>
#include <vector>

#include "xir/attrs/attrs.hpp"
#include "xir/util/data_type.hpp"

namespace xir {

class Op;

/**
 * @brief  XIR Tensor interface
 *
 * This class defines the basic XIR Tensor Interface.
 */
class Tensor {
 public:
  /**
   * @brief Create a Tensor instance.
   *
   * @param name The name of the Tensor.
   *
   * @param shape A vector to indicate the tensors shape.
   *
   * @param data_type Indicates the type of the Tensor data.
   *
   * @return A unique pointer to the new Tensor object.
   */
  static std::unique_ptr<Tensor> create(const std::string& name,
                                        const std::vector<std::int32_t>& shape,
                                        const DataType& data_type);

  /**
   * @brief Create a Tensor instance. This API will be removed in the future
   * release version!
   *
   * @param name The name of the Tensor.
   *
   * @param shape A vector to indicate the tensors dimensions.
   *
   * @param data_type Indicates the type of the Tensor data.
   *
   * @param bit_width Indicates the bit width of the Tensor data.
   *
   * @return A unique pointer to the new Tensor object.
   */
  [[deprecated(
      "This API will be removed in the future release version. Please use "
      "static std::unique_ptr<Tensor> Tensor::create(const std::string& name, "
      "const std::vector<std::int32_t>& shape, const DataType& data_type) "
      "instead.")]] static std::unique_ptr<Tensor>
  create(const std::string& name, const std::vector<std::int32_t>& shape,
         const DataType::Type& data_type, const std::int32_t bit_width);

  /**
   * @brief Create a Tensor instance of the Tensor from an existing one.
   *
   * @param tensor A raw pinter to the existing instance.
   *
   * @return A unique pointer to the new Tensor object.
   */
  static std::unique_ptr<Tensor> clone(const Tensor* tensor);

  /**
   * @brief Create a Tensor instance from an existing one and rename it.
   *
   * @param tensor A raw pointer to the existing Tensor.
   *
   * @param name The name of the new tensor.
   *
   * @return A unique pointer to the new Tensor object.
   */
  static std::unique_ptr<Tensor> clone(const Tensor* tensor,
                                       const std::string& name);

 public:
  /**
   * @brief Get the name of the current Tensor object.
   *
   * @return The name of current Tensor object.
   */
  virtual const std::string get_name() const = 0;

  /**
   * brief Get the producer op of this tensor.
   *
   * @return The raw Op pointer or a nullptr while this tensor is not attached
   * to any op.
   */
  virtual const Op* get_producer() const = 0;

  /**
   * brief Get the producer op of this tensor.
   *
   * @return The raw Op pointer or a nullptr while this tensor is not attached
   * to any op.
   */
  virtual Op* get_producer() = 0;

  /**
   * @brief Get the tensor shape.

   * @return A vector of the tensor shape.
   */
  virtual const std::vector<std::int32_t> get_shape() const = 0;

  /**
   * @brief Get the tensor shape. This API will be removed in the future release
   * version.
   *
   * @return A vector of the tensor shape.
   */
  [[deprecated(
      "Tensor::get_dims() will be removed in the future version. Please use "
      "the Tensor::get_shape() instead.")]] virtual const std::
      vector<std::int32_t>
      get_dims() const = 0;

  /**
   * @brief Get the tensor shape size. This API will be removed in the future
   * release version.
   *
   * @return A vector of the tensor shape size.
   */
  [
      [deprecated("Tensor::get_dim_num() will be removed in the future "
                  "version. Please use the Tensor::get_shape().get_size() "
                  "instead.")]] virtual std::int32_t
  get_dim_num() const = 0;

  /**
   * @brief Get the dimension size of one specific dimension indicated by idx.
   * This API will be removed in the future release version.
   *
   * @param idx Indicate the dimension requested.
   *
   * @return The dimension size.
   */
  [[deprecated(
      "Tensor::get_dim_size(std::int32_t idx) will be removed in the future "
      "version. Please use the Tensor::get_shape().at(idx) "
      "instead.")]] virtual std::int32_t
  get_dim_size(std::int32_t idx) const = 0;

  /**
   * @brief Get the number of data in the current Tensor object.
   *
   * @return The number of data.
   */
  virtual std::int64_t get_element_num() const = 0;

  /**
   * @brief Get the data type of the current Tensor object.
   *
   * @return Data type.
   */
  virtual const DataType& get_data_type() const = 0;

  /**
   * @brief Get the bit_width of the current Tensor object. It will be removed
   * in the future release version!
   *
   * @return bit_width.
   */
  [[deprecated(
      "Tensor::get_bit_width() API will be removed in the future version, "
      "please use the Tensor::get_data_type() API to get the data type and "
      "read the bit width information in it.")]] virtual std::int32_t
  get_bit_width() const = 0;

  /**
   * @brief Get the number of elements in the current Tensor object.
   *
   * @return Number of elements.
   */
  virtual std::uint64_t get_data_size() const = 0;

  /**
   * @brief Get the Attrs object of the current Tensor object.
   *
   * @return A unique pointer to the Attrs object.
   */
  virtual std::unique_ptr<Attrs> get_attrs() const = 0;

  /**
   * @brief Set a new Attrs object to the current Tensor object.
   *
   * @param attrs A unique pointer to the new Attrs object.
   */
  virtual Tensor* set_attrs(std::unique_ptr<Attrs> attrs) = 0;

  /**
   * @brief Check the existence of the attribute indicated by key.
   *
   * @param key The attribute name.
   *
   * @return True if exist, else false.
   */
  virtual bool has_attr(const std::string& key) const = 0;

  /**
   * @brief Get the attribute value indicated by key.
   *
   * @param key The attribute name.
   *
   * @return The attribute value.
   */
  virtual const xir::any get_attr(const std::string& key) const = 0;

  /**
   * @brief Set the attribute value indicated by <key, value> pair.
   *
   * @param key The attribute name.
   *
   * @param value The attribute value.
   *
   * @return A raw pointer to the current Tensor object.
   */
  virtual Tensor* set_attr(const std::string& key, const xir::any& value) = 0;

  /**
   * @brief Get the attribute value indicated by key.
   *
   * @param key The attribute name.
   *
   * @return The attribute value.
   */
  template <typename Dtype>
  const Dtype get_attr(const std::string& key) const {
    return xir::stdx::any_cast<Dtype>(this->get_attr(key));
  }

  /**
   * @brief Set the attribute value indicated by <key, value> pair.
   *
   * @param key The attribute name.
   *
   * @param value The attribute value.
   *
   * @return A raw pointer to the current Tensor object.
   */
  template <typename Dtype>
  Tensor* set_attr(const std::string& key, const Dtype& value) {
    this->set_attr(key, xir::any{value});
    return this;
  }

  /**
   * @brief Rename the tensor.
   *
   * @param name The new name for the tensor.
   *
   * @return A raw pointer to the current Tensor object.
   *
   * @details it will check the identity of the name, and it's not recommended
   * to use.
   */
  virtual Tensor* rename(const std::string& name) = 0;

  /**
   * @brief Return the brief info of tensor in std::string format.
   *
   * @return Breif info of tensor in std::string format.
   */
  virtual const std::string to_string(
      const std::string& delimiter = ",",     //
      const std::string& left_bracket = "{",  //
      const std::string& right_bracket = "}") const = 0;

 public:
  virtual ~Tensor() = default;
};

}  // namespace xir
