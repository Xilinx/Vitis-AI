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
#include <string>

#include "UniLog/UniLog.hpp"
#include "xir/util/any.hpp"

namespace xir {

/**
 * @brief Attrs is a interface class for users to access a set of attributes.
 *
 */
class Attrs {
 public:
  Attrs() = default;
  Attrs(const Attrs& other) = default;
  virtual ~Attrs() = default;

 public:
  /**
   * @brief Create a new empty Attrs object, and return a unique pointer of the
   * new object.
   *
   * @return A unique pointer to the created Attrs object.
   */
  static std::unique_ptr<Attrs> create();

  /**
   * @brief Clone a Attrs object from a existing Attrs object, and return a
   * unique pointer of the new Attrs object.
   *
   * @param attr A row point of an existing Attrs attrs.
   *
   * @return A unique pointer to the new Attrs object.
   */
  static std::unique_ptr<Attrs> clone(Attrs* attr);

 public:
  /**
   * @brief Get the attribute value indexed by key.
   *
   * @details If the attribute requested is not existing, there will be a fatal.
   *
   * @param key A string to indicate the key of the attribute to get.
   *
   * @return The corresponding type of value
   */
  template <typename Dtype>
  const Dtype get_attr(const std::string& key) const {
    return xir::stdx::any_cast<Dtype>(this->get_attr(key));
  }

  /**
   * @brief Set the <key, value> pair to this Attrs object.
   *
   * @param key A string to index the attribute.
   *
   * @param value The attribute content.
   *
   * @return A raw pointer of this Attrs object.
   */
  template <typename Dtype>
  Attrs* set_attr(const std::string& key, const Dtype& value) {
    this->set_attr(key, xir::any{value});
    return this;
  }

  /**
   * @brief Check the existence of the attribute indicated by key.
   *
   * @param key A string to indicate the key of the attribute to check.
   *
   * @param type_id a type for checking, if it is not typeid(void)
   * then it is not only to check existing or not but also check the
   * type is matched or not
   *
   * @return true for existing, false for not or type is not matched.
   */
  virtual bool has_attr(const std::string& key,
                        const std::type_info& type_id = typeid(void)) const = 0;

  /**
   * @brief Get the attribute value indexed by key.
   *
   * @details If the attribute requested is not existing, there will be a fatal.
   *
   * @param key A string to indicate the key of the attribute to get.
   *
   * @return The corresponding type of value
   */
  virtual const xir::any& get_attr(const std::string& key) const = 0;
  virtual xir::any& get_attr(const std::string& key) = 0;

  /**
   * @brief Get all the keys in this Attrs object.
   *
   * @return A vector of the keys.
   */
  virtual std::vector<std::string> get_keys() const = 0;

  /**
   * @brief Set the <key, value> pair to this Attrs object.
   *
   * @param key A string to index the attribute.
   *
   * @param value The attribute content.
   *
   * @return A raw pointer of this Attrs object.
   */
  virtual Attrs* set_attr(const std::string& key, const xir::any& value) = 0;

  /**
   * @brief Remove an attribute if existing.
   *
   * @param key A string to indicate the key of the attribute to be removed.
   *
   * @return true for existing, false for not.
   */
  virtual bool del_attr(const std::string& key) = 0;

  /**
   * @brief Get a string of all the information inside the Attrs object.
   *
   * @return A string object.
   */
  virtual std::string debug_info() const = 0;

  /**
   * @brief Get all the keys should be serialized.
   *
   * @return A vector of string.
   */
  //  virtual const std::vector<std::string> get_pbattr_keys() const = 0;

  /**
   * @brief helper function to compare std::any
   *
   * @param a
   *
   * @param b
   *
   * @return 1 if a == b, 0 if a != b, and -1 if uncertain.
   *
   * @note if a and b are not same type, return 0. But they has a same
   * type, we only support a limit nubmer of types, e.g. int,
   * vector<int>, map<string, int> etc, if the type is not supported,
   * return -1, i.e. uncertain.
   */

  static int cmp(const any& a, const any& b);
};

}  // namespace xir
