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

#include <functional>
#include <map>
#include <string>
#include <type_traits>
#include <typeindex>
#include <typeinfo>
#include <vector>
#include "UniLog/UniLog.hpp"
#include "xir/op/op.hpp"

#include "xir/util/any.hpp"

namespace xir {

extern std::type_index TYPE_INDEX_BOOL;
extern std::type_index TYPE_INDEX_INT8;
extern std::type_index TYPE_INDEX_UINT8;
extern std::type_index TYPE_INDEX_INT16;
extern std::type_index TYPE_INDEX_UINT16;
extern std::type_index TYPE_INDEX_INT32;
extern std::type_index TYPE_INDEX_UINT32;
extern std::type_index TYPE_INDEX_INT64;
extern std::type_index TYPE_INDEX_UINT64;
extern std::type_index TYPE_INDEX_FLOAT;
extern std::type_index TYPE_INDEX_DOUBLE;
extern std::type_index TYPE_INDEX_STRING;
extern std::type_index TYPE_INDEX_BYTES;

extern std::type_index TYPE_INDEX_BOOL_VEC;
extern std::type_index TYPE_INDEX_INT8_VEC;
extern std::type_index TYPE_INDEX_UINT8_VEC;
extern std::type_index TYPE_INDEX_INT16_VEC;
extern std::type_index TYPE_INDEX_UINT16_VEC;
extern std::type_index TYPE_INDEX_INT32_VEC;
extern std::type_index TYPE_INDEX_UINT32_VEC;
extern std::type_index TYPE_INDEX_INT64_VEC;
extern std::type_index TYPE_INDEX_UINT64_VEC;
extern std::type_index TYPE_INDEX_FLOAT_VEC;
extern std::type_index TYPE_INDEX_DOUBLE_VEC;
extern std::type_index TYPE_INDEX_STRING_VEC;
extern std::type_index TYPE_INDEX_BYTES_VEC;

extern std::type_index TYPE_INDEX_MAP_STR_2_INT32;
extern std::type_index TYPE_INDEX_MAP_STR_2_VEC_CHAR;
extern std::type_index TYPE_INDEX_MAP_STR_2_STR;

/*
 *@struct AttrDef
 *@brief attribute definition
 *This struct defines an attribute, like 'kernel_w' for conv2d
 */
struct AttrDef {
  /**
   * @brief Element Occurence Specifier
   */
  enum OccurenceType {
    /// Once and only once
    REQUIRED,
    /// Never or once
    OPTIONAL,
    NUM
  };

  /// Name of the op attribute
  const std::string name;
  /// Data type
  const std::type_index data_type;
  /// Occurence type
  const OccurenceType occur_type;
  /// List size for validation, 0 for variable length
  const std::uint32_t list_length;
  /// Some comments
  const std::string annotation;
  /// Default value of the attribute
  const xir::any default_value;
};

template <typename T>
struct is_std_vector : std::false_type {};
template <typename T>
struct is_std_vector<std::vector<T>> : std::true_type {};
template <typename T, typename = void>
struct is_std_map : std::false_type {};
template <typename T>
struct is_std_map<
    T, std::enable_if_t<std::is_same<
           typename T::value_type, std::pair<const typename T::key_type,
                                             typename T::mapped_type>>::value>>
    : std::true_type {};

template <typename T, typename... P0toN>
struct is_one_of : public std::false_type {};
template <typename T, typename... P1toN>
struct is_one_of<T, T, P1toN...> : public std::true_type {};
template <typename T, typename U, typename... P1toN>
struct is_one_of<T, U, P1toN...> : public is_one_of<T, P1toN...> {};

template <typename T, typename Enable = void>
struct AttrDefBuilder {};

namespace {
using namespace xir;
template <typename T>
const AttrDef build_required_attr(const std::string& name,
                                  const AttrDef::OccurenceType& occur_type,
                                  const std::uint32_t& length,
                                  const std::string& annotation) {
  UNI_LOG_CHECK(occur_type == AttrDef::OccurenceType::REQUIRED,
                XIR_UNEXPECTED_VALUE)
    << "REQUIRED item does not need to have a default value";
  return AttrDef{name,
                 std::type_index{typeid(T)},
                 AttrDef::OccurenceType::REQUIRED,
                 length,
                 annotation,
                 T{}};
}

template <typename T>
const AttrDef build_optional_attr(const std::string& name,
                                  const AttrDef::OccurenceType& occur_type,
                                  const std::uint32_t& length,
                                  const std::string& annotation,
                                  const T& default_value) {
  UNI_LOG_CHECK(occur_type == AttrDef::OccurenceType::OPTIONAL,
                XIR_UNEXPECTED_VALUE)
    << "OPTIONAL item needs to have a default value";
  return AttrDef{name,
                 std::type_index{typeid(T)},
                 AttrDef::OccurenceType::OPTIONAL,
                 length,
                 annotation,
                 default_value};
}
}  // namespace

// for scale type
template <typename T>
struct AttrDefBuilder<
  T, typename std::enable_if<is_one_of<
       T, bool, std::int8_t, std::uint8_t, std::int16_t, std::uint16_t,
       std::int32_t, std::uint32_t, std::int64_t, std::uint64_t, float,
       double, std::string, char>::value>::type> {
  static const AttrDef build(std::string name,
                             AttrDef::OccurenceType occur_type,
                             std::string annotation) {
    return build_required_attr<T>(name,
                                  occur_type,
                                  1,
                                  annotation);
  }
  static const AttrDef build(std::string name,
                             AttrDef::OccurenceType occur_type,
                             std::string annotation,
                             const T& default_value) {
    return build_optional_attr<T>(name,
                                  occur_type,
                                  1,
                                  annotation,
                                  default_value);
  }
};

// for map type
template <typename T>
struct AttrDefBuilder<
  T, typename std::enable_if<
       is_std_map<T>::value &&
       is_one_of<typename T::key_type, bool, std::int8_t, std::uint8_t,
                 std::int16_t, std::uint16_t, std::int32_t, std::uint32_t,
                 std::int64_t, std::uint64_t, float, double, std::string,
                 char>::value &&
       is_one_of<typename T::mapped_type, bool, std::int8_t, std::uint8_t,
                 std::int16_t, std::uint16_t, std::int32_t, std::uint32_t,
                 std::int64_t, std::uint64_t, float, double, std::string,
                 char, std::vector<char>>::value>::type> {
  static const AttrDef build(std::string name,
                             AttrDef::OccurenceType occur_type,
                             std::string annotation) {
    return build_required_attr<T>(name,
                                  occur_type,
                                  1,
                                  annotation);
  }
  static const AttrDef build(std::string name,
                             AttrDef::OccurenceType occur_type,
                             std::string annotation,
                             const T& default_value) {
    return build_optional_attr<T>(name,
                                  occur_type,
                                  1,
                                  annotation,
                                  default_value);
  }
};

// for vector type
template <typename T>
struct AttrDefBuilder<
  T, typename std::enable_if<
       is_std_vector<T>::value &&
       is_one_of<typename T::value_type, bool, std::int8_t, std::uint8_t,
                 std::int16_t, std::uint16_t, std::int32_t, std::uint32_t,
                 std::int64_t, std::uint64_t, float, double, std::string,
                 char>::value>::type> {
  static const AttrDef build(std::string name,
                             AttrDef::OccurenceType occur_type,
                             std::uint32_t length,
                             std::string annotation) {
    return build_required_attr<T>(name,
                                  occur_type,
                                  length,
                                  annotation);
  }
  static const AttrDef build(std::string name,
                             AttrDef::OccurenceType occur_type,
                             std::uint32_t length,
                             std::string annotation,
                             const T& default_value) {
    return build_optional_attr<T>(name,
                                  occur_type,
                                  length,
                                  annotation,
                                  default_value);
  }
};

}  // namespace xir
