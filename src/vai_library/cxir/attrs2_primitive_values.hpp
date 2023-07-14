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
#include <any>
#include <cassert>
#include <cstring>

#include "xir/attrs/attrs_imp.hpp"
#include "xir/cxir.h"

namespace {
struct no_type_hint_t {};

template <typename from_t, typename to_t, typename type_hint = no_type_hint_t>
struct convert {
  static to_t conv(const from_t& value) { return value; }
};

template <typename>
struct type_hint_t {
  static constexpr enum xir_attr_value_tag_t tag = XIR_ATTR_TYPE_TAG_NONE;
  static constexpr enum xir_attr_value_tag_t vec_tag = XIR_ATTR_TYPE_TAG_NONE;
  static constexpr enum xir_attr_value_tag_t map_tag = XIR_ATTR_TYPE_TAG_NONE;
  static constexpr enum xir_attr_value_tag_t map_vec_tag =
      XIR_ATTR_TYPE_TAG_NONE;
  using cxx_type_t = void;
};

#define IMP_TYPE_HINT(name, c_type, cxx_type)                                  \
  template <>                                                                  \
  struct type_hint_t<cxx_type> {                                               \
    static constexpr enum xir_attr_value_tag_t tag = XIR_ATTR_TYPE_TAG_##name; \
    static constexpr enum xir_attr_value_tag_t vec_tag =                       \
        XIR_ATTR_TYPE_TAG_VEC_##name;                                          \
    static constexpr enum xir_attr_value_tag_t map_tag =                       \
        XIR_ATTR_TYPE_TAG_MAP_##name;                                          \
    static constexpr enum xir_attr_value_tag_t map_vec_tag =                   \
        XIR_ATTR_TYPE_TAG_MAP_VEC_##name;                                      \
    using cxx_type_t = cxx_type;                                               \
  };

XIR_ATTRS_SUPPORTED_PRIMITIVE_TYPES2(IMP_TYPE_HINT)

template <>
struct convert<xir_bool_t, bool> {
  static bool conv(xir_bool_t value) { return value.value != 0; }
};
template <>
struct convert<bool, xir_bool_t> {
  static xir_bool_t conv(bool value) { return xir_bool_t{value}; }
};

template <>
struct convert<xir_string_t, string> {
  static string conv(xir_string_t value) {
    return string(value.data, value.data + value.size);
  }
};
template <>
struct convert<string, xir_string_t> {
  static xir_string_t conv(const string& value) {
    return xir_string_t{value.c_str(), value.size()};
  }
};
template <>
struct convert<xir_bytes_t, vector<char>> {
  static vector<char> conv(const xir_bytes_t& value) {
    auto ret = vector<char>(value.size);
    memcpy(&ret[0], value.data, value.size);
    return ret;
  }
};
template <>
struct convert<vector<char>, xir_bytes_t> {
  static xir_bytes_t conv(const vector<char>& value) {
    return xir_bytes_t{(char*)(&value[0]), value.size()};
  }
};

#define IMP_FROM_CXX_TYPE_TO_ATTR_VALUE(name, c_type, cxx_type)                \
  template <>                                                                  \
  struct convert<cxx_type, xir_attr_value_t> {                                 \
    static xir_attr_value_t conv(const cxx_type& value) {                      \
      xir_attr_value_t ret;                                                    \
      ret.tag = XIR_ATTR_TYPE_TAG_##name;                                      \
      ret.u.name##_value = convert<cxx_type, c_type>::conv(value);             \
      return ret;                                                              \
    }                                                                          \
  };
XIR_ATTRS_SUPPORTED_PRIMITIVE_TYPES2(IMP_FROM_CXX_TYPE_TO_ATTR_VALUE)

#define IMP_FROM_ATTR_VALUE_TO_CXX_TYPE(name, c_type, cxx_type)                \
  template <>                                                                  \
  struct convert<xir_attr_value_t, cxx_type> {                                 \
    static cxx_type conv(xir_attr_value_t value) {                             \
      CHECK_EQ(value.tag, XIR_ATTR_TYPE_TAG_##name);                           \
      return convert<c_type, cxx_type>::conv(value.u.name##_value);            \
    }                                                                          \
  };
XIR_ATTRS_SUPPORTED_PRIMITIVE_TYPES2(IMP_FROM_ATTR_VALUE_TO_CXX_TYPE)

}  // namespace
