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

#include "./attrs2_primitive_values.hpp"
#include "xir/attrs/attrs_imp.hpp"
#include "xir/cxir.h"

namespace {

template <typename iterator_t, typename cxx_type>
static xir_attr_pair_t map_next_adaptor(void* self) {
  auto cxx_iter = static_cast<pair<iterator_t, iterator_t>*>(self);
  auto b = cxx_iter->first;
  auto e = cxx_iter->second;
  auto ret = xir_attr_pair_t{none(), none()};
  if (b != e) {
    // convert c++ type to c
    ret.first.tag = xir_attr_value_tag_t::XIR_ATTR_TYPE_TAG_string;
    ret.first.u.string_value = convert<string, xir_string_t>::conv(b->first);
    ret.second = convert<cxx_type, xir_attr_value_t>::conv(b->second);
    cxx_iter->first++;
  }
  return ret;
}

template <typename iterator_t, typename cxx_type>
struct make_c_map_iterator {
  xir_attr_value_map_iterator_t* operator()(iterator_t begin, iterator_t end) {
    auto c_iter = new xir_attr_value_map_iterator_t;
    auto cxx_iter = new pair<iterator_t, iterator_t>(begin, end);
    c_iter->self = (void*)cxx_iter;
    c_iter->next = map_next_adaptor<iterator_t, cxx_type>;
    c_iter->destroy = destroy_adaptor<cxx_type>;
    return c_iter;
  }
};

template <typename cxx_type>
struct to_xir_map_iter_t {
  static xir_attr_value_map_iterator_t* conv(
      const map<string, cxx_type>& value) {
    return make_c_map_iterator<decltype(value.begin()), cxx_type>()(
        value.begin(), value.end());
  }
};

template <typename cxx_type>
struct to_map {
  static map<string, cxx_type> conv(const xir_attr_value_map_iterator_t* iter) {
    auto ret = map<string, cxx_type>();
    for (xir_attr_pair_t x = iter->next(iter->self); !is_none(x.first);
         x = iter->next(iter->self)) {
      auto key = x.first.u.string_value;
      CHECK_EQ(x.first.tag, xir_attr_value_tag_t::XIR_ATTR_TYPE_TAG_string);
      string key2(key.data, key.data + key.size);
      ret.insert({key2, convert<xir_attr_value_t, cxx_type>::conv(x.second)});
    }
    return ret;
  }
};

}  // namespace
