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
static constexpr xir_attr_value_t none() {
  struct xir_attr_value_t ret = xir_attr_value_t{tag : XIR_ATTR_TYPE_TAG_NONE};
  return ret;
}

static constexpr bool is_none(xir_attr_value_t v) {
  return v.tag == XIR_ATTR_TYPE_TAG_NONE;
}

template <typename cxx_type>
struct to_vec {
  static vector<cxx_type> conv(const xir_attr_value_iterator_t* iter) {
    auto ret = vector<cxx_type>();
    xir_size_hint_t size_hint = {0, 0, 0};
    if (iter->size_hint) {
      size_hint = iter->size_hint(iter->self);
    }
    if (size_hint.has_upper_bound) {
      ret.reserve(size_hint.upper_bound);
    }
    for (xir_attr_value_t x = iter->next(iter->self); !is_none(x);
         x = iter->next(iter->self)) {
      ret.push_back(convert<xir_attr_value_t, cxx_type>::conv(x));
    }
    if (iter->destroy) {
      iter->destroy(iter->self);
    }
    delete iter;
    return ret;
  }
};

template <typename cxx_type>
struct convert<vector<cxx_type>, xir_attr_value_t> {
  static xir_attr_value_t conv(const vector<cxx_type>& value);
};

template <typename cxx_type>
struct convert<xir_attr_value_t, vector<cxx_type>> {
  static vector<cxx_type> conv(xir_attr_value_t value);
};

template <typename iterator_t, typename cxx_type>
static xir_attr_value_t vec_next_adaptor(void* self) {
  auto cxx_iter = static_cast<pair<iterator_t, iterator_t>*>(self);
  auto b = cxx_iter->first;
  auto e = cxx_iter->second;
  auto ret = none();
  if (b != e) {
    // convert c++ type to c
    ret = convert<cxx_type, xir_attr_value_t>::conv(*b);
    cxx_iter->first++;
  }
  return ret;
}

template <typename cxx_type>
static void destroy_adaptor(void* self) {
  using cxx_iter_t = typename vector<cxx_type>::const_iterator;
  auto cxx_iter = static_cast<pair<cxx_iter_t, cxx_iter_t>*>(self);
  delete cxx_iter;
}

template <typename cxx_type>
static xir_size_hint_t size_hint_adaptor(void* self) {
  using cxx_iter_t = typename vector<cxx_type>::const_iterator;
  auto cxx_iter = static_cast<pair<cxx_iter_t, cxx_iter_t>*>(self);
  auto b = cxx_iter->first;
  auto e = cxx_iter->second;
  auto lower_bound = static_cast<size_t>(e - b);
  return xir_size_hint_t{lower_bound, true, lower_bound};
}

template <typename iterator_t, typename cxx_type>
struct make_vec_iterator {
  static xir_attr_value_iterator_t* conv(iterator_t begin, iterator_t end) {
    auto c_iter = new xir_attr_value_iterator_t;
    auto cxx_iter = new pair<iterator_t, iterator_t>(begin, end);
    c_iter->self = (void*)cxx_iter;
    c_iter->next = vec_next_adaptor<iterator_t, cxx_type>;
    c_iter->destroy = destroy_adaptor<cxx_type>;
    c_iter->size_hint = size_hint_adaptor<cxx_type>;
    return c_iter;
  }
};

template <typename cxx_type>
struct to_xir_iter_t {
  static xir_attr_value_iterator_t* conv(const vector<cxx_type>& value) {
    return make_vec_iterator<decltype(value.begin()), cxx_type>::conv(
        value.begin(), value.end());
  }
};

}  // namespace
