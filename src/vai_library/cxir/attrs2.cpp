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

#include <any>
#include <cassert>
#include <cstring>

#include "xir/attrs/attrs_imp.hpp"
#include "xir/cxir.h"

using namespace std;
// clang-format off
#include "./attrs2_primitive_values.hpp"
#include "./attrs2_any.hpp"
#include "./attrs2_vec.hpp"
#include "./attrs2_map.hpp"
// clang-format on
#include "./attrs2_any.tcc"

namespace xir {
class c_api {
 public:
  static xir_attr_value_t xir2_attrs_get(AttrsImp* self, xir_string_t key) {
    string key2(key.data, key.data + key.size);
    auto it = self->attrs_.find(key2);
    auto end = self->attrs_.end();
    if (it == end) {
      return none();
    }
    const auto& a_value = it->second;
    LOG_IF(INFO, false) << "get " << key2 << " from type "
                        << a_value.type().name() << " ";
    return convert<any, xir_attr_value_t>::conv(a_value);
  }

  static void xir2_attrs_set(AttrsImp* self, xir_string_t key,
                             xir_attr_value_t value) {
    string key2(key.data, key.data + key.size);
    auto& a_value = self->attrs_[key2];
    a_value = convert<xir_attr_value_t, any>::conv(value);
    LOG(INFO) << "set " << key2 << " to type " << a_value.type().name() << " "
              << " this=" << (void*)self << " size=" << self->attrs_.size();
  }

  static xir_attr_value_t xir2_attrs_keys(AttrsImp* self) {
    xir_attr_value_t ret;
    ret.tag = XIR_ATTR_TYPE_TAG_MAP;
    ret.u.map_value =
        make_c_map_iterator<decltype(self->attrs_.begin()), any>()(
            self->attrs_.begin(), self->attrs_.end());
    LOG(INFO) << self->attrs_.size() << " this=" << self;
    return ret;
  }
};
}  // namespace xir
extern "C" xir_attr_value_t xir2_attrs_get(xir_attrs_t attrs,
                                           xir_string_t key) {
  return xir::c_api::xir2_attrs_get(static_cast<xir::AttrsImp*>(attrs), key);
};

extern "C" void xir2_attrs_set(xir_attrs_t attrs, xir_string_t key,
                               xir_attr_value_t value) {
  return xir::c_api::xir2_attrs_set(static_cast<xir::AttrsImp*>(attrs), key,
                                    value);
};

extern "C" xir_attr_value_t xir2_attrs_keys(xir_attrs_t attrs) {
  return xir::c_api::xir2_attrs_keys(static_cast<xir::AttrsImp*>(attrs));
}

extern "C" xir_attrs_t xir_attrs_create() {
  return static_cast<xir_attrs_t>(xir::Attrs::create().release());
}

extern "C" void xir_attrs_destroy(xir_attrs_t attrs) {
  delete static_cast<xir::Attrs*>(attrs);
}
