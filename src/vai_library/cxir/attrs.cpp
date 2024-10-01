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
#include <cstring>

#include "./cxir.h"
#include "xir/attrs/attrs_imp.hpp"

namespace xir {
class c_api {
 public:
  static size_t xir_attrs_get_num_of_keys(AttrsImp* self) {
    return self->attrs_.size();
  }
  static const char* xir_attrs_get_key(AttrsImp* self, size_t idx) {
    auto num_of_buckets = self->attrs_.bucket_count();
    auto tmp_idx = 0u;
    auto bucket_index = 0u;
    for (bucket_index = 0u;
         bucket_index < num_of_buckets &&
         idx >= tmp_idx + self->attrs_.bucket_size(bucket_index);
         ++bucket_index) {
      tmp_idx = tmp_idx + self->attrs_.bucket_size(bucket_index);
    }
    CHECK_LE(tmp_idx, idx);
    auto it = self->attrs_.begin(bucket_index);
    auto end = self->attrs_.end(bucket_index);
    for (; tmp_idx < idx && it != end; tmp_idx++) {
      ++it;
    };
    return it->first.c_str();
  }
};
}  // namespace xir

extern "C" xir_attrs_t xir_attrs_create() {
  return static_cast<xir_attrs_t>(xir::Attrs::create().release());
}

extern "C" void xir_attrs_destroy(xir_attrs_t attrs) {
  delete static_cast<xir::Attrs*>(attrs);
}
