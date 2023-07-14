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

#include <algorithm>

#include "xir/attrs/attr_def.hpp"

#define XIR_STATIC_ATTR(TARGET, ATTR)                                          \
  std::pair<xir::AttrExpander::Target, xir::AttrDef> {                         \
    xir::AttrExpander::Target::TARGET, ATTR                                    \
  }

#define XIR_EXPAND_STATIC_ATTRS(...)                                           \
  extern "C" void expand_expanded_attrs(xir::AttrExpander* self) {             \
    auto attrs =                                                               \
        std::vector<std::pair<xir::AttrExpander::Target, xir::AttrDef>>{       \
            __VA_ARGS__};                                                      \
    std::for_each(                                                             \
        attrs.begin(), attrs.end(),                                            \
        [self](const std::pair<xir::AttrExpander::Target, xir::AttrDef>& p) {  \
          self->expand(p.first, p.second);                                     \
        });                                                                    \
  }

namespace xir {

class AttrExpander {
 public:
  enum class Target : int { Op = 0, Tensor, Subgraph, NUM };
  virtual void expand(Target target, const AttrDef& def) = 0;

 public:
  virtual ~AttrExpander() = default;
};

}  // namespace xir
