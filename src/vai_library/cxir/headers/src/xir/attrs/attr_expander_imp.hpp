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
#include "xir/attrs/attr_expander.hpp"

#include <map>
#include <string>
#include <vector>

namespace xir {

class AttrExpanderImp : public AttrExpander {
 public:
  AttrExpanderImp();
  void expand(Target target, const AttrDef& def) override;
  const std::vector<AttrDef> get_expanded_attrs(Target target) const;

 private:
  std::map<Target, std::vector<AttrDef>> store_;
};

const AttrExpanderImp* attr_expander();

}  // namespace xir
