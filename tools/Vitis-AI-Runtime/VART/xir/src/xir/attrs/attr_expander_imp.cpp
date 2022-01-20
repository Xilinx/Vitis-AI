/*
 * Copyright 2019 Xilinx Inc.
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

#include "xir/attrs/attr_expander_imp.hpp"
#include "UniLog/UniLog.hpp"

#include <algorithm>
#include <iostream>
#include <iterator>
#include <sstream>
#include "xir/util/dynamic_load.hpp"

namespace xir {

static std::vector<std::string> str_split(const std::string& str) {
  auto ret = std::vector<std::string>{};
  std::istringstream iss(str);
  std::copy(std::istream_iterator<std::string>(iss),
            std::istream_iterator<std::string>(), back_inserter(ret));
  return ret;
}

static void load_expanded_attrs_library(AttrExpanderImp* self,
                                        const std::string& file_name) {
  typedef void (*INIT_FUN)(AttrExpander*);
  INIT_FUN expand_func = NULL;
  auto handle = dlopen(file_name.c_str(), RTLD_LAZY);
  UNI_LOG_CHECK(handle, XIR_OPERATION_FAILED)
      << "Cannot open library " << file_name;

  expand_func = (INIT_FUN)dlsym(handle, "expand_expanded_attrs");

  UNI_LOG_CHECK(dlerror() == NULL, XIR_OPERATION_FAILED)
      << "Cannot load symbol 'expand_expanded_attrs' from " << file_name;

  UNI_LOG_INFO << "Load expanded attributes from " << file_name;
  expand_func(self);
}

const AttrExpanderImp* attr_expander() {
  static std::unique_ptr<AttrExpanderImp> self;
  if (self == nullptr) {
    self = std::make_unique<AttrExpanderImp>();
    auto expanded_attrs_library_list = getenv("EXPANDED_ATTRS_LIBRARY");
    if (expanded_attrs_library_list != nullptr) {
      for (const auto& expanded_attrs_library :
           str_split(std::string{expanded_attrs_library_list})) {
        load_expanded_attrs_library(self.get(), expanded_attrs_library);
      }
    }
  }
  return self.get();
}

AttrExpanderImp::AttrExpanderImp() : store_{} {
  for (auto target = 0; target < static_cast<int>(Target::NUM); ++target) {
    store_.emplace(static_cast<Target>(target), std::vector<AttrDef>{});
  }
}

void AttrExpanderImp::expand(Target target, const AttrDef& def) {
  auto iter = ::std::find_if(store_.at(target).begin(), store_.at(target).end(),
                             [&def](const AttrDef& this_def) -> bool {
                               return def.name == this_def.name;
                             });
  UNI_LOG_CHECK(iter == store_.at(target).end(),
                XIR_MULTI_REGISTERED_EXPANDED_ATTR)
      << def.name;
  store_.at(target).push_back(def);
}

const std::vector<AttrDef> AttrExpanderImp::get_expanded_attrs(
    Target target) const {
  return store_.at(target);
}

}  // namespace xir
