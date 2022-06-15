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

#include "xir/attrs/attrs_imp.hpp"

#include "UniLog/UniLog.hpp"
#include "xir/op/op_def_factory_imp.hpp"

namespace xir {

bool AttrsImp::has_attr(const std::string& key,
                        const std::type_info& type_id) const {
  auto ret = attrs_.count(key) > 0;
  if (type_id != typeid(void)) {
    // check the type of attr
    ret = ret && attrs_.at(key).type() == type_id;
  }
  return ret;
}

const xir::any& AttrsImp::get_attr(const std::string& key) const {
  UNI_LOG_CHECK(attrs_.count(key) > 0, XIR_UNREGISTERED_ATTR)
      << "Attrs doesn't contain attribute " << key;
  return attrs_.at(key);
}

xir::any& AttrsImp::get_attr(const std::string& key) {
  return const_cast<xir::any&>(
      static_cast<const AttrsImp&>(*this).get_attr(key));
}

std::vector<std::string> AttrsImp::get_keys() const {
  std::vector<std::string> ret;
  ret.reserve(this->attrs_.size());
  for (auto& map_pair : this->attrs_) {
    ret.push_back(map_pair.first);
  }
  return ret;
}

Attrs* AttrsImp::set_attr(const std::string& key, const xir::any& value) {
  attrs_[key] = value;
  return this;
}

bool AttrsImp::del_attr(const std::string& key) {
  return attrs_.erase(key) != 0;
}

std::string AttrsImp::debug_info() const {
  std::string ret;
  for (auto iter = attrs_.begin(); iter != attrs_.end(); ++iter) {
    ret.append(iter == attrs_.begin() ? "[\"" : "\"");
    ret.append(iter->first);
    ret.append("\": ");
    ret.append(iter->second.type().name());
    ret.append(std::next(iter) != attrs_.end() ? ", " : "]");
  }
  return ret;
}
}  // namespace xir
