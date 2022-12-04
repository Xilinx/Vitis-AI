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

#pragma once
#include <string>
#include <unordered_map>

#include "xir/op/op_def.hpp"

namespace xir {

class XIR_DLLESPEC OpDefFactoryImp : public OpDefFactory {
 public:
  void register_h(const OpDef& def) override;
  const OpDef* create(const std::string& type) const;
  const std::vector<std::string> get_registered_ops() const;
  const OpDef* get_op_def(const std::string& type,
                          bool register_custome_op_if_not_exists = true);
  const OpDef& get_const_op_def(const std::string& type) const;

 private:
  void register_customized_operator_definition(const std::string& type);

 private:
  std::unordered_map<std::string, OpDef> store_;
};

XIR_DLLESPEC OpDefFactoryImp* op_def_factory();

}  // namespace xir
