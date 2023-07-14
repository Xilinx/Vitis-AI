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
#include "xir/op/op_def.hpp"

#include <string>
#include <unordered_map>

namespace xir {

class OpDefFactoryImp : public OpDefFactory {
 public:
  void register_h(const OpDef& def) override;
  const OpDef* create(const std::string& type) const;
  const std::vector<std::string> get_registered_ops() const;

 private:
  std::unordered_map<std::string, OpDef> store_;
};

const OpDefFactoryImp* op_def_factory();

}  // namespace xir
