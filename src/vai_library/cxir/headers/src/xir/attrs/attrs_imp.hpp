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

#include <unordered_map>

#include "xir/attrs/attrs.hpp"
#include "xir/op/op_def.hpp"

namespace xir {

class AttrsImp : public Attrs {
 public:
  using AttrMap = std::unordered_map<std::string, xir::any>;

 public:
  AttrsImp() = default;
  AttrsImp(const AttrsImp& other) = default;
  virtual ~AttrsImp() = default;

 public:
  virtual bool has_attr(const std::string& key,
                        const std::type_info& type_id) const override;

  virtual const xir::any& get_attr(const std::string& key) const override;
  virtual xir::any& get_attr(const std::string& key) override;

  virtual std::vector<std::string> get_keys() const override;

  virtual Attrs* set_attr(const std::string& key,
                          const xir::any& value) override;
  virtual bool del_attr(const std::string& key) override;
  virtual std::string debug_info() const override;

  // const std::vector<std::string> get_pbattr_keys() const override;

 private:
  AttrMap attrs_;
  friend class c_api;
};

}  // namespace xir
