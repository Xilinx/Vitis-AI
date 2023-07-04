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

#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>
#include <functional>
#include "xir/op/op.hpp"

namespace xir {

class OpTemplate {
 public:
  virtual const std::string get_name() const = 0;

  virtual const std::set<std::string> get_types() const = 0;

  virtual int get_input_num() const = 0;

  virtual const std::set<OpTemplate*> get_input_ops() const = 0;

  virtual int get_fanout_num() const = 0;

  virtual const std::set<OpTemplate*> get_fanout_ops() const = 0;

  virtual void set_filter(const std::function<bool(Op*)>&) = 0;

  virtual const std::function<bool(Op*)>& get_filter() const = 0;

 public:
  virtual ~OpTemplate() = default;
};

class GraphTemplate {
 public:
  static std::unique_ptr<GraphTemplate> create(std::string name);

  virtual const std::string get_name() const = 0;

  virtual OpTemplate* add_op(const std::string name,
                             const std::set<std::string> types) = 0;

  virtual OpTemplate* add_op(
      const std::string name, const std::set<std::string> types,
      const std::map<OpTemplate*, std::string> input_ops) = 0;

  virtual OpTemplate* get_op(const std::string op_name) = 0;

  virtual void set_filter(const std::function<bool(std::map<OpTemplate*, Op*>)>&) = 0;

  virtual const std::function<bool(std::map<OpTemplate*, Op*>)>& get_filter() const = 0;

  virtual int get_op_num() const = 0;

  virtual const std::vector<OpTemplate*> topological_sort() const = 0;

  virtual void save_to_dot(const std::string& file_name) = 0;

  virtual void visualize(const std::string& file_name,
                         const std::string& format) = 0;

 public:
  virtual ~GraphTemplate() = default;
};

}  // namespace xir
