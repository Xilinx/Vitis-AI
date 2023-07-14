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
#include <string>
#include <typeindex>
#include <vector>
#include "xir/util/any.hpp"

namespace serial {
class AttrValue;
class ExpandAttrValue;
class OPNode;
class SubGraph;
class Graph;
class AttrDef;
class Tensor;
}  // namespace serial

namespace xir {
class Graph;
namespace v2 {

class Serialize {
 public:
 public:
  Serialize() = default;
  ~Serialize() = default;
  Serialize(const Serialize&) = delete;
  Serialize& operator=(const Serialize&) = delete;
  Serialize(Serialize&&) = delete;
  Serialize& operator=(Serialize&&) = delete;

  std::unique_ptr<Graph> read(const std::string& pb_fname);
  void write(const Graph* graph, const std::string& pb_fname);
  std::unique_ptr<Graph> read_from_string(const std::string& str);
  void write_to_string(const Graph* graph, std::string* str);

 private:
};
}  // namespace v2
}  // namespace xir
