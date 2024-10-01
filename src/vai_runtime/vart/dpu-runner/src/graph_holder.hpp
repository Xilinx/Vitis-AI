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
#include "xir/graph/subgraph.hpp"

namespace vart {
namespace dpu {

class GraphHolder {
 public:
  explicit GraphHolder(const std::string& filename);
  ~GraphHolder() = default;

 public:
  const xir::Subgraph* get_subgraph(const std::string& kernel_name) const;

 private:
  void init_subgraph();

 public:
  std::unique_ptr<xir::Graph> graph;
  std::vector<xir::Subgraph*> subgraph;
};

}  // namespace dpu
}  // namespace vart
