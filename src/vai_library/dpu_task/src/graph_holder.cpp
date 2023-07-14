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
#include "graph_holder.hpp"

#include <glog/logging.h>

#include <fstream>
#include <vitis/ai/env_config.hpp>
#include <xir/graph/graph.hpp>

namespace vitis {
namespace ai {

GraphHolder::GraphHolder(const std::string& filename) {
  graph_ = xir::Graph::deserialize(filename);
  init_subgraph();
}

void GraphHolder::init_subgraph() {
  auto root = graph_->get_root_subgraph();
  auto children = root->children_topological_sort();
  for (auto c : children) {
    CHECK(c->has_attr("device"));
    auto device = c->get_attr<std::string>("device");
    if (device == "DPU") {
      subgraphs_.emplace_back(c);
    }
  }
}

const xir::Graph* GraphHolder::get_graph() const {  //
  return graph_.get();
}

std::vector<const xir::Subgraph*> GraphHolder::get_subgraphs() const {
  return subgraphs_;
}

}  // namespace ai
}  // namespace vitis
