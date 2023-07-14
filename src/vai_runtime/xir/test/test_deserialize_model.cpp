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

#include "UniLog/UniLog.hpp"
#include "xir/graph/graph.hpp"
std::vector<const xir::Subgraph*> get_dpu_subgraph(const xir::Graph* graph) {
  auto root = graph->get_root_subgraph();
  auto children = root->children_topological_sort();
  auto ret = std::vector<const xir::Subgraph*>();
  for (auto c : children) {
    CHECK(c->has_attr("device"));
    auto device = c->get_attr<std::string>("device");
    if (device == "DPU") {
      ret.emplace_back(c);
    }
  }
  return ret;
}

int main(int, char* argv[]) {
  UniLog::Initial(argv[0], UNI_LOG_STD, UNI_LOG_LEVEL_INFO,
                  UNI_LOG_STD_LEVEL_INFO);
  auto graph = xir::Graph::deserialize(argv[1]);
  auto subgraphs = get_dpu_subgraph(graph.get());
  UNI_LOG_INFO << "subgraphs size is " << subgraphs.size();
  for (auto subgraph : subgraphs) {
    UNI_LOG_INFO << "subgraph name " << subgraph->get_name();
    auto libs =
        subgraph->get_attr<std::map<std::string, std::string>>("runner");
    auto iter_lib = libs.find("run");
    UNI_LOG_INFO << "lib : " << iter_lib->second.c_str();
  }
  return 0;
}
