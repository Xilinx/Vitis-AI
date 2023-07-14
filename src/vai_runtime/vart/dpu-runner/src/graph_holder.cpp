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
#include <UniLog/UniLog.hpp>

DEF_ENV_PARAM(DEBUG_DPU_RUNNER, "0")
DEF_ENV_PARAM(XLNX_CHECK_COMMIT_ID_ENABLE, "0");
/*DEF_ENV_PARAM_2(XCOM_XIR_VERSION_MD5,
                "1aa6b12a2ee2734382f3c2ba0c5735c6a3e701fe", std::string);
DEF_ENV_PARAM_2(XCOMPILER_VERSION_MD5,
                "24c0e84ba8f17b75bffc824f95ce1e1619930a71", std::string);
*/
namespace vart {
namespace dpu {
void check_commit_id(const xir::Graph* graph) {
  std::map<std::string, std::string> xcompiler_version_map = {
      {"XCOM : xir : 0.0.1", "fde84c06b50c4542eac4264372cfaed69c7fb95d"},
      {"xcompiler:0.0.1", "7e7b99e98a8de082b16ee9cf240b8bf9f4571cf5"}};
  UNI_LOG_CHECK(graph->has_attr("tools_commit_id"), VART_GRAPH_ERROR);
  auto tools_commit_id =
      graph->get_attr<std::map<std::string, std::string>>("tools_commit_id");
  for (auto& it : xcompiler_version_map) {
    auto kv = tools_commit_id.find(it.first);
    UNI_LOG_CHECK(kv != tools_commit_id.end(), VART_GRAPH_ERROR)
      << it.first << " not find !";
    UNI_LOG_CHECK(it.second == kv->second, VART_VERSION_MISMATCH)
        << it.first << " no match! the right version is " << it.second
        << ", now use version is " << kv->second;
  }
}

GraphHolder::GraphHolder(const std::string& filename) {
  graph = xir::Graph::deserialize(filename);
  init_subgraph();
  // check commit
  if (ENV_PARAM(XLNX_CHECK_COMMIT_ID_ENABLE)) {
    check_commit_id(graph.get());
  }
}

void GraphHolder::init_subgraph() {
  auto root = graph->get_root_subgraph();
  auto children = root->children_topological_sort();
  for (auto c : children) {
    UNI_LOG_CHECK(c->has_attr("device"), VART_GRAPH_ERROR);
    auto device = c->get_attr<std::string>("device");
    if (device == "DPU") {
      subgraph.emplace_back(c);
    }
  }
}
const xir::Subgraph* GraphHolder::get_subgraph(
    const std::string& kernel_name) const {
  auto index = 0u;
  for (index = 0u; index < subgraph.size(); ++index) {
    if (subgraph[index]->get_name() == kernel_name) {
      LOG_IF(INFO, ENV_PARAM(DEBUG_DPU_RUNNER))
          << "found subgraph. kernel name match. kernel_name=" << kernel_name;
      return subgraph[index];
    }
  }
  if (kernel_name.size() > 2 && kernel_name[kernel_name.size() - 2] == '_' &&
      isdigit(kernel_name[kernel_name.size() - 1])) {
    index = ((size_t)kernel_name[kernel_name.size() - 1]) - '0';
  }
  UNI_LOG_CHECK(index < subgraph.size(), VART_OUT_OF_RANGE)
    << "kernel_name " << kernel_name << " "  //
      ;
  return subgraph[index];
}
}  // namespace dpu
}  // namespace vart
