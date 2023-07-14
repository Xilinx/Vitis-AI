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
#include "vitis/ai/graph_runner.hpp"

#include "graph_runner.hpp"
#include "vitis/ai/env_config.hpp"

#ifndef GRAPH_RUNNER
#define GRAPH_RUNNER "libvitis_ai_library-graph_runner.so.3"
#endif

#ifndef CPU_TASK
#define CPU_TASK "libvitis_ai_library-cpu_task.so.3"
#endif

DEF_ENV_PARAM(DEBUG_GRAPH_RUNNER_USE_CPU_TASK, "1")

namespace vitis {
namespace ai {

std::unique_ptr<vart::RunnerExt> GraphRunner::create_graph_runner(
    const xir::Graph* graph1, xir::Attrs* attrs) {
  auto graph = const_cast<xir::Graph*>(graph1);
  auto subgraph = graph->get_root_subgraph();

  if (!subgraph->has_attr("device")) {
    subgraph->set_attr<std::string>("device", "graph");
  }

  if (!subgraph->has_attr("runner")) {
    subgraph->set_attr<std::map<std::string, std::string>>(
        "runner",
        {{"ref", GRAPH_RUNNER}, {"sim", GRAPH_RUNNER}, {"run", GRAPH_RUNNER}});
  }

  if (ENV_PARAM(DEBUG_GRAPH_RUNNER_USE_CPU_TASK)) {
    if (subgraph->get_attr<std::string>("device") == "CPU") {
      subgraph->set_attr<std::map<std::string, std::string>>(
          "runner", {{"ref", CPU_TASK}, {"sim", CPU_TASK}, {"run", CPU_TASK}});
    }
  }

  auto runner = vart::RunnerExt::create_runner(subgraph, attrs);
  return runner;
}

}  // namespace ai
}  // namespace vitis

#include "vitis/ai/graph_runner.h"
extern "C" vart_runner_t vai_lib_create_graph_runner(xir_graph_t graph,
                                                     xir_attrs_t attrs) {
  return static_cast<vart_runner_t>(vitis::ai::GraphRunner::create_graph_runner(
                                        static_cast<const xir::Graph*>(graph),
                                        static_cast<xir::Attrs*>(attrs))
                                        .release());
}
