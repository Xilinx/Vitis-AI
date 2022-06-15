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
#include "vart/runner_ext.hpp"
#include "xir/graph/graph.hpp"

namespace vitis {
namespace ai {

class GraphRunner {
 public:
  /**
   * @brief Factory fucntion to create an instance of runner by
   * graph and attrs
   * @param graph  XIR Graph
   * @param attrs XIR attrs object, this object is shared among all
   * runners on the same graph.
   * @return An instance of runner.
   */
  static std::unique_ptr<vart::RunnerExt> create_graph_runner(
      const xir::Graph* graph, xir::Attrs* attrs);
};

}  // namespace ai
}  // namespace vitis
