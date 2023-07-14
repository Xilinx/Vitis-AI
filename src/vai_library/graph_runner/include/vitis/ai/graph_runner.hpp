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
#include "vart/runner_ext.hpp"
#include "xir/graph/graph.hpp"

namespace vitis {
namespace ai {

class GraphRunner {
 public:
  /**
   * @brief Factory fucntion to create an instance of runner by
   * graph and attributes
   * @param graph  XIR Graph
   * @param attrs XIR attrs object, this object is shared among all
   * runners on the same graph.
   * @return An instance of runner.
   
   Usage:

   @code
    auto graph = xir::Graph::deserialize(xmodel_file);
    auto attrs = xir::Attrs::create();
    auto runner = vitis::ai::GraphRunner::create_graph_runner(graph.get(), attrs.get());
    auto input_tensor_buffers = runner->get_inputs();
   @endcode
   
   Graph runner Example
   
   Sample code:

   @code
   // The way to create graph runner and the APIs usage of runner are shown below.
   auto graph = xir::Graph::deserialize(xmodel_file);
   auto attrs = xir::Attrs::create();
   auto runner = vitis::ai::GraphRunner::create_graph_runner(graph.get(), attrs.get());
   // get input and output tensor buffers
   auto input_tensor_buffers = runner->get_inputs();
   auto output_tensor_buffers = runner->get_outputs();
   // sync input tensor buffers
   for (auto& input : input_tensor_buffers) {
       input->sync_for_write(0, input->get_tensor()->get_data_size() /
       input->get_tensor()->get_shape()[0]);
   }
   // run graph runner
   auto v = runner->execute_async(input_tensor_buffers, output_tensor_buffers);
   auto status = runner->wait((int)v.first, 1000000000);
   // sync output tensor buffers
   for (auto& output : output_tensor_buffers) {
       output->sync_for_read(0, output->get_tensor()->get_data_size() /
       output->get_tensor()->get_shape()[0]);
   }
   @endcode
   */
  static std::unique_ptr<vart::RunnerExt> create_graph_runner(
      const xir::Graph* graph, xir::Attrs* attrs);
};

}  // namespace ai
}  // namespace vitis
