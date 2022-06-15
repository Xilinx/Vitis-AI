/*
 * Copyright 2019 Xilinx Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <cmath>

/* header file for Vitis AI unified API */
#include <vart/mm/host_flat_tensor_buffer.hpp>
#include <vart/runner.hpp>
#include <vart/runner_ext.hpp>
#include <xir/graph/graph.hpp>
#include <xir/tensor/tensor.hpp>
#include <xir/util/data_type.hpp>

struct TensorShape {
  unsigned int height;
  unsigned int width;
  unsigned int channel;
  unsigned int size;
};

struct GraphInfo {
  struct TensorShape* inTensorList;
  struct TensorShape* outTensorList;
  std::vector<int> output_mapping;
};

int getTensorShape(vart::Runner* runner, GraphInfo* shapes, int cntin,
    const std::vector<std::string> output_names);
int getTensorShape(vart::Runner* runner, GraphInfo* shapes, int cntin,
    int cnout);

inline std::vector<const xir::Subgraph*> get_dpu_subgraph (
    const xir::Graph* graph) {

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
