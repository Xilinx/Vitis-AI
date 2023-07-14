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
#include <glog/logging.h>
#include <google/protobuf/message.h>

#include <fstream>
#include <iostream>
#include <xir/tensor/tensor.hpp>

#include "vart/assistant/tensor_buffer_allocator.hpp"
#include "vart/tensor_buffer.hpp"
#include "vitis/ai/collection_helper.hpp"
#include "vitis/ai/env_config.hpp"
#include "vitis/ai/parse_value.hpp"
#include "xir/graph/graph.hpp"

using namespace std;
DEF_ENV_PARAM(BATCH, "1");
DEF_ENV_PARAM(LOCATION, "0");
DEF_ENV_PARAM_2(CU_NAME, "DPU", std::string);
int main(int argc, char* argv[]) {
  if (((argc - 2) - 1) % 3 != 0 || ((argc - 2) - 1) / 3 == 0) {
    cout << "usage: test_allocator <xmodel> <subgraph_idx> <reg_id> <offset> "
            "<size> ..."
         << endl;
    return 0;
  }
  {
    auto graph = xir::Graph::deserialize(std::string(argv[1]));
    auto subgraph_idx = std::stoi(std::string(argv[2]));
    std::vector<std::unique_ptr<xir::Tensor>> input_tensors;
    std::vector<std::unique_ptr<xir::Tensor>> output_tensors;
    for (auto i = 3; i < argc - 3; i = i + 3) {
      size_t reg = 0ul;
      vitis::ai::parse_value(std::string{argv[i + 0]}, reg);
      size_t offset = 0ul;
      vitis::ai::parse_value(std::string{argv[i + 1]}, offset);
      size_t size = 0ul;
      vitis::ai::parse_value(std::string{argv[i + 2]}, size);
      auto dims = std::vector<int32_t>{ENV_PARAM(BATCH), (int)size};
      input_tensors.emplace_back(
          xir::Tensor::create(std::string("tesnor_") + std::to_string(i / 3),
                              dims, xir::DataType{xir::DataType::XINT, 8}));
      input_tensors.back()->set_attr<int>("reg_id", (int)reg);
      input_tensors.back()->set_attr<int>("ddr_addr", (int)offset);
    }
    for (auto i = argc - 3; i < argc; i = i + 3) {
      size_t reg = 0ul;
      vitis::ai::parse_value(std::string{argv[i + 0]}, reg);
      size_t offset = 0ul;
      vitis::ai::parse_value(std::string{argv[i + 1]}, offset);
      size_t size = 0ul;
      vitis::ai::parse_value(std::string{argv[i + 2]}, size);
      auto dims = std::vector<int32_t>{ENV_PARAM(BATCH), (int)size};
      output_tensors.emplace_back(
          xir::Tensor::create(std::string("tesnor_") + std::to_string(i / 3),
                              dims, xir::DataType{xir::DataType::XINT, 8}));
      output_tensors.back()->set_attr<int>("reg_id", (int)reg);
      output_tensors.back()->set_attr<int>("ddr_addr", (int)offset);
    }

    auto subgraph =
        graph->get_root_subgraph()->children_topological_sort()[subgraph_idx];
    auto attrs = xir::Attrs::create();
    attrs->set_attr<size_t>("__device_id__", 0u);
    attrs->set_attr<size_t>("__batch__", ENV_PARAM(BATCH));
    attrs->set_attr<int>("__tensor_buffer_location__", ENV_PARAM(LOCATION));
    attrs->set_attr<std::string>("__cu_name__", ENV_PARAM(CU_NAME));
    auto allocator =
        vart::assistant::TensorBufferAllocator::create(attrs.get());
    std::vector<std::pair<std::vector<std::unique_ptr<vart::TensorBuffer>>,
                          std::vector<std::unique_ptr<vart::TensorBuffer>>>>
        all(2u);
    for (auto count = 0u; count < all.size(); count++) {
      all[count] = allocator->allocate(
          subgraph, vitis::ai::vector_unique_ptr_get_const(input_tensors),
          vitis::ai::vector_unique_ptr_get_const(output_tensors));
      for (auto i = 0u; i < all[count].first.size(); ++i) {
        cout << __FILE__ << ":" << __LINE__ << ": [" << __FUNCTION__
             << "] input tensor buffer : " << (all[count].first)[i]->to_string()
             << " " << endl;
      }
      for (auto i = 0u; i < all[count].second.size(); ++i) {
        cout << __FILE__ << ":" << __LINE__ << ": [" << __FUNCTION__
             << "] output tensor buffer : "
             << (all[count].second)[i]->to_string() << " " << endl;
      }
    }
    LOG(INFO) << "press enter to releaes memory and continue ... \n"
                 "you can use xbutil query -d 0 to check memory usage ..."
              << endl;
  }
  char c = 0;
  std::cin >> c;
  LOG(INFO) << "BYEBYE";
  return 0;
}
