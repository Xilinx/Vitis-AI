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

std::shared_ptr<xir::Graph> create_test_graph();

int main(int, char* argv[]) {
  UniLog::Initial(argv[0], UNI_LOG_STD, UNI_LOG_LEVEL_INFO,
                  UNI_LOG_STD_LEVEL_INFO);
  auto graph = create_test_graph();
  auto root = graph->get_root_subgraph();
  root->create_children();
  auto child = root->merge_children({root->find_op("conv0")});
  for(auto tensors_i : child->get_input_tensors())
    UNI_LOG_INFO << "Conv0 Input Tensors: " << tensors_i->get_name();
  for(auto tensors_sort_i : child->get_sorted_input_tensors())
    UNI_LOG_INFO << "Conv0 Input Sorted Tensors: " << tensors_sort_i->get_name();

  for(auto tensors_o : root->get_output_tensors())
    UNI_LOG_INFO << "Graph Output Tensors: " << tensors_o->get_name();
  for(auto tensors_sort_o : root->get_sorted_output_tensors())
    UNI_LOG_INFO << "Graph Output Sorted Tensors: " << tensors_sort_o->get_name();
  return 0;
}

xir::Op* add_conv(std::string name, std::shared_ptr<xir::Graph> graph,
                  xir::Op* input) {
  auto attrs = xir::Attrs::create();
  attrs->set_attr<std::vector<int>>("kernel", {3, 3});
  attrs->set_attr<std::vector<int>>("stride", {3, 3});
  attrs->set_attr<std::string>("pad_mode", "SAME");
  attrs->set_attr<std::vector<int>>("pad", {1, 1, 1, 1});

  auto weights_attrs = xir::Attrs::create();
  auto weights_data = std::vector<char>(3 * 3 * 3 * 3 * 4, 1);
  weights_attrs->set_attr<std::vector<std::int32_t>>("shape", {3, 3, 3, 3})
      ->set_attr<std::string>("data_type", "FLOAT32")
      ->set_attr<std::vector<char>>("data", weights_data);
  auto weights = graph->add_op(name + "_w",               //
                               "const",                   //
                               std::move(weights_attrs),  //
                               {});
  auto bias_attrs = xir::Attrs::create();
  auto bias_data = std::vector<char>(3 * 4, 1);
  bias_attrs->set_attr<std::vector<std::int32_t>>("shape", {3})
      ->set_attr<std::string>("data_type", "FLOAT32")
      ->set_attr<std::vector<char>>("data", weights_data);
  auto bias = graph->add_op(name + "_b",            //
                            "const",                //
                            std::move(bias_attrs),  //
                            {});
  return graph->add_op(
      name,              //
      "conv2d",          //
      std::move(attrs),  //
      {{"weights", {weights}}, {"bias", {bias}}, {"input", {input}}});
}

std::shared_ptr<xir::Graph> create_test_graph() {
  std::shared_ptr<xir::Graph> graph = xir::Graph::create("graph_test");
  auto data_attrs = xir::Attrs::create();
  data_attrs->set_attr<std::vector<std::int32_t>>("shape", {1, 224, 224, 3})
      ->set_attr<std::string>("data_type", "FLOAT32");
  auto data = graph->add_op("data0","data",std::move(data_attrs),{});
  add_conv("conv2", graph, data);
  add_conv("conv1", graph, data);
  add_conv("conv0", graph, data);
  return graph;
}
