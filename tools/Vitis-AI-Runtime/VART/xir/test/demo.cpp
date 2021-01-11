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

// helper function to add a conv
xir::Op* add_conv(std::string name, std::shared_ptr<xir::Graph> graph,
                  xir::Op* input);
// helper function to creat a graph
std::shared_ptr<xir::Graph> create_test_graph();

int main(int, char* argv[]) {
  UniLog::Initial(argv[0], UNI_LOG_STD, UNI_LOG_LEVEL_INFO,
                  UNI_LOG_STD_LEVEL_INFO);
  auto graph = create_test_graph();
  UNI_LOG_INFO << "Create graph: " << graph->to_string();
  auto root = graph->get_root_subgraph();
  UNI_LOG_INFO << "Get root subgraph: " << root->to_string();
  UNI_LOG_INFO << "Root subgraph is_root: "
               << (root->is_root() ? "true" : "false");
  UNI_LOG_INFO << "Root subgraph is_leaf: "
               << (root->is_leaf() ? "true" : "false");
  UNI_LOG_INFO << "Root subgraph op num: " << root->get_op_num();

  UNI_LOG_INFO << "Root subgraph children num: " << root->get_children_num();
  UNI_LOG_INFO << "Create children subgraph for the root " << root->to_string();
  root->create_children();
  UNI_LOG_INFO << "Root subgraph children num: " << root->get_children_num();
  UNI_LOG_INFO << "Root subgraph is_root: "
               << (root->is_root() ? "true" : "false");
  UNI_LOG_INFO << "Root subgraph is_leaf: "
               << (root->is_leaf() ? "true" : "false");

  UNI_LOG_INFO
      << "Create a DPU subgraph and a CPU subgraph to maintain the ops.";
  auto dpu =
      root->merge_children({root->find_op("data"), root->find_op("conv0_w"),
                            root->find_op("conv0_b"), root->find_op("conv0"),
                            root->find_op("conv1_w"), root->find_op("conv1_b"),
                            root->find_op("conv1"), root->find_op("conv2_w"),
                            root->find_op("conv2_b"), root->find_op("conv2")});
  UNI_LOG_INFO << "DPU subgraph op num: " << dpu->get_op_num();
  auto cpu =
      root->merge_children({root->find_op("concat"), root->find_op("conv3_w"),
                            root->find_op("conv3_b"), root->find_op("conv3")});
  UNI_LOG_INFO << "CPU subgraph op num: " << cpu->get_op_num();
  UNI_LOG_INFO << "Root subgraph children num: " << root->get_children_num();

  UNI_LOG_INFO << "DPU subgraph children num: " << dpu->get_children_num();
  dpu->create_children();
  UNI_LOG_INFO << "DPU subgraph children num: " << dpu->get_children_num();

  UNI_LOG_INFO << "DPU has data op: "
               << (dpu->has_op("data") ? "true" : "false");
  UNI_LOG_INFO << "CPU has data op: "
               << (cpu->has_op("data") ? "true" : "false");

  UNI_LOG_INFO << "Root depth: " << root->get_depth();
  UNI_LOG_INFO << "DPU depth: " << dpu->get_depth();
  UNI_LOG_INFO << "CPU depth: " << cpu->get_depth();

  auto root_ref = dpu->get_parent();
  UNI_LOG_INFO << dpu->to_string() << " is a child of " << root_ref->to_string()
               << ": " << (root_ref->is_child(dpu) ? "true" : "false");

  if (dpu->has_op("conv2")) {
    UNI_LOG_INFO << dpu->to_string() << " has an "
                 << dpu->find_op("conv2")->to_string();
  } else {
    UNI_LOG_WARNING << dpu->to_string()
                    << " doesn't have an op named \"conv2\".";
  }

  if (dpu->has_op("conv3")) {
    UNI_LOG_INFO << dpu->to_string() << " has an "
                 << dpu->find_op("conv3")->to_string();
  } else {
    UNI_LOG_WARNING << dpu->to_string()
                    << " doesn't have an op named \"conv3\".";
  }

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
  auto data = graph->add_op("data",                 //
                            "data",                 //
                            std::move(data_attrs),  //
                            {});
  auto conv0 = add_conv("conv0", graph, data);
  auto conv1 = add_conv("conv1", graph, conv0);
  auto conv2 = add_conv("conv2", graph, conv0);

  auto concat_attrs = xir::Attrs::create();
  concat_attrs->set_attr<std::int32_t>("axis", 3);
  auto concat = graph->add_op("concat",                 //
                              "concat",                 //
                              std::move(concat_attrs),  //
                              {{"input", {conv1, conv2}}});
  add_conv("conv3", graph, concat);
  return graph;
}
