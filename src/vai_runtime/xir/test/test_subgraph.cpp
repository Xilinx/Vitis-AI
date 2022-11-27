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

#include <iostream>
#include <string>
#include <vector>

#include "UniLog/UniLog.hpp"
#include "xir/attrs/attrs.hpp"
#include "xir/graph/graph.hpp"

std::shared_ptr<xir::Graph> create_test_graph();

int main(int, char* argv[]) {
  UniLog::Initial(argv[0], UNI_LOG_STD, UNI_LOG_LEVEL_INFO,
                  UNI_LOG_STD_LEVEL_INFO);
  auto graph = create_test_graph();
  UNI_LOG_INFO << "create graph " << graph->get_name();
  auto root = graph->get_root_subgraph();
  UNI_LOG_INFO << "root subgraph is_root "
               << (root->is_root() ? "true" : "false");
  UNI_LOG_INFO << "root subgraph is_leaf "
               << (root->is_leaf() ? "true" : "false");
  UNI_LOG_INFO << "root subgraph op num " << root->get_op_num();

  UNI_LOG_INFO << "root subgraph children num " << root->get_children_num();
  root->create_children();
  UNI_LOG_INFO << "root subgraph children num " << root->get_children_num();
  UNI_LOG_INFO << "root subgraph is_root "
               << (root->is_root() ? "true" : "false");
  UNI_LOG_INFO << "root subgraph is_leaf "
               << (root->is_leaf() ? "true" : "false");

  auto dpu =
      root->merge_children({root->find_op("data"), root->find_op("conv0_w"),
                            root->find_op("conv0_b"), root->find_op("conv0"),
                            root->find_op("conv1_w"), root->find_op("conv1_b"),
                            root->find_op("conv1"), root->find_op("conv2_w"),
                            root->find_op("conv2_b"), root->find_op("conv2")});
  UNI_LOG_INFO << "dpu subgraph op num " << dpu->get_op_num();
  auto cpu =
      root->merge_children({root->find_op("concat"), root->find_op("conv3_w"),
                            root->find_op("conv3_b"), root->find_op("conv3")});
  UNI_LOG_INFO << "cpu subgraph op num " << cpu->get_op_num();
  UNI_LOG_INFO << "root subgraph children num " << root->get_children_num();

  UNI_LOG_INFO << "dpu subgraph children num " << dpu->get_children_num();
  dpu->create_children();
  UNI_LOG_INFO << "dpu subgraph children num " << dpu->get_children_num();
  auto super0 = dpu->merge_children(
      {dpu->find_op("data"), dpu->find_op("conv0_w"), dpu->find_op("conv0_b"),
       dpu->find_op("conv0"), dpu->find_op("conv1_w"), dpu->find_op("conv1_b"),
       dpu->find_op("conv1")});
  UNI_LOG_INFO << "super0 subgraph op num " << super0->get_op_num();
  auto super1 =
      dpu->merge_children({dpu->find_op("conv2_w"), dpu->find_op("conv2_w"),
                           dpu->find_op("conv2_b"), dpu->find_op("conv2")});
  UNI_LOG_INFO << "super1 subgraph op num " << super1->get_op_num();
  UNI_LOG_INFO << "dpu subgraph children num " << dpu->get_children_num();

  UNI_LOG_INFO << "dpu has data " << (dpu->has_op("data") ? "true" : "false");
  UNI_LOG_INFO << "super0 has data "
               << (super0->has_op("data") ? "true" : "false");
  UNI_LOG_INFO << "super1 has data "
               << (super1->has_op("data") ? "true" : "false");

  auto root_ref = super0->get_root();
  UNI_LOG_INFO << "root and super0->get_root is_same "
               << (root == root_ref ? "true" : "false");
  UNI_LOG_INFO << "root and dpu is_same " << (root == dpu ? "true" : "false");

  UNI_LOG_INFO << "root depth " << root->get_depth();
  UNI_LOG_INFO << "dpu depth " << dpu->get_depth();
  UNI_LOG_INFO << "super0 depth " << super0->get_depth();

  auto dpu_ref = super1->get_parent();
  UNI_LOG_INFO << "super0 is child of super1->get_parent "
               << (dpu_ref->is_child(super1) ? "true" : "false");

  return 0;
}

xir::Op* add_conv(std::string name, std::shared_ptr<xir::Graph> graph,
                  xir::Op* input) {
  auto attrs = xir::Attrs::create();
  attrs->set_attr<std::vector<int>>("kernel", {3, 3});
  attrs->set_attr<std::vector<int>>("stride", {3, 3});
  attrs->set_attr<int>("pad_mode", 0);
  attrs->set_attr<std::vector<int>>("pad", {1, 1, 1, 1});

  auto weights = graph->add_op(name + "_w", "const", xir::Attrs::create(), {},
                               xir::DataType{"FLOAT32"});
  auto bias = graph->add_op(name + "_b", "const", xir::Attrs::create(), {},
                            xir::DataType{"FLOAT32"});
  return graph->add_op(
      name, "conv2d", xir::Attrs::clone(attrs.get()),
      {{"weights", {weights}}, {"bias", {bias}}, {"input", {input}}},
      xir::DataType{"FLOAT32"});
}

std::shared_ptr<xir::Graph> create_test_graph() {
  std::shared_ptr<xir::Graph> graph = xir::Graph::create("graph_test");
  auto data = graph->add_op("data", "data", xir::Attrs::create(), {},
                            xir::DataType{"FLOAT32"});
  auto conv0 = add_conv("conv0", graph, data);
  auto conv1 = add_conv("conv1", graph, conv0);
  auto conv2 = add_conv("conv2", graph, conv0);
  auto concat =
      graph->add_op("concat", "concat", xir::Attrs::create(),
                    {{"input", {conv1, conv2}}}, xir::DataType{"FLOAT32"});
  add_conv("conv3", graph, concat);
  return graph;
}
