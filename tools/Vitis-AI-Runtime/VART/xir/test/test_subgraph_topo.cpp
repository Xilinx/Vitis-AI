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
void test_topo(xir::Subgraph* sub);
void test_children_topo(xir::Subgraph* sub);

int main(int, char* argv[]) {
  UniLog::Initial(argv[0], UNI_LOG_STD, UNI_LOG_LEVEL_INFO,
                  UNI_LOG_STD_LEVEL_INFO);
  auto graph = create_test_graph();
  auto root = graph->get_root_subgraph();
  root->set_attr<std::string>("name", "root");

  root->create_children();
  auto dpu = root->merge_children({
      root->find_op("data"),
      root->find_op("conv0_w"),
      root->find_op("conv0_b"),
      root->find_op("conv0"),
      root->find_op("conv1_w"),
      root->find_op("conv1_b"),
      root->find_op("conv1"),
      root->find_op("conv2_w"),
      root->find_op("conv2_b"),
      root->find_op("conv2"),
      root->find_op("concat"),
  });
  dpu->set_attr<std::string>("name", "dpu");

  auto cpu = root->merge_children({
      root->find_op("conv3_w"),
      root->find_op("conv3_b"),
      root->find_op("conv3"),
  });
  cpu->set_attr<std::string>("name", "cpu");

  dpu->create_children();
  auto super0 = dpu->merge_children({
      dpu->find_op("data"),
      dpu->find_op("conv0_w"),
      dpu->find_op("conv0_b"),
      dpu->find_op("conv0"),
  });
  super0->set_attr<std::string>("name", "super0");
  auto super1 = dpu->merge_children({
      dpu->find_op("conv1_w"),
      dpu->find_op("conv1_b"),
      dpu->find_op("conv1"),
      dpu->find_op("concat"),
  });
  super1->set_attr<std::string>("name", "super1");
  auto super2 = dpu->merge_children({
      dpu->find_op("conv2_w"),
      dpu->find_op("conv2_b"),
      dpu->find_op("conv2"),
  });
  super2->set_attr<std::string>("name", "super2");

  test_children_topo(root);
  test_children_topo(dpu);

  test_topo(root);
  test_topo(dpu);
  test_topo(super0);
  test_topo(super1);

  return 0;
}

void test_topo(xir::Subgraph* sub) {
  auto topo = sub->topological_sort();
  std::cout << "topo sort " << sub->get_attr<std::string>("name") << ": ";
  for (auto elem : topo) {
    std::cout << elem->get_name() << " ";
  }
  std::cout << std::endl;
}

void test_children_topo(xir::Subgraph* sub) {
  auto topo = sub->children_topological_sort();
  std::cout << "children topo sort " << sub->get_attr<std::string>("name")
            << ": ";
  for (auto elem : topo) {
    std::cout << elem->get_attr<std::string>("name") << " ";
  }
  std::cout << std::endl;
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
