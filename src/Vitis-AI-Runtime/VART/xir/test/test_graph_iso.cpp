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

int main(int, char* argv[]) {
  UniLog::Initial(argv[0], UNI_LOG_STD, UNI_LOG_LEVEL_INFO,
                  UNI_LOG_STD_LEVEL_INFO);
  auto graph = create_test_graph();

  auto graph_template = xir::GraphTemplate::create("conv-conv");
  auto conv0 = graph_template->add_op("conv0", {"conv2d"}, {});
  auto conv1 = graph_template->add_op("conv1", {"conv2d"}, {{conv0, ""}});
  UNI_LOG_INFO << graph_template->get_name() << ": " << conv0->get_name()
               << ", " << conv1->get_name();

  auto result = graph->isomorphism(graph_template.get());
  UNI_LOG_INFO << result.size();

  for (auto idx = 0U; idx < result.size(); ++idx) {
    std::cout << "result" << idx << ": ";
    std::cout << "conv0->" << result[idx][conv0]->get_name() << ", ";
    std::cout << "conv1->" << result[idx][conv1]->get_name() << std::endl;
  }

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
