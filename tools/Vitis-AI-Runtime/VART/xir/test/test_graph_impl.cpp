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
#include "xir/util/data_type.hpp"

int main(int, char* argv[]) {
  UniLog::Initial(argv[0], UNI_LOG_STD, UNI_LOG_LEVEL_INFO,
                  UNI_LOG_STD_LEVEL_INFO);

  auto g = xir::Graph::create("test");
  g->set_attr<std::string>("test", "test");
  UNI_LOG_CHECK(g->has_attr("test"), ERROR_SAMPLE);
  UNI_LOG_INFO << g->get_attr<std::string>("test");

  UNI_LOG_INFO << "Graph name: " << g->get_name();
  auto w1 = g->add_op("conv1_w", "const", xir::Attrs::create(), {},
                      xir::DataType{"FLOAT32"});
  UNI_LOG_INFO << "Add Op " << w1->get_name();
  auto i = g->add_op("input_data", "data", xir::Attrs::create(), {},
                     xir::DataType{"FLOAT32"});
  UNI_LOG_INFO << "Add Op " << i->get_name();
  auto attrs = xir::Attrs::create();
  attrs->set_attr<std::vector<int>>("kernel", {3, 3});
  attrs->set_attr<std::vector<int>>("stride", {3, 3});
  attrs->set_attr<int>("pad_mode", 0);
  attrs->set_attr<std::vector<int>>("pad", {1, 1, 1, 1});
  std::cout << attrs->debug_info() << std::endl;
  auto c1 =
      g->add_op("conv1", "conv2d", xir::Attrs::clone(attrs.get()),
                {{"weights", {w1}}, {"input", {i}}}, xir::DataType{"FLOAT32"});
  UNI_LOG_INFO << "Add Op " << c1->get_name();

  auto w2 = g->add_op("conv2_w", "const", xir::Attrs::create(), {},
                      xir::DataType{"FLOAT32"});
  UNI_LOG_INFO << "Add Op " << w2->get_name();
  auto c2 =
      g->add_op("conv2", "conv2d", xir::Attrs::clone(attrs.get()),
                {{"weights", {w2}}, {"input", {c1}}}, xir::DataType{"FLOAT32"});
  UNI_LOG_INFO << "Add Op " << c2->get_name();

  auto topo = g->topological_sort();
  for (auto op : topo) {
    std::cout << op->get_name() << std::endl;
  }
  g->remove_op(c2);
  g->visualize("test", "png");

  auto o = g->get_op("conv1_w");
  std::cout << "find " << o->get_name() << std::endl;

  return 0;
}
