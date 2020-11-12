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

#include "xir/attrs/attrs.hpp"
#include "xir/graph/graph.hpp"
#include "xir/op/op_def_factory_imp.hpp"

#include "UniLog/UniLog.hpp"
#include "xir/op/shape_inference.hpp"
#include "xir/util/data_type.hpp"

using namespace std;
using namespace xir;

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

int main(int, char* argv[]) {
  UniLog::Initial(argv[0], UNI_LOG_STD, UNI_LOG_LEVEL_INFO,
                  UNI_LOG_STD_LEVEL_INFO);

  std::shared_ptr<xir::Graph> graph = xir::Graph::create("graph_test");
  auto input_attr = xir::Attrs::create();
  input_attr->set_attr<std::vector<std::int32_t>>("shape", {1, 224, 224, 3});
  auto data = graph->add_op("data", "data", xir::Attrs::clone(input_attr.get()),
                            {}, xir::DataType{"FLOAT32"});
  auto conv0 = add_conv("conv0", graph, data);
  auto in = conv0->get_input_tensor("input");
  auto out = conv0->get_output_tensor();
  for (auto size : in->get_shape()) std::cout << size << " ";
  std::cout << "\n";
  for (auto size : out->get_shape()) std::cout << size << " ";
  std::cout << "\n";
  conv0->shape_infer();
  for (auto size : conv0->get_output_tensor()->get_shape())
    std::cout << size << " ";
  std::cout << "\n";
  return 0;
}
