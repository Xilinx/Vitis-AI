/* Copyright 2019 Xilinx Inc.
**
** Licensed under the Apache License, Version 2.0 (the "License");
** you may not use this file except in compliance with the License.
** You may obtain a copy of the License at
**
**     http://www.apache.org/licenses/LICENSE-2.0
**
** Unless required by applicable law or agreed to in writing, software
** distributed under the License is distributed on an "AS IS" BASIS,
** WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
** See the License for the specific language governing permissions and
** limitations under the License.
*/

#include "plugin-smfc/plugin-smfc.hpp"

std::string plugin_smfc::get_plugin_name() { return "plugin_smfc"; }

std::string plugin_smfc::get_device() { return "SMFC"; }

std::map<std::string, std::string> plugin_smfc::get_runner() {
  return {
    {"run", "libvart-softmax-runner.so"},  //
    {"sim", "libvart-softmax-runner.so"},  //
    {"ref", "libvart-softmax-runner.so"},  //
  };
}

std::set<xir::Subgraph*> plugin_smfc::partition(xir::Graph* graph) {
  auto softmaxs = PluginHelper::filter_by_type(graph, "softmax");
  std::set<xir::Subgraph*> targets;
  for (auto softmax : softmaxs) {
    auto op = *(softmax->get_ops().begin());
    auto input = op->get_input_ops("input")[0];
    if (input->get_type() == "fix") {
      bool if_signed = input->has_attr("if_signed")
                         ? input->get_attr<bool>("if_signed")
                         : true;
      auto dtype = if_signed ? xir::DataType::XINT : xir::DataType::XUINT;
      auto bit_width = input->get_attr<int>("bit_width");
      auto op_float2fix =
        graph->add_op(input->get_name() + "_float2fix",
                      "float2fix",
                      input->get_attrs(),
                      {{"input", input->get_input_ops("input")}},
                      xir::DataType{dtype, bit_width},
                      graph->get_leaf_subgraph(input)->get_parent());
      op_float2fix->get_output_tensor()->set_attrs(input->get_attrs());
      auto op_fix2float =
        graph->add_op(input->get_name() + "_fix2float",
                      "fix2float",
                      input->get_attrs(),
                      {{"input", {op_float2fix}}},
                      xir::DataType{xir::DataType::FLOAT, 32},
                      graph->get_leaf_subgraph(input)->get_parent());
      op_fix2float->get_output_tensor()->set_attrs(input->get_attrs());
      op->replace_input_op(input, op_fix2float);
      graph->remove_op(input);
      auto merged = PluginHelper::merge_subgraph(
        {graph->get_root_subgraph()->find_op(op_fix2float), softmax});
      std::transform(merged.begin(),
                     merged.end(),
                     std::inserter(targets, targets.end()),
                     [](auto sub) { return sub; });
    }
  }
  return targets;
}

void plugin_smfc::compile(xir::Subgraph* subgraph) {
  subgraph->set_attr("generated_code", std::string("HELLO WORLD"));
}
