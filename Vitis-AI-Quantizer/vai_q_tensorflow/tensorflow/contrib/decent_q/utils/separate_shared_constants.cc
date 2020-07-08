/*Copyright 2019 Xilinx Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.*/

/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/contrib/decent_q/utils/quantize_utils.h"
#include "tensorflow/contrib/decent_q/utils/separate_shared_constants.h"

namespace tensorflow {
namespace decent_q {

Status SeparateSharedConstants(const GraphDef& input_graph_def,
                               GraphDef* output_graph_def) {
  GraphDef current_graph_def;
  std::map<string, const NodeDef*> nodes_map;
  MapNamesToNodes(input_graph_def, &nodes_map);

  std::unordered_set<string> convfc_types{
      "Conv2D", "DepthwiseConv2d", "DepthwiseConv2dNative",
      "MatMul", "Conv3D",          "BiasAdd"};
  std::unordered_map<string, int> weight_nodes;
  std::unordered_map<string, string> convfc_weight_map;

  for (auto i = 0; i < input_graph_def.node_size(); ++i) {
    const NodeDef& cur_node = input_graph_def.node(i);
    if (convfc_types.count(cur_node.op())) {
      weight_nodes[cur_node.input(1)] = 0;
      convfc_weight_map[cur_node.name()] = cur_node.input(1);
    }
  }

  for (auto i = 0; i < input_graph_def.node_size(); ++i) {
    const NodeDef& cur_node = input_graph_def.node(i);
    if (weight_nodes.count(cur_node.name())) {
      // Skip
    } else if (convfc_weight_map.count(cur_node.name())) {
      const string& weight_name = convfc_weight_map[cur_node.name()];
      if (weight_nodes[weight_name] == 0) {
        *(current_graph_def.mutable_node()->Add()) = *nodes_map[weight_name];
        *(current_graph_def.mutable_node()->Add()) = cur_node;
      } else {
        // Copy and change weight name
        NodeDef new_weight = *nodes_map[weight_name];
        new_weight.set_name(new_weight.name() +
                            std::to_string(weight_nodes[weight_name]));
        NodeDef new_convfc = cur_node;
        new_convfc.set_input(1, new_weight.name());
        *(current_graph_def.mutable_node()->Add()) = new_weight;
        *(current_graph_def.mutable_node()->Add()) = new_convfc;
        DLOG_INFO(1) << "increase weight: " << new_weight.name();
      }
      weight_nodes[weight_name] = weight_nodes[weight_name] + 1;
    } else {
      *(current_graph_def.mutable_node()->Add()) = cur_node;
    }
  }

  *output_graph_def = current_graph_def;
  return Status::OK();
}

// Command Wrapper
Status SeparateSharedConstantsCommand(const GraphDef& input_graph_def,
                                      const TransformFuncContext& context,
                                      GraphDef* output_graph_def) {
  return (SeparateSharedConstants(input_graph_def, output_graph_def));
}

REGISTER_DECENT_Q_GRAPH_TRANSFORM("separate_shared_constants",
                                  SeparateSharedConstantsCommand);

}  // namespace decent_q
}  // namespace tensorflow
