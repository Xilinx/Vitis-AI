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

#include "quantize_utils.h"
#include "separate_shared_constants.h"

namespace tensorflow {
namespace decent_q {

Status SeparateSharedConstants(const GraphDef &input_graph_def,
                               GraphDef *output_graph_def, bool is_quantized) {
  GraphDef current_graph_def;
  std::map<string, const NodeDef *> nodes_map;
  MapNamesToNodes(input_graph_def, &nodes_map);

  std::unordered_set<string> convfc_types{
      "Conv2D", "DepthwiseConv2d", "DepthwiseConv2dNative",
      "MatMul", "Conv3D",          "BiasAdd"};
  if (is_quantized) {
    std::unordered_map<string, int> weight_nodes;
    std::unordered_map<string, int> fn_nodes;
    std::unordered_map<string, int> fn_input_nodes;
    std::unordered_map<string, string> convfc_weight_map;
    std::unordered_map<string, string> convfc_fn_map;
    std::unordered_map<string, string> convfc_fn_input_map;

    // record conv to weigths map and necessary node info for different pattern
    for (auto i = 0; i < input_graph_def.node_size(); ++i) {
      const NodeDef &cur_node = input_graph_def.node(i);
      if (convfc_types.count(cur_node.op())) {
        NodeDef fn_node = *nodes_map[cur_node.input(1)];
        NodeDef fn_input_node = *nodes_map[fn_node.input(0)];
        if (fn_input_node.op() == "Mul") {
          // folded bn pattern, weights; seperate weigths
          weight_nodes[fn_input_node.input(1)] = 0;
          convfc_weight_map[cur_node.name()] = fn_input_node.input(1);
          fn_input_nodes[fn_input_node.name()] = 0;
          convfc_fn_input_map[cur_node.name()] = fn_input_node.name();
        } else if (fn_input_node.op() == "Sub") {
          // folded bn pattern, bias; do nothing
        } else if (fn_input_node.op() == "Add") {
          // folded conv + bias + bn pattern, bias; seperate bias
          NodeDef mul_node = *nodes_map[fn_input_node.input(1)];
          weight_nodes[mul_node.input(1)] = 0;
          convfc_weight_map[cur_node.name()] = mul_node.input(1);
          fn_input_nodes[mul_node.name()] = 0;
          convfc_fn_input_map[cur_node.name()] = mul_node.name();
        } else if (fn_input_node.op() == "Const") {
          // regular conv pattern pattern
          fn_nodes[fn_node.name()] = 0;
          convfc_fn_map[cur_node.name()] = fn_node.name();
          weight_nodes[fn_input_node.name()] = 0;
          convfc_weight_map[cur_node.name()] = fn_input_node.name();
        } else {
          DLOG_WARNING << "Found unknown input type for fix neuron "
                       << fn_node.name();
        }
      }
    }

    for (auto i = 0; i < input_graph_def.node_size(); ++i) {
      const NodeDef &cur_node = input_graph_def.node(i);
      if (weight_nodes.count(cur_node.name()) ||
          fn_nodes.count(cur_node.name()) ||
          fn_input_nodes.count(cur_node.name())) {
        // Skip weights node, fix neuron nodes, input nodes of fix neuron node
        // these will be copied below
        DLOG_INFO(2) << "Skip node in seperate shared weights: "
                     << cur_node.name();
      } else if (convfc_weight_map.count(cur_node.name())) {
        if (convfc_fn_map.count(cur_node.name())) {
          // regular conv pattern
          const string &weight_name = convfc_weight_map[cur_node.name()];
          const string &fn_name = convfc_fn_map[cur_node.name()];
          if (weight_nodes[weight_name] == 0) {
            *(current_graph_def.mutable_node()->Add()) =
                *nodes_map[weight_name];
            *(current_graph_def.mutable_node()->Add()) = *nodes_map[fn_name];
            *(current_graph_def.mutable_node()->Add()) = cur_node;
          } else {
            // Copy and change weight name and fix neuron name, reconnect
            NodeDef new_weight = *nodes_map[weight_name];
            new_weight.set_name(new_weight.name() +
                                std::to_string(weight_nodes[weight_name]));
            NodeDef new_fn = *nodes_map[fn_name];
            new_fn.set_name(new_fn.name() + std::to_string(fn_nodes[fn_name]));
            new_fn.set_input(0, new_weight.name());
            NodeDef new_convfc = cur_node;
            new_convfc.set_input(1, new_fn.name());
            *(current_graph_def.mutable_node()->Add()) = new_weight;
            *(current_graph_def.mutable_node()->Add()) = new_fn;
            *(current_graph_def.mutable_node()->Add()) = new_convfc;
            DLOG_INFO(1) << "increase weight: " << new_weight.name();
          }
          weight_nodes[weight_name] = weight_nodes[weight_name] + 1;
          fn_nodes[fn_name] = fn_nodes[fn_name] + 1;
        } else if (convfc_fn_input_map.count(cur_node.name())) {
          // folded bn conv pattern
          const string &weight_name = convfc_weight_map[cur_node.name()];
          const string &fn_input_name = convfc_fn_input_map[cur_node.name()];
          if (weight_nodes[weight_name] == 0) {
            *(current_graph_def.mutable_node()->Add()) =
                *nodes_map[weight_name];
            *(current_graph_def.mutable_node()->Add()) =
                *nodes_map[fn_input_name];
            *(current_graph_def.mutable_node()->Add()) = cur_node;
          } else {
            // Copy and change weight name reconnect node
            NodeDef new_weight = *nodes_map[weight_name];
            new_weight.set_name(new_weight.name() +
                                std::to_string(weight_nodes[weight_name]));
            NodeDef new_convfc = cur_node;
            NodeDef fn_input_node = *nodes_map[fn_input_name];
            NodeDef new_fn_input_node = fn_input_node;
            new_fn_input_node.set_input(1, new_weight.name());
            *(current_graph_def.mutable_node()->Add()) = new_weight;
            *(current_graph_def.mutable_node()->Add()) = new_fn_input_node;
            *(current_graph_def.mutable_node()->Add()) = new_convfc;
            DLOG_INFO(1) << "increase weight: " << new_weight.name();
          }
          weight_nodes[weight_name] = weight_nodes[weight_name] + 1;
        }
      } else {
        *(current_graph_def.mutable_node()->Add()) = cur_node;
      }
    }

    *output_graph_def = current_graph_def;
    return Status::OK();
  }
  // for graph that have not inserted fix neuron
  std::unordered_map<string, int> weight_nodes;
  std::unordered_map<string, string> convfc_weight_map;

  for (auto i = 0; i < input_graph_def.node_size(); ++i) {
    const NodeDef &cur_node = input_graph_def.node(i);
    if (convfc_types.count(cur_node.op())) {
      weight_nodes[cur_node.input(1)] = 0;
      convfc_weight_map[cur_node.name()] = cur_node.input(1);
    }
  }

  for (auto i = 0; i < input_graph_def.node_size(); ++i) {
    const NodeDef &cur_node = input_graph_def.node(i);
    if (weight_nodes.count(cur_node.name())) {
      // Skip weights node, it will be copied below
    } else if (convfc_weight_map.count(cur_node.name())) {
      const string &weight_name = convfc_weight_map[cur_node.name()];
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
Status SeparateSharedConstantsCommand(const GraphDef &input_graph_def,
                                      const TransformFuncContext &context,
                                      GraphDef *output_graph_def) {
  return (SeparateSharedConstants(input_graph_def, output_graph_def));
}

REGISTER_DECENT_Q_GRAPH_TRANSFORM("separate_shared_constants",
                                  SeparateSharedConstantsCommand);

} // namespace decent_q
} // namespace tensorflow
