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

#include "check_graph.h"
#include "file_utils.h"
#include "quantize_utils.h"

namespace tensorflow {
namespace decent_q {

void PrintNodeInfo(const NodeDef *node) {
  string shape_description = "None";
  if (node->attr().count("shape")) {
    TensorShapeProto shape_proto = node->attr().at("shape").shape();
    Status shape_status = PartialTensorShape::IsValidShape(shape_proto);
    if (shape_status.ok()) {
      shape_description = PartialTensorShape(shape_proto).DebugString();
    } else {
      shape_description = shape_status.error_message();
    }
  }
  DataType dtype = DT_INVALID;
  if (node->attr().count("dtype")) {
    dtype = node->attr().at("dtype").type();
  }
  std::cout << "(name=" << node->name();
  std::cout << ", type=" << DataTypeString(dtype) << "(" << dtype << ")";
  std::cout << ", shape=" << shape_description << ") ";
}

Status PrintStructure(const GraphDef &graph) {
  GraphDef sorted_graph;
  TF_RETURN_IF_ERROR(SortByExecutionOrder(graph, &sorted_graph));
  for (const NodeDef &node : sorted_graph.node()) {
    std::cout << node.name() << " (" << node.op() << "): ["
              << str_util::Join(node.input(), ", ") << "]";
    if (node.op() == "Const") {
      Tensor tensor;
      if (node.attr().count("value") &&
          tensor.FromProto(node.attr().at("value").tensor())) {
        std::cout << ", value=" << tensor.DebugString();
      } else {
        DLOG_WARNING << "Decoding Tensor failed for node" << node.name();
      }
    }
    std::cout << std::endl;
  }
  return Status::OK();
}

std::vector<const NodeDef *> FindInputs(const GraphDef &graph) {
  std::vector<const NodeDef *> placeholders;
  for (const NodeDef &node : graph.node()) {
    if (node.op() == "Placeholder") {
      placeholders.push_back(&node);
    }
  }

  if (placeholders.empty()) {
    std::cout << "No inputs found." << std::endl;
  } else {
    std::cout << "Found " << placeholders.size() << " possible inputs: ";
    for (const NodeDef *node : placeholders) {
      PrintNodeInfo(node);
    }
    std::cout << std::endl;
  }
  return placeholders;
}

std::vector<const NodeDef *> FindVariables(const GraphDef &graph) {
  std::vector<const NodeDef *> variables;
  for (const NodeDef &node : graph.node()) {
    if (node.op() == "Variable" || node.op() == "VariableV2") {
      variables.push_back(&node);
    }
  }

  if (variables.empty()) {
    std::cout << "No variables found." << std::endl;
  } else {
    std::cout << "Found " << variables.size() << " variables: ";
    for (const NodeDef *node : variables) {
      PrintNodeInfo(node);
    }
    std::cout << std::endl;
  }
  return variables;
}

std::vector<const NodeDef *> FindOutputs(const GraphDef &graph) {
  std::map<string, std::vector<const NodeDef *>> output_map;
  MapNodesToOutputs(graph, &output_map);
  std::vector<const NodeDef *> outputs;
  std::unordered_set<string> unlikely_output_types = {"Const", "Assign", "NoOp",
                                                      "Placeholder"};
  for (const NodeDef &node : graph.node()) {
    if ((output_map.count(node.name()) == 0) &&
        (unlikely_output_types.count(node.op()) == 0)) {
      outputs.push_back(&node);
    }
  }

  if (outputs.empty()) {
    std::cout << "No outputs found." << std::endl;
  } else {
    std::cout << "Found " << outputs.size() << " possible outputs: ";
    for (const NodeDef *node : outputs) {
      std::cout << "(name=" << node->name();
      std::cout << ", op=" << node->op() << ") ";
    }
    std::cout << std::endl;
  }
  return outputs;
}

std::map<string, int> GetOpCount(const GraphDef &graph) {
  std::map<string, int> op_count;
  for (const NodeDef &node : graph.node()) {
    ++op_count[node.op()];
  }
  for (const FunctionDef &function : graph.library().function()) {
    for (const NodeDef &node : function.node_def()) {
      ++op_count[node.op()];
    }
  }
  std::vector<std::pair<string, int>> op_count_vec(op_count.begin(),
                                                   op_count.end());
  std::sort(op_count_vec.begin(), op_count_vec.end(),
            [](std::pair<string, int> a, std::pair<string, int> b) {
              return (a.second > b.second);
            });
  std::cout << "Op types used: ";
  bool is_first = true;
  for (const std::pair<string, int> &op_count : op_count_vec) {
    if (!is_first) {
      std::cout << ", ";
    } else {
      is_first = false;
    }
    std::cout << op_count.second << " " << op_count.first;
  }
  std::cout << std::endl;
  return op_count;
}

int64 GetParameterCount(const GraphDef &graph) {
  int64 const_parameter_count = 0;
  for (const NodeDef &node : graph.node()) {
    if (node.op() == "Const") {
      Tensor tensor;
      if (node.attr().count("value") &&
          tensor.FromProto(node.attr().at("value").tensor())) {
        const size_t num_elements = tensor.NumElements();
        const_parameter_count += num_elements;
      } else {
        DLOG_WARNING << "Decoding Tensor failed for node" << node.name();
      }
    }
  }

  std::cout << "Found " << const_parameter_count << " ("
            << strings::HumanReadableNum(const_parameter_count)
            << ") const parameters." << std::endl;
  return const_parameter_count;
}

int64 GetVariableCount(const GraphDef &graph) {
  int64 variable_parameter_count = 0;
  for (const NodeDef &node : graph.node()) {
    if ((node.op() == "Variable") || (node.op() == "VariableV2")) {
      Tensor tensor;
      if (node.attr().count("value") &&
          tensor.FromProto(node.attr().at("value").tensor())) {
        const size_t num_elements = tensor.NumElements();
        variable_parameter_count += num_elements;
      } else {
        DLOG_WARNING << "Decoding Tensor failed for node" << node.name();
      }
    }
  }

  std::cout << "Found " << variable_parameter_count << " ("
            << strings::HumanReadableNum(variable_parameter_count)
            << ") variable parameters." << std::endl;
  return variable_parameter_count;
}

int GetControlEdgeCount(const GraphDef &graph) {
  int control_edge_count = 0;
  for (const NodeDef &node : graph.node()) {
    for (const string &input : node.input()) {
      if (input.substr(0, 1) == "^") {
        ++control_edge_count;
      }
    }
  }

  std::cout << "Found " << control_edge_count << " control_edges" << std::endl;
  return control_edge_count;
}

std::map<string, int> GetDeviceCount(const GraphDef &graph) {
  std::map<string, int> device_counts;
  for (const NodeDef &node : graph.node()) {
    if (!node.device().empty()) {
      ++device_counts[node.device()];
    }
  }

  if (!device_counts.empty()) {
    for (const auto &device_info : device_counts) {
      std::cout << device_info.second << " nodes assigned to device '"
                << device_info.first << "'";
    }
  }
  return device_counts;
}

} // namespace decent_q
} // namespace tensorflow
