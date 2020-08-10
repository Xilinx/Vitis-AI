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
/

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
#include "tensorflow/core/common_runtime/shape_refiner.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/public/session.h"

namespace tensorflow {
namespace decent_q {

template <typename T>
T quantize_kernel_cpu(const T x, const T step, const T lower_bound,
                      const T upper_bound) {
  return std::fmin(std::fmax(std::round(x / step) * step, lower_bound),
                   upper_bound);
}

// explicit instantiation
template float quantize_kernel_cpu<float>(const float x, const float step,
                                          const float lower_bound,
                                          const float upper_bound);
template double quantize_kernel_cpu<double>(const double x, const double step,
                                            const double lower_bound,
                                            const double upper_bound);

template <typename T>
void quantize_cpu(const int n, const T* x, T* y, const int bit_width,
                  const int pos) {
  T step = std::pow(T(2), -pos);
  T lower_bound = -std::pow(T(2), bit_width - 1) * step;
  T upper_bound = std::pow(T(2), bit_width - 1) * step - step;
  for (auto i = 0; i < n; ++i) {
    y[i] = quantize_kernel_cpu(x[i], step, lower_bound, upper_bound);
  }
}

// explicit instantiation
template void quantize_cpu<float>(const int n, const float* x, float* y,
                                  const int bit_width, const int pos);
template void quantize_cpu<double>(const int n, const double* x, double* y,
                                   const int bit_width, const int pos);

Status GetNodeTypeByName(const GraphDef& input_graph_def,
                         const string& node_name, string* node_type) {
  string real_node_name = NodeNameFromInput(node_name);
  for (auto i = 0; i < input_graph_def.node_size(); ++i) {
    if (input_graph_def.node(i).name() == real_node_name) {
      *node_type = input_graph_def.node(i).op();
      return Status::OK();
    }
  }
  return errors::NotFound(node_name, " not found in graphdef.");
}

Status RecordMatchedPatterns(
    std::vector<std::tuple<int, NodeMatch>>& matched_node_patterns,
    const int& pattern_id, const NodeMatch& match) {
  matched_node_patterns.emplace_back(pattern_id, match);
  return Status::OK();
}

Status RecordMatchedNodes(std::unordered_map<string, int>& matched_nodes,
                          const NodeMatch& match, const int& match_id,
                          const std::set<string>& irrelevant_nodes) {
  for (const NodeMatch& input_match : match.inputs) {
    RecordMatchedNodes(matched_nodes, input_match, match_id, irrelevant_nodes);
  }
  if (!irrelevant_nodes.count(match.node.name())) {
    DLOG_INFO(2) << "    record matched node: " << match.node.name();
    matched_nodes.insert(std::pair<string, int>(match.node.name(), match_id));
  }
  return Status::OK();
}

bool CheckAnyMatchedNodes(const std::unordered_map<string, int>& matched_nodes,
                          const NodeMatch& match,
                          const std::set<string>& irrelevant_nodes) {
  for (const NodeMatch& input_match : match.inputs) {
    if (CheckAnyMatchedNodes(matched_nodes, input_match, irrelevant_nodes)) {
      return true;
    }
  }
  if (irrelevant_nodes.count(match.node.name())) {
    return false;
  }
  if (matched_nodes.count(match.node.name())) {
    DLOG_INFO(2) << "    found matched nodes: " << match.node.name();
    return true;
  }
  return false;
}

void PrintMatchedNodePatterns(
    const std::vector<std::tuple<int, NodeMatch>>& matched_node_patterns) {
  DLOG_INFO(1) << "Matched Node Patterns:";
  for (int i = 0; i < matched_node_patterns.size(); ++i) {
    int pattern_id = std::get<0>(matched_node_patterns[i]);
    DLOG_INFO(1) << i << ": " << std::get<0>(known_patterns[pattern_id]);
    PrintNodeMatch(std::get<1>(matched_node_patterns[i]), 2);
  }
  return;
}

bool CheckAnyIgnoredNodes(const std::set<string>& ignore_nodes,
                          const NodeMatch& match,
                          const std::set<string>& irrelevant_nodes) {
  if (irrelevant_nodes.count(match.node.name())) {
    return false;
  }
  if (ignore_nodes.count(match.node.name())) {
    return true;
  }
  for (const NodeMatch& input_match : match.inputs) {
    if (CheckAnyIgnoredNodes(ignore_nodes, input_match, irrelevant_nodes)) {
      return true;
    }
  }
  return false;
}

void CheckSpecifiedNodeNames(
    const GraphDef& input_graph_def,
    std::unordered_map<string, QuantizeConfig> ops_to_quantize,
    const std::map<string, int>& nodes_to_bit) {
  std::unordered_set<string> node_names;
  for (const auto& node : input_graph_def.node()) {
    node_names.insert(node.name());
  }
  for (const auto& key_val : nodes_to_bit) {
    if (!node_names.count(key_val.first)) {
      DLOG_WARNING << key_val.first
                   << " not found in quantized graph, please specify node in "
                      "quantized graph";
      continue;
    }
    if (!ops_to_quantize.count(key_val.first)) {
      DLOG_WARNING << key_val.first
                   << " will not insert fix neuron, so this specification of "
                      "bit width will be ignored.";
    }
  }
  return;
}

Status _ConvertConstantsToVariables(const GraphDef& input_graph_def,
                                    GraphDef* output_graph_def) {
  std::map<string, string> inputs_to_rename;
  std::unordered_set<string> nodes_to_ignore;
  GraphDef replaced_graph_def;
  TF_RETURN_IF_ERROR(ReplaceMatchingOpTypes(
      input_graph_def,  // clang-format off
      {"Const", // add_node
      },  // clang-format on
      [&inputs_to_rename, &nodes_to_ignore](
          const NodeMatch& match, const std::set<string>& input_nodes,
          const std::set<string>& output_nodes,
          std::vector<NodeDef>* new_nodes) {
        // Find all the nodes we expect in the subgraph.
        const NodeDef& const_node = match.node;

        // Construct the new variable nodes.
        NodeDef variable_node;
        variable_node.set_name(const_node.name());
        variable_node.set_op("VariableV2");
        Tensor const_tensor = GetNodeTensorAttr(const_node, "value");
        TensorShape const_shape = const_tensor.shape();
        SetNodeAttr("shape", const_shape, &variable_node);
        CopyNodeAttr(const_node, "dtype", "dtype", &variable_node);

        NodeDef read_node;
        read_node.set_name(variable_node.name() + "/read");
        read_node.set_op("Identity");
        AddNodeInput(variable_node.name(), &read_node);
        CopyNodeAttr(const_node, "dtype", "T", &read_node);

        inputs_to_rename[const_node.name()] = read_node.name();
        nodes_to_ignore.insert(read_node.name());

        new_nodes->push_back(variable_node);
        new_nodes->push_back(read_node);
        return Status::OK();
      },
      {true}, &replaced_graph_def));
  TF_RETURN_IF_ERROR(RenameNodeInputs(replaced_graph_def, inputs_to_rename,
                                      nodes_to_ignore, output_graph_def));
  return Status::OK();
}

Status ParseGraph(
    const GraphDef& input_graph_def,
    std::vector<std::tuple<int, NodeMatch>>& matched_node_patterns,
    std::unordered_map<string, int>& matched_nodes,
    std::set<string>& ignore_nodes,
    std::unordered_set<string>& unmatched_nodes) {
  // parse ignore nodes
  for (auto pattern_id = 0; pattern_id < known_ignore_patterns.size();
       ++pattern_id) {
    auto pattern = known_ignore_patterns[pattern_id];
    auto pattern_name = get_ignore_pattern_name_from_id(pattern_id);
    DLOG_INFO(1) << "Parsing ignore pattern: " << pattern_name;

    GraphMatcher matcher(input_graph_def);
    std::vector<NodeMatch> matches;
    bool allow_intersection = true;
    TF_RETURN_IF_ERROR(matcher.GetOpTypeMatches(std::get<1>(pattern), &matches,
                                                allow_intersection));
    DLOG_INFO(2) << "  found number of ignore pattern " << pattern_name << ": "
                 << matches.size();
    for (const NodeMatch& match : matches) {
      std::vector<const NodeDef*> ignore_node_defs =
          get_ignore_nodes(match, pattern_name);
      std::set<string> input_node_names;
      for (int i = 0; i < ignore_node_defs.size(); i++) {
        DLOG_INFO(2) << "    found ignore nodes: "
                     << ignore_node_defs[i]->name();
        ignore_nodes.insert(ignore_node_defs[i]->name());
      }
    }
  }

  int match_id = 0;
  for (auto pattern_id = 0; pattern_id < known_patterns.size(); ++pattern_id) {
    auto pattern = known_patterns[pattern_id];
    auto pattern_name = get_pattern_name_from_id(pattern_id);
    DLOG_INFO(1) << "Parsing: " << pattern_name;

    GraphMatcher matcher(input_graph_def);
    std::vector<NodeMatch> matches;
    bool allow_intersection = true;
    TF_RETURN_IF_ERROR(matcher.GetOpTypeMatches(std::get<1>(pattern), &matches,
                                                allow_intersection));
    DLOG_INFO(2) << "  found number of pattern " << pattern_name << ": "
                 << matches.size();
    for (const NodeMatch& match : matches) {
      std::vector<const NodeDef*> input_nodes =
          get_input_nodes(match, pattern_name);
      std::set<string> input_node_names;
      for (int i = 0; i < input_nodes.size(); i++) {
        input_node_names.insert(input_nodes[i]->name());
      }
      DLOG_INFO(2) << "  found pattern: " << pattern_name;
      if (CheckAnyMatchedNodes(matched_nodes, match, input_node_names)) {
        continue;
      }
      DLOG_INFO(2) << "  record pattern: " << pattern_name;
      RecordMatchedPatterns(matched_node_patterns, pattern_id, match);
      RecordMatchedNodes(matched_nodes, match, match_id, input_node_names);
      match_id++;
    }
  }
  for (auto node : input_graph_def.node()) {
    if (!matched_nodes.count(node.name())) {
      unmatched_nodes.insert(node.name());
    }
  }
  return Status::OK();
}

Status InsertIdForNodes(const GraphDef& input_graph_def,
                        GraphDef* output_graph_def,
                        const std::vector<string>& node_names) {
  std::unordered_map<string, DataType> data_type_of_output_nodes;
  TF_RETURN_IF_ERROR(GetDataTypeOfNodes(input_graph_def, node_names,
                                        &data_type_of_output_nodes));

  output_graph_def->Clear();
  for (int i = 0; i < input_graph_def.node_size(); i++) {
    NodeDef cur_node = input_graph_def.node(i);
    auto node_name = cur_node.name();
    if (std::find(node_names.begin(), node_names.end(), node_name) !=
        node_names.end()) {
      cur_node.set_name(node_name + "_float");
      *(output_graph_def->mutable_node()->Add()) = cur_node;

      NodeDef id_node;
      id_node.set_name(node_name);
      id_node.set_op("Identity");
      AddNodeInput(cur_node.name(), &id_node);
      SetNodeAttr("T", data_type_of_output_nodes[node_name], &id_node);
      *(output_graph_def->mutable_node()->Add()) = id_node;
    } else {
      *(output_graph_def->mutable_node()->Add()) = cur_node;
    }
  }
  return Status::OK();
}

Status GetNodeDataType(const NodeDef& node, DataType* data_type) {
  if (node.op() == "Cast") {
    *data_type = node.attr().at("DstT").type();
  } else if (node.attr().count("dtype")) {
    *data_type = node.attr().at("dtype").type();
  } else if (node.attr().count("T")) {
    *data_type = node.attr().at("T").type();
  } else if (node.attr().count("type")) {
    *data_type = node.attr().at("type").type();
  } else {
    return errors::NotFound("Fail to get data type of node: \n",
                            node.DebugString());
  }
  return Status::OK();
}

Status GetDataTypeOfNodes(
    const GraphDef& input_graph_def, const std::vector<std::string> node_names,
    std::unordered_map<string, DataType>* data_type_of_nodes) {
  // Create empty feed_dict
  for (auto i = 0; i < node_names.size(); i++) {
    bool found = false;
    for (auto j = 0; j < input_graph_def.node_size(); j++) {
      const NodeDef& node = input_graph_def.node(j);
      if (node_names[i] == node.name()) {
        DataType data_type;
        TF_RETURN_IF_ERROR(GetNodeDataType(node, &data_type));
        (*data_type_of_nodes)[node_names[i]] = data_type;
        found = true;
      }
    }
    if (!found) {
      LOG(FATAL) << "Fail to find node " << node_names[i] << " in graph.";
    }
  }
  return Status::OK();
}

Status GetShapeOfNodes(
    const GraphDef& input_graph_def, const std::vector<std::string> node_names,
    const int batch_size,
    std::unordered_map<string, std::vector<int>>* shape_of_nodes) {
  std::unique_ptr<Session> sess(NewSession(SessionOptions()));
  TF_RETURN_IF_ERROR(sess->Create(input_graph_def));

  // Create empty feed_dict
  std::vector<std::pair<string, Tensor>> feed_dict;
  for (auto i = 0; i < input_graph_def.node_size(); i++) {
    const NodeDef& node = input_graph_def.node(i);
    if (node.op() == "Placeholder") {
      TensorShapeProto shape = node.attr().at("shape").shape();
      DataType data_type;
      TF_RETURN_IF_ERROR(GetNodeDataType(node, &data_type));
      if (shape.dim(0).size() == -1) {
        shape.mutable_dim(0)->set_size(batch_size);
      }
      feed_dict.emplace_back(node.name(), Tensor(data_type, shape));
    }
  }

  std::vector<Tensor> out_tensors;
  TF_RETURN_IF_ERROR(sess->Run({feed_dict}, node_names, {}, &out_tensors));
  for (auto i = 0; i < node_names.size(); i++) {
    std::vector<int> node_shape(out_tensors[i].dims());
    for (auto j = 0; j < out_tensors[i].dims(); j++) {
      if (out_tensors[i].dims() > 1 && j == 0) {
        node_shape[j] = batch_size;
      } else {
        node_shape[j] = out_tensors[i].dim_size(j);
      }
    }
    (*shape_of_nodes)[node_names[i]] = node_shape;
  }
  TF_RETURN_IF_ERROR(sess->Close());
  return Status::OK();
}

Status InferenceShape(const GraphDef& input_graph_def,
                      GraphDef* output_graph_def, const int batch_size) {
  std::set<string> op_types_with_shape({"Reshape", "ResizeNearestNeighbor"});
  std::vector<std::string> nodes_with_shape;
  for (auto i = 0; i < input_graph_def.node_size(); i++) {
    const NodeDef& node = input_graph_def.node(i);
    if (node.op() == "Reshape" || node.op() == "ResizeNearestNeighbor") {
      nodes_with_shape.push_back(node.name());
    }
  }
  if (nodes_with_shape.size() == 0) {
    *output_graph_def = input_graph_def;
    return Status::OK();
  }

  std::unordered_map<string, std::vector<int>> shape_of_nodes;
  TF_RETURN_IF_ERROR(GetShapeOfNodes(input_graph_def, nodes_with_shape,
                                     batch_size, &shape_of_nodes));

  TF_RETURN_IF_ERROR(ReplaceMatchingOpTypes(
      input_graph_def,  // clang-format off
      {"Reshape|ResizeNearestNeighbor",
        {
          {"*"},
          {"*"},
        }
      },  // clang-format on
      [&shape_of_nodes](const NodeMatch& match,
                        const std::set<string>& input_nodes,
                        const std::set<string>& output_nodes,
                        std::vector<NodeDef>* new_nodes) {
        const NodeDef& node_with_shape = match.node;
        const NodeDef& input_node = match.inputs[0].node;
        new_nodes->push_back(input_node);

        NodeDef shape_node;
        shape_node.set_op("Const");
        shape_node.set_name(node_with_shape.name() + "/shape");
        SetNodeAttr("dtype", DT_INT32, &shape_node);

        std::vector<int> node_shape = shape_of_nodes[node_with_shape.name()];
        Tensor shape_tensor(DT_INT32, {(int)node_shape.size()});
        if (node_with_shape.op() == "Reshape") {
          for (auto j = 0; j < node_shape.size(); j++) {
            shape_tensor.flat<int>()(j) = node_shape[j];
          }
        } else if (node_with_shape.op() == "ResizeNearestNeighbor") {
          shape_tensor.flat<int>()(0) = node_shape[1];
          shape_tensor.flat<int>()(1) = node_shape[2];
        }
        SetNodeTensorAttr<int>("value", shape_tensor, &shape_node);

        new_nodes->push_back(shape_node);

        NodeDef new_node_with_shape = node_with_shape;
        new_node_with_shape.mutable_input()->Clear();
        AddNodeInput(node_with_shape.input(0), &new_node_with_shape);
        AddNodeInput(shape_node.name(), &new_node_with_shape);
        new_nodes->push_back(new_node_with_shape);
        return Status::OK();
      },
      {}, output_graph_def));
  return Status::OK();
}

Status ConvertMeanToAvgpool(const GraphDef& input_graph_def,
                            GraphDef* output_graph_def) {
  GraphDef current_graph_def, processed_graph_def;
  std::map<string, string> inputs_to_rename;
  std::unordered_set<string> nodes_to_ignore;

  std::vector<std::string> mean_nodes;
  for (auto i = 0; i < input_graph_def.node_size(); i++) {
    const NodeDef& node = input_graph_def.node(i);
    if (node.op() == "Mean") {
      mean_nodes.push_back(node.name());
      mean_nodes.push_back(node.input(0));
    }
  }
  if (mean_nodes.size() == 0) {
    *output_graph_def = input_graph_def;
    return Status::OK();
  }

  std::unordered_map<string, std::vector<int>> shape_of_means;
  TF_RETURN_IF_ERROR(
      GetShapeOfNodes(input_graph_def, mean_nodes, 1, &shape_of_means));

  current_graph_def = input_graph_def;
  TF_RETURN_IF_ERROR(ReplaceMatchingOpTypes(
      current_graph_def,  // clang-format off
      {"Mean",
        {
          {"*"}, // input node
          {"Const"}, // reduction_indices node
        }
      },  // clang-format on
      [&inputs_to_rename, &nodes_to_ignore, &shape_of_means](
          const NodeMatch& match, const std::set<string>& input_nodes,
          const std::set<string>& output_nodes,
          std::vector<NodeDef>* new_nodes) {
        const NodeDef& mean_node = match.node;
        const NodeDef& input_node = match.inputs[0].node;
        const NodeDef& ri_node = match.inputs[1].node;
        new_nodes->push_back(input_node);

        bool keep_dims = mean_node.attr().at("keep_dims").b();
        Tensor ri_tensor = GetNodeTensorAttr(ri_node, "value");
        if (keep_dims && ri_tensor.flat<int>()(0) == 1 &&
            ri_tensor.flat<int>()(1) == 2) {
          DLOG_WARNING << "Convert mean node " << mean_node.name()
                       << " to AvgPool";
          int kernel_h = shape_of_means.at(input_node.name())[1] /
                         shape_of_means.at(mean_node.name())[1];
          int kernel_w = shape_of_means.at(input_node.name())[2] /
                         shape_of_means.at(mean_node.name())[2];

          // Build avgpool node
          NodeDef avgpool_node;
          avgpool_node.set_name(mean_node.name());
          avgpool_node.set_op("AvgPool");
          AddNodeInput(input_node.name(), &avgpool_node);
          SetNodeAttr("ksize", std::vector<int>({1, kernel_h, kernel_w, 1}),
                      &avgpool_node);
          SetNodeAttr("padding", "VALID", &avgpool_node);
          SetNodeAttr("T", DT_FLOAT, &avgpool_node);
          SetNodeAttr("strides", std::vector<int>({1, 1, 1, 1}), &avgpool_node);
          SetNodeAttr("data_format", "NHWC", &avgpool_node);

          new_nodes->push_back(avgpool_node);
        } else {
          new_nodes->push_back(ri_node);
          new_nodes->push_back(mean_node);
        }
        return Status::OK();
      },
      {}, &processed_graph_def));
  TF_RETURN_IF_ERROR(RenameNodeInputs(processed_graph_def, inputs_to_rename,
                                      nodes_to_ignore, output_graph_def));
  return Status::OK();
}

Status SimulateDPU(const GraphDef& input_graph_def,
                   GraphDef* output_graph_def) {
  GraphDef current_graph_def, processed_graph_def;
  std::map<string, string> inputs_to_rename;
  std::unordered_set<string> nodes_to_ignore;

  // Add scale for avgpool with specific kernel_sizes, to simulate the behaviour
  // on DPU which uses bit-shift to do dividing.
  current_graph_def = input_graph_def;
  TF_RETURN_IF_ERROR(
      ConvertMeanToAvgpool(current_graph_def, &processed_graph_def));
  current_graph_def = processed_graph_def;
  TF_RETURN_IF_ERROR(ReplaceMatchingOpTypes(
      current_graph_def,  // clang-format off
      {"AvgPool",
        {
          {"*"},
        }
      },  // clang-format on
      [&inputs_to_rename, &nodes_to_ignore](
          const NodeMatch& match, const std::set<string>& input_nodes,
          const std::set<string>& output_nodes,
          std::vector<NodeDef>* new_nodes) {
        const NodeDef& avgpool_node = match.node;
        const NodeDef& input_node = match.inputs[0].node;
        new_nodes->push_back(input_node);
        new_nodes->push_back(avgpool_node);

        if (avgpool_node.attr().count("data_format")) {
          CHECK(avgpool_node.attr().at("data_format").s() == "NHWC")
              << " Only NHWC format is supported.";
        }

        int kernel_h = avgpool_node.attr().at("ksize").list().i(1);
        int kernel_w = avgpool_node.attr().at("ksize").list().i(2);

        float scale_factor = 1.0;
        bool need_scale = false;
        if (kernel_h == 3 && kernel_w == 3) {
          scale_factor = 9.0 * 7.f / 64.f;
          need_scale = true;
        } else if (kernel_h == 5 && kernel_w == 5) {
          scale_factor = 25.0 * 10.f / 256.f;
          need_scale = true;
        } else if (kernel_h == 6 && kernel_w == 6) {
          scale_factor = 36.0 * 7.f / 256.f;
          need_scale = true;
        } else if (kernel_h == 7 && kernel_w == 7) {
          scale_factor = 49.0 * 21.f / 1024.f;
          need_scale = true;
        } else if (kernel_h == 14 && kernel_w == 14) {
          scale_factor = 196.0 * 21.f / 4096.f;
          need_scale = true;
        }

        if (need_scale) {
          DLOG_WARNING << "Scale output of avg_pool node "
                       << avgpool_node.name() << " to simulate DPU.";

          // Construct the new nodes.
          NodeDef scale_value_node;
          scale_value_node.set_op("Const");
          scale_value_node.set_name(avgpool_node.name() + "/scale_value");
          SetNodeAttr("dtype", DT_FLOAT, &scale_value_node);

          Tensor scale_tensor(DT_FLOAT, {});
          scale_tensor.flat<float>()(0) = scale_factor;
          SetNodeTensorAttr<float>("value", scale_tensor, &scale_value_node);
          new_nodes->push_back(scale_value_node);

          NodeDef mul_node;
          mul_node.set_name(avgpool_node.name() + "/mul");
          mul_node.set_op("Mul");
          SetNodeAttr("T", DT_FLOAT, &mul_node);
          AddNodeInput(avgpool_node.name(), &mul_node);
          AddNodeInput(scale_value_node.name(), &mul_node);
          new_nodes->push_back(mul_node);

          inputs_to_rename[avgpool_node.name()] = mul_node.name();
          nodes_to_ignore.insert(mul_node.name());
        }
        return Status::OK();
      },
      {}, &processed_graph_def));
  TF_RETURN_IF_ERROR(RenameNodeInputs(processed_graph_def, inputs_to_rename,
                                      nodes_to_ignore, &current_graph_def));

  // For leaky_relu, DPU only support alpha=0.1 and will convert 0.1 to 26/256
  // on DPU which uses bit-shift to do dividing.
  inputs_to_rename.clear();
  nodes_to_ignore.clear();
  TF_RETURN_IF_ERROR(ReplaceMatchingOpTypes(
      current_graph_def,  // clang-format off
      {"Maximum",
        {
          {"Mul",
            {
              {"Const"}, // alpha node
              {"*"}, // input node
            }
          },
          {"*"}, // input node
        }
      },  // clang-format on
      [&inputs_to_rename, &nodes_to_ignore](
          const NodeMatch& match, const std::set<string>& input_nodes,
          const std::set<string>& output_nodes,
          std::vector<NodeDef>* new_nodes) {
        const NodeDef& maximum_node = match.node;
        const NodeDef& mul_node = match.inputs[0].node;
        const NodeDef& alpha_node = match.inputs[0].inputs[0].node;
        const NodeDef& input_node = match.inputs[1].node;
        new_nodes->push_back(input_node);

        Tensor alpha_tensor = GetNodeTensorAttr(alpha_node, "value");
        float alpha = alpha_tensor.flat<float>()(0);

        if (fabs(alpha - 0.1) < 0.00001) {
          DLOG_WARNING << "Convert alpha of leaky_relu node "
                       << maximum_node.name()
                       << " from 0.1 to 26/256 to simulate DPU.";

          // Construct the new nodes.
          NodeDef new_alpha_node = alpha_node;
          Tensor new_alpha_tensor(DT_FLOAT, {});
          new_alpha_tensor.flat<float>()(0) = 26. / 256;
          SetNodeTensorAttr<float>("value", new_alpha_tensor, &new_alpha_node);
          new_nodes->push_back(new_alpha_node);

        } else {
          new_nodes->push_back(alpha_node);
        }

        new_nodes->push_back(mul_node);
        new_nodes->push_back(maximum_node);
        return Status::OK();
      },
      {}, &processed_graph_def));
  TF_RETURN_IF_ERROR(RenameNodeInputs(processed_graph_def, inputs_to_rename,
                                      nodes_to_ignore, &current_graph_def));

  // Keras LeakyReLU
  inputs_to_rename.clear();
  nodes_to_ignore.clear();
  TF_RETURN_IF_ERROR(ReplaceMatchingOpTypes(
      current_graph_def,  // clang-format off
      {"Sub",
        {
          {"Relu"},
          {"Mul",
            {
              {"Const"}, // alpha node
              {"Relu",
                {
                  {"Neg",
                    {
                      {"*"}, // input node
                    }
                  },
                }
              },
            }
          },
        }
      },  // clang-format on
      [&inputs_to_rename, &nodes_to_ignore](
          const NodeMatch& match, const std::set<string>& input_nodes,
          const std::set<string>& output_nodes,
          std::vector<NodeDef>* new_nodes) {
        const NodeDef& sub_node = match.node;
        const NodeDef& relu2_node = match.inputs[0].node;
        const NodeDef& mul_node = match.inputs[1].node;
        const NodeDef& alpha_node = match.inputs[1].inputs[0].node;
        const NodeDef& relu1_node = match.inputs[1].inputs[1].node;
        const NodeDef& neg_node = match.inputs[1].inputs[1].inputs[0].node;
        const NodeDef& input_node =
            match.inputs[1].inputs[1].inputs[0].inputs[0].node;
        new_nodes->push_back(input_node);
        new_nodes->push_back(neg_node);
        new_nodes->push_back(relu1_node);

        Tensor alpha_tensor = GetNodeTensorAttr(alpha_node, "value");
        float alpha = alpha_tensor.flat<float>()(0);

        if (fabs(alpha - 0.1) < 0.00001) {
          DLOG_WARNING << "Convert alpha of leaky_relu node " << sub_node.name()
                       << " from 0.1 to 26/256 to simulate DPU.";

          // Construct the new nodes.
          NodeDef new_alpha_node = alpha_node;
          Tensor new_alpha_tensor(DT_FLOAT, {});
          new_alpha_tensor.flat<float>()(0) = 26. / 256;
          SetNodeTensorAttr<float>("value", new_alpha_tensor, &new_alpha_node);
          new_nodes->push_back(new_alpha_node);

        } else {
          new_nodes->push_back(alpha_node);
        }

        new_nodes->push_back(mul_node);
        new_nodes->push_back(relu2_node);
        new_nodes->push_back(sub_node);
        return Status::OK();
      },
      {}, &processed_graph_def));
  TF_RETURN_IF_ERROR(RenameNodeInputs(processed_graph_def, inputs_to_rename,
                                      nodes_to_ignore, &current_graph_def));

  // Fuse LeakyRelu
  inputs_to_rename.clear();
  nodes_to_ignore.clear();
  TF_RETURN_IF_ERROR(ReplaceMatchingOpTypes(
      current_graph_def,  // clang-format off
      {"LeakyRelu",
        {
          {"*"}, // input node
        }
      },  // clang-format on
      [&inputs_to_rename, &nodes_to_ignore](
          const NodeMatch& match, const std::set<string>& input_nodes,
          const std::set<string>& output_nodes,
          std::vector<NodeDef>* new_nodes) {
        const NodeDef& leakyrelu_node = match.node;
        const NodeDef& input_node = match.inputs[0].node;

        new_nodes->push_back(input_node);

        float alpha = 0;
        GetNodeAttr(leakyrelu_node, "alpha", &alpha);

        if (fabs(alpha - 0.1) < 0.00001) {
          DLOG_WARNING << "Convert alpha of leaky_relu node "
                       << leakyrelu_node.name()
                       << " from 0.1 to 26/256 to simulate DPU.";

          // Construct the new nodes.
          NodeDef new_leakyrelu_node = leakyrelu_node;
          float new_alpha = 26.0 / 256.0;
          SetNodeAttr("alpha", new_alpha, &new_leakyrelu_node);
          new_nodes->push_back(new_leakyrelu_node);
        } else {
          new_nodes->push_back(leakyrelu_node);
        }

        return Status::OK();
      },
      {}, &processed_graph_def));
  TF_RETURN_IF_ERROR(RenameNodeInputs(processed_graph_def, inputs_to_rename,
                                      nodes_to_ignore, &current_graph_def));

  *output_graph_def = current_graph_def;
  return Status::OK();
}

Status SaveGraphForDebugging(const GraphDef& graph_def, const string& file_name,
                             const string& output_dir) {
  if (getenv("DECENT_DEBUG") && std::atoi(getenv("DECENT_DEBUG")) >= 1) {
    const string debug_dir = io::JoinPath(output_dir, "decent_debug");
    string make_debug_dir = "mkdir -p " + debug_dir;
    if (system(make_debug_dir.c_str()) != 0) {
      LOG(FATAL) << "Fail to mkdir " + debug_dir;
    }
    string file_path = io::JoinPath(debug_dir, file_name);
    std::ofstream output(file_path);
    graph_def.SerializeToOstream(&output);
    output.close();
  }
  return Status::OK();
}

Status SaveQuantizeInfoForDebugging(const GraphDef& input_graph_def,
                                    const string& output_dir) {
  if (getenv("DECENT_DEBUG") && std::atoi(getenv("DECENT_DEBUG")) >= 1) {
    const string debug_dir = io::JoinPath(output_dir, "decent_debug");
    string make_debug_dir = "mkdir -p " + debug_dir;
    if (system(make_debug_dir.c_str()) != 0) {
      LOG(FATAL) << "Fail to mkdir " + debug_dir;
    }
    string filename = io::JoinPath(debug_dir, "quantize_info");
    std::ofstream ofile(filename);
    if (!ofile.is_open()) {
      LOG(FATAL) << "Cannot open file: " << filename;
    }
    for (auto i = 0; i < input_graph_def.node_size(); i++) {
      auto node = input_graph_def.node(i);
      ofile << node.name() << " ";
      for (auto key : {"ipos", "opos", "wpos", "bpos"}) {
        if (node.attr().count(key)) {
          for (auto n = 0; n < node.attr().at(key).list().i_size(); n++) {
            ofile << node.attr().at(key).list().i(n) << " ";
          }
        }
      }
      ofile << std::endl;
    }
    DLOG_INFO(1) << "quantize_info saved to file: " << filename;
  }
  return Status::OK();
}

Status LoadQuantizeInfoFromFile(
    const GraphDef& input_graph_def,
    std::unordered_map<string, std::vector<int>>* quantize_info_map,
    const string& output_dir) {
  for (const NodeDef& node : input_graph_def.node()) {
    if (node.op() == "FixNeuron") {
      string filename;
      filename = output_dir + "/temp/" +
                 str_util::StringReplace(node.name(), "/", "_", true);
      std::ifstream ifile(filename);
      if (!ifile.is_open()) {
        DLOG_WARNING << "Cannot find quantize info file: " << filename
                     << ". Use default quantize info.";
        (*quantize_info_map)[node.name()] = {8, 0};
        continue;
      }
      string op_name;
      int bit_width, pos;
      ifile >> op_name >> bit_width >> pos;
      (*quantize_info_map)[op_name] = {bit_width, pos};
      DLOG_INFO(2) << "quantize_info_map[" << op_name << "]: " << bit_width
                   << " " << pos;
      ifile.close();
      // remove(filename.c_str());
    }
  }
  return Status::OK();
}

Status SaveNodeGroupsToFile(const std::set<NodeGroup>& node_groups,
                            const string& output_dir) {
  string filename;
  filename = output_dir + "/temp/node_groups";

  std::ofstream ofile(filename);
  if (!ofile.is_open()) {
    DLOG_WARNING << "Cannot find node groups file: " << filename;
  }

  for (auto g : node_groups) {
    ofile << g[0] << " " << g[1] << " " << g[2] << " " << g[3] << " " << g[4]
          << std::endl;
  }

  ofile.close();
  return Status::OK();
}

Status LoadNodeGroupsFromFile(std::set<NodeGroup>& node_groups,
                              const string& output_dir) {
  string filename;
  filename = output_dir + "/temp/node_groups";
  std::ifstream ifile(filename);
  if (!ifile.is_open()) {
    DLOG_WARNING << "Cannot find node groups file: " << filename;
  }

  std::vector<string> node_names(5);
  while (ifile >> node_names[0] >> node_names[1] >> node_names[2] >>
         node_names[3] >> node_names[4]) {
    node_groups.insert(node_names);
  }

  ifile.close();
  return Status::OK();
}

}  // namespace decent_q
}  // namespace tensorflow
