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

#include "cross_layers_equalization.h"
#include "quantize_utils.h"
#include "tensorflow/core/platform/str_util.h"

namespace tensorflow {
namespace decent_q {

namespace {

void CheckInputNode(
    const std::map<string, std::vector<const NodeDef *>> &output_node_map,
    const NodeDef *cur_node, std::vector<string> &node_to_be_checked,
    bool &consecutive) {
  if (cur_node->input_size() > 0) {
    // check stoppable node, add all input node into node_to_be_checked
    if (!consecutive) {
      for (int i = 0; i < cur_node->input_size(); ++i) {
        const string input_name = GetRealName(cur_node->input(i));
        if (std::find(node_to_be_checked.begin(), node_to_be_checked.end(),
                      input_name) == node_to_be_checked.end()) {
          DLOG_INFO(2) << "Check stoppable node: " << cur_node->name()
                       << ", input node: " << input_name;
          node_to_be_checked.emplace(node_to_be_checked.begin(), input_name);
        }
      }
      return;
    }
    // check conv and white list node, which only have one put, add the input
    // into node_to_be_checked
    const string input_name = GetRealName(cur_node->input(0));
    if (std::find(node_to_be_checked.begin(), node_to_be_checked.end(),
                  input_name) == node_to_be_checked.end()) {
      DLOG_INFO(2) << "Check node: " << cur_node->name()
                   << ", input node: " << input_name;
      node_to_be_checked.push_back(input_name);
      // check if have multi-output
      if (!output_node_map.count(input_name) ||
          output_node_map.at(input_name).size() > 1) {
        consecutive = false;
        DLOG_INFO(2) << "Consecutive node: " << cur_node->name()
                     << " is stoppable";
      }
    } else {
      DLOG_INFO(2) << "Consecutive node: " << cur_node->name()
                   << " is stoppable";
      consecutive = false;
    }
    return;
  }
  return;
}
} // namespace

string GetRealName(const string &node_name) {
  std::vector<string> split_input_names = str_util::Split(node_name, ":");
  return split_input_names[0];
}

void GetHeadWeights(const Tensor &head_w, const Tensor &head_b,
                    const string &head_w_op, const bool &head_has_bias,
                    Eigen::Tensor<float, 2, Eigen::RowMajor> &head_weights) {
  // conv2d: final_w = concatate(w.reshape(-1, k_out), b, axis=0)
  // Depthwise : final_w = concatate(w.reshape(-1, k_in* k_out), b, axis=0)
  const int k_h = head_w.shape().dim_size(0);
  const int k_w = head_w.shape().dim_size(1);
  const int k_in = head_w.shape().dim_size(2);
  const int k_out = head_w.shape().dim_size(3);
  if (head_w_op == "Conv2D") {
    Eigen::Tensor<float, 2, Eigen::RowMajor> head_w_flatten =
        head_w.flat_inner_dims<float>();
    if (head_has_bias) {
      Eigen::Tensor<float, 2, Eigen::RowMajor> head_b_flatten =
          head_b.shaped<float, 2>({1, k_out});
      head_weights = head_w_flatten.concatenate(head_b_flatten, 0);
    } else {
      head_weights = head_w_flatten;
    }
  } else if (head_w_op == "DepthwiseConv2dNative") {
    Eigen::Tensor<float, 4, Eigen::RowMajor> head_w_tensor =
        head_w.tensor<float, 4>();
    Eigen::array<int, 2> two_dims{{k_h * k_w, k_in * k_out}};
    Eigen::Tensor<float, 2, Eigen::RowMajor> head_w_flatten =
        head_w_tensor.reshape(two_dims);
    if (head_has_bias) {
      Eigen::Tensor<float, 2, Eigen::RowMajor> head_b_flatten =
          head_b.shaped<float, 2>({1, k_in * k_out});
      head_weights = head_w_flatten.concatenate(head_b_flatten, 0);
    } else {
      head_weights = head_w_flatten;
    }
  }
  return;
}

void GetTailWeights(const Tensor &tail_w, const string &tail_w_op,
                    Eigen::Tensor<float, 2, Eigen::RowMajor> &tail_weights) {
  // final_w = w.transpose((0, 1, 3, 2)).reshape(-1, k_in)
  // for some DepthwiseConv2dNative, the k_out > 1, so need to transpose
  const int k_h = tail_w.shape().dim_size(0);
  const int k_w = tail_w.shape().dim_size(1);
  const int k_in = tail_w.shape().dim_size(2);
  const int k_out = tail_w.shape().dim_size(3);
  Eigen::Tensor<float, 4, Eigen::RowMajor> tail_w_tensor =
      tail_w.tensor<float, 4>();
  Eigen::array<int, 4> shuffling({0, 1, 3, 2});
  Eigen::Tensor<float, 4, Eigen::RowMajor> tail_w_shuffle =
      tail_w_tensor.shuffle(shuffling);
  Eigen::array<int, 2> two_dims{{k_h * k_w * k_out, k_in}};
  Eigen::Tensor<float, 2, Eigen::RowMajor> tail_w_flatten =
      tail_w_shuffle.reshape(two_dims);
  tail_weights = tail_w_flatten;
  return;
}

const std::set<string> cle_conv_op{"Conv2D", "DepthwiseConv2dNative"};
const std::set<string> cle_white_list_op{"Relu6",   "Relu", "AvgPool",
                                         "MaxPool", "Mul",  "Pad"};

Status ReplaceRelu6WithRelu(const GraphDef &input_graph_def,
                            GraphDef *output_graph_def) {
  output_graph_def->Clear();
  for (int i = 0; i < input_graph_def.node_size(); ++i) {
    NodeDef cur_node = input_graph_def.node(i);
    string op_type = cur_node.op();
    if (op_type == "Relu6") {
      cur_node.set_op("Relu");
      string node_name = cur_node.name();
      *(output_graph_def->mutable_node()->Add()) = cur_node;
    } else {
      *(output_graph_def->mutable_node()->Add()) = cur_node;
    }
  }
  return Status::OK();
}

void MapNamesToOutputNodesName(
    const GraphDef &graph_def,
    std::map<string, std::vector<const NodeDef *>> &output_node_map) {
  for (const NodeDef &node : graph_def.node()) {
    for (int i = 0; i < node.input_size(); ++i) {
      const string input_name = GetRealName(node.input(i));
      output_node_map[input_name].push_back(&node);
    }
  }
}

Status ParseConvPairs(const GraphDef &input_graph_def,
                      std::vector<std::vector<ConvBiasPair>> &conv_group,
                      const std::vector<string> &input_nodes,
                      const std::vector<string> &output_nodes) {
  std::map<string, const NodeDef *> node_map;
  MapNamesToNodes(input_graph_def, &node_map);
  std::map<string, std::vector<const NodeDef *>> output_node_map;
  MapNamesToOutputNodesName(input_graph_def, output_node_map);
  std::set<string> visited_node;
  std::vector<string> node_to_be_checked;
  for (auto i = 0; i < output_nodes.size(); ++i) {
    node_to_be_checked.emplace_back(output_nodes[i]);
  }
  std::vector<ConvBiasPair> consecutive_conv;
  bool consecutive = true;
  while (!node_to_be_checked.empty()) {
    const string cur_node_name = node_to_be_checked.back();
    const NodeDef *cur_node = node_map[cur_node_name];
    node_to_be_checked.pop_back();
    string op_type = cur_node->op();
    string op_name = cur_node->name();
    DLOG_INFO(3) << "Parsing node: " << op_name << " , type: " << op_type;
    bool is_input_node = std::find(input_nodes.begin(), input_nodes.end(),
                                   op_name) != input_nodes.end();
    if (is_input_node || visited_node.count(op_name)) {
      consecutive = false;
    }
    // 1 non-conv node will be skip and push input node into vector
    // 2 conv+bias and conv will be appended to consecutive_conv
    // 3 other op will stop appending conv into consecutive conv, and push the
    // consecutive_conv into conv_group
    if (!visited_node.count(op_name)) {
      if (cle_white_list_op.count(op_type)) {
        visited_node.insert(op_name);
        CheckInputNode(output_node_map, cur_node, node_to_be_checked,
                       consecutive);
        // DLOG_INFO(1) << "Parsing consecutive node " << op_name;
      } else if (op_type == "BiasAdd" || op_type == "Add" ||
                 op_type == "AddV2") {
        const NodeDef *conv_node = node_map[cur_node->input(0)];
        string input_op_type = conv_node->op();
        if (input_op_type == "Conv2D" ||
            input_op_type == "DepthwiseConv2dNative") {
          // DLOG_INFO(1) << "Parsing conv bias node:  " << conv_node->name();
          consecutive_conv.emplace_back(std::make_pair(conv_node, cur_node));
          visited_node.insert(op_name);
          visited_node.insert(conv_node->name());
          CheckInputNode(output_node_map, conv_node, node_to_be_checked,
                         consecutive);
        } else {
          // add op is not biasadd
          visited_node.insert(op_name);
          // DLOG_INFO(1) << "Parsing stoppable node: " << op_name;
          consecutive = false;
          CheckInputNode(output_node_map, cur_node, node_to_be_checked,
                         consecutive);
          // change consecutive after CheckInputNode
        }
      } else if (op_type == "Conv2D" || op_type == "DepthwiseConv2dNative") {
        if (!visited_node.count(op_name)) {
          // DLOG_INFO(1) << "Parsing conv node: " << op_name;
          consecutive_conv.emplace_back(std::make_pair(cur_node, nullptr));
          visited_node.insert(op_name);
          CheckInputNode(output_node_map, cur_node, node_to_be_checked,
                         consecutive);
        }
      } else {
        // other op will break Consecutive conv
        visited_node.insert(op_name);
        consecutive = false;
        // DLOG_INFO(1) << "Parsing stoppable node: " << op_name;
        CheckInputNode(output_node_map, cur_node, node_to_be_checked,
                       consecutive);
      }
    }

    if (!consecutive) {
      if (consecutive_conv.size() > 1) {
        DLOG_INFO(2) << "Consecutive conv group end with " << op_name;
        std::reverse(consecutive_conv.begin(), consecutive_conv.end());
        conv_group.push_back(consecutive_conv);
      }
      consecutive_conv.clear();
      consecutive = true;
    }
  }
  std::reverse(conv_group.begin(), conv_group.end());
  DLOG_INFO(1) << "Found " << conv_group.size() << " consecutive conv group";
  for (auto i = 0; i < conv_group.size(); ++i) {
    auto consecutive_conv = conv_group[i];
    for (auto j = 0; j < consecutive_conv.size(); ++j) {
      DLOG_INFO(1) << "conv_group index: " << i
                   << " consecutive_conv idex : " << j
                   << " conv name : " << consecutive_conv[j].first->name();
    }
  }
  return Status::OK();
}

Status EqualizeConvPair(const GraphDef &input_graph_def,
                        GraphDef &output_graph_def, const ConvBiasPair &head,
                        const ConvBiasPair &tail, const string &method,
                        const float &threshold) {
  if (input_graph_def.node_size() == 0) {
    DLOG_WARNING << "Empty input_graph_def, please check input_graph_def";
  }
  const string head_conv_name = head.first->name();
  const string tail_conv_name = tail.first->name();
  const string head_w_name = head.first->input(1);
  const string tail_w_name = tail.first->input(1);
  const string head_w_op = head.first->op();
  const string tail_w_op = tail.first->op();

  NodeDef head_w_node, head_b_node, tail_w_node, tail_b_node;
  Tensor head_w, head_b, tail_w;
  std::map<string, const NodeDef *> node_map;
  MapNamesToNodes(input_graph_def, &node_map);
  head_w = GetNodeTensorAttr(*(node_map[head_w_name]), "value");
  tail_w = GetNodeTensorAttr(*(node_map[tail_w_name]), "value");
  const bool head_has_bias = (head.second != nullptr);
  string head_b_name;
  if (head_has_bias) {
    head_b_name = head.second->input(1);
    head_b = GetNodeTensorAttr(*(node_map[head_b_name]), "value");
  }

  // reshape head w,b
  const int k_in = head_w.shape().dim_size(2);
  const int k_out = head_w.shape().dim_size(3);
  const int scale_dim =
      head_w_op == "DepthwiseConv2dNative" ? k_in * k_out : k_out;
  CHECK_EQ(scale_dim, tail_w.shape().dim_size(2))
      << " head conv node is " << head_conv_name << ", tail conv node is "
      << tail_conv_name;
  Eigen::Tensor<float, 2, Eigen::RowMajor> head_weights;
  GetHeadWeights(head_w, head_b, head_w_op, head_has_bias, head_weights);
  Eigen::Tensor<float, 2, Eigen::RowMajor> tail_weights;
  GetTailWeights(tail_w, tail_w_op, tail_weights);

  // Calculate scale
  Eigen::Tensor<float, 1, Eigen::RowMajor> head_max(scale_dim);
  Eigen::Tensor<float, 1, Eigen::RowMajor> tail_max(scale_dim);
  Eigen::Tensor<float, 1, Eigen::RowMajor> scale(scale_dim);
  Eigen::array<int, 1> dims({0});
  head_max = head_weights.abs().maximum(dims);
  tail_max = tail_weights.abs().maximum(dims);
  if (method == "max") {
    scale = (tail_max / head_max).sqrt();
  } else if (method == "mean") {
    scale =
        (tail_weights.abs().mean(dims) / head_weights.abs().mean(dims)).sqrt();
  } else {
    DLOG_WARNING << "unknown equlization method " << method;
  }
  // filter scale
  // np.clip(scale, 0.01, 100)
  for (int i = 0; i < scale_dim; ++i) {
    scale(i) = scale(i) < 0.01 ? 0.01 : scale(i);
    scale(i) = scale(i) > 100 ? 100 : scale(i);
  }
  // scale[(head_max+tail_max) < threshold] = 1
  for (int i = 0; i < scale_dim; ++i) {
    scale(i) = head_max(i) + tail_max(i) < threshold ? 1 : scale(i);
  }

  // implement scale to weights
  output_graph_def.Clear();
  for (int i = 0; i < input_graph_def.node_size(); ++i) {
    NodeDef cur_node = input_graph_def.node(i);
    const string op_name = cur_node.name();
    const string op_type = cur_node.op();
    if (op_name == head_w_name) {
      if (head_w_op == "Conv2D") {
        Tensor weights = GetNodeTensorAttr(cur_node, "value");
        auto weights_flatten = weights.flat_inner_dims<float>();
        CHECK_EQ(weights_flatten.dimension(1), scale.dimension(0))
            << " head conv node is " << head_conv_name << ", tail conv node is "
            << tail_conv_name;
        for (int i = 0; i < weights_flatten.dimension(0); ++i) {
          for (int j = 0; j < weights_flatten.dimension(1); ++j) {
            weights_flatten(i, j) *= scale(j);
          }
        }
        DLOG_INFO(3) << "Do CLE for head Conv2D node " << op_name
                     << "  op_type: " << op_type;
        SetNodeTensorAttr<float>("value", weights, &cur_node);
        *(output_graph_def.mutable_node()->Add()) = cur_node;
      } else if (head_w_op == "DepthwiseConv2dNative") {
        Tensor weights = GetNodeTensorAttr(cur_node, "value");
        auto weights_tensor = weights.tensor<float, 4>();
        const int k_h = weights_tensor.dimension(0);
        const int k_w = weights_tensor.dimension(1);
        const int k_in = weights_tensor.dimension(2);
        const int k_out = weights_tensor.dimension(3);
        // channle of DepthwiseConv2dNative output is k_in * k_out
        CHECK_EQ(k_in * k_out, scale.dimension(0))
            << " head conv node is " << head_conv_name << ", tail conv node is "
            << tail_conv_name;
        for (int i = 0; i < k_h; ++i) {
          for (int j = 0; j < k_w; ++j) {
            for (int k = 0; k < k_in; ++k) {
              for (int l = 0; l < k_out; ++l) {
                weights_tensor(i, j, k, l) *= scale(k * k_out + l);
              }
            }
          }
        }
        DLOG_INFO(3) << "Do CLE for head "
                        "DepthwiseConv2dNative node "
                     << op_name << "  op_type: " << op_type;
        SetNodeTensorAttr<float>("value", weights, &cur_node);
        *(output_graph_def.mutable_node()->Add()) = cur_node;
      }
      // calculate head bias
    } else if (head_has_bias && op_name == head_b_name) {
      Tensor bias = GetNodeTensorAttr(cur_node, "value");
      auto bias_flatten = bias.flat<float>();
      CHECK_EQ(bias.NumElements(), scale.dimension(0))
          << " head conv node is " << head_conv_name << ", tail conv node is "
          << tail_conv_name;
      for (int col = 0; col < bias.NumElements(); ++col) {
        bias_flatten(col) *= scale(col);
      }
      DLOG_INFO(3) << "Do CLE for head bias add node " << op_name
                   << "  op_type: " << op_type;
      SetNodeTensorAttr<float>("value", bias, &cur_node);
      *(output_graph_def.mutable_node()->Add()) = cur_node;
      // calculate tail W : w /= sacle
    } else if (op_name == tail_w_name) {
      Tensor weights = GetNodeTensorAttr(cur_node, "value");
      auto weights_tensor = weights.tensor<float, 4>();
      CHECK_EQ(weights_tensor.dimension(2), scale.dimension(0))
          << " head conv node is " << head_conv_name << ", tail conv node is "
          << tail_conv_name;
      for (int i = 0; i < weights_tensor.dimension(0); ++i) {
        for (int j = 0; j < weights_tensor.dimension(1); ++j) {
          for (int k = 0; k < weights_tensor.dimension(2); ++k) {
            for (int l = 0; l < weights_tensor.dimension(3); ++l) {
              weights_tensor(i, j, k, l) /= scale(k);
            }
          }
        }
      }
      DLOG_INFO(3) << "Do CLE for tail conv node " << op_name
                   << "  op_type: " << op_type;
      SetNodeTensorAttr<float>("value", weights, &cur_node);
      *(output_graph_def.mutable_node()->Add()) = cur_node;
    } else {
      *(output_graph_def.mutable_node()->Add()) = cur_node;
    }
  }
  return Status::OK();
}
} // namespace decent_q
} // namespace tensorflow
