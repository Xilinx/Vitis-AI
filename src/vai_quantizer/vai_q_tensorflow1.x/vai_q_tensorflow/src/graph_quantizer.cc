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
#include "cross_layers_equalization.h"
#include "deploy_quantized_graph.h"
#include "flatten_atrous.h"
#include "fold_batch_norms.h"
#include "fold_constants.h"
#include "graph_quantizer.h"
#include "known_patterns.h"
#include "quantize_utils.h"
#include "remove_nodes.h"
#include "separate_shared_constants.h"
#include "strip_unused_nodes.h"
#include "tensorflow/core/lib/strings/str_util.h"

namespace tensorflow {
namespace decent_q {

namespace {

const QuantizeConfig GetWtConfig(const QuantizeConfig &config,
                                 const string &op = "") {
  QuantizeConfig wt_config(config);
  if (op == "DepthwiseConv2d" || op == "DepthwiseConv2dNative") {
    wt_config.mode = QuantizeMode::DW_WEIGHT;
  } else {
    wt_config.mode = QuantizeMode::WEIGHT;
  }
  return wt_config;
}

const QuantizeConfig GetActConfig(const QuantizeConfig &config) {
  QuantizeConfig act_config(config);
  act_config.mode = QuantizeMode::ACTIVATION;
  return act_config;
}

const QuantizeConfig GetInputConfig(const QuantizeConfig &config) {
  QuantizeConfig input_config(config);
  input_config.method = QuantizeMethod::NOOF;
  input_config.mode = QuantizeMode::ACTIVATION;
  return input_config;
}

typedef std::unordered_map<string, QuantizeConfig> NodeConfigMap;

bool MergeNodeConfigMap(NodeConfigMap &target, const NodeConfigMap &source) {
  bool new_inserted = false;
  for (auto iter = source.begin(); iter != source.end(); iter++) {
    if (!target.count(iter->first)) {
      target.insert(*iter);
      new_inserted = true;
    }
  }
  return new_inserted;
}

bool UpdateNodeConfigMap(const NodeDef &node, const QuantizeConfig &config,
                         NodeConfigMap &ops_to_quantize) {
  DataType type = DT_FLOAT;
  if (node.attr().count("dtype")) {
    type = node.attr().at("dtype").type();
  } else if (node.attr().count("T")) {
    type = node.attr().at("T").type();
  }
  if (type != DT_FLOAT) {
    DLOG_WARNING << "Found node with non-quantizable dtype: " << node.name()
                 << " (type: " << node.op() << " dtype: " << type << ").";
    ops_to_quantize.clear();
    return false;
  }
  if (ops_to_quantize.count(node.name())) {
    const auto &existed_config = ops_to_quantize[node.name()];
    if (existed_config.phase != config.phase ||
        existed_config.method != config.method ||
        existed_config.weight_bit != config.weight_bit ||
        existed_config.activation_bit != config.activation_bit ||
        existed_config.mode != config.mode) {
      DLOG_WARNING << "Found repeated node with different config, using last "
                      "config. node name: "
                   << node.name();
    }
  }
  ops_to_quantize[node.name()] = config;
  return true;
}

bool CheckDtype(const NodeDef &node) {
  bool pass = true;
  if (node.attr().count("dtype") &&
      node.attr().at("dtype").type() != DT_FLOAT) {
    DLOG_WARNING << "Found node with non-quantizable dtype: " << node.name()
                 << " (op: " << node.op()
                 << ", dtype: " << node.attr().at("dtype").type() << ")";
    pass = false;

  } else if (node.attr().count("T") && node.attr().at("T").type() != DT_FLOAT) {
    DLOG_WARNING << "Found node with non-quantizable T: " << node.name()
                 << " (op: " << node.op()
                 << ", T: " << node.attr().at("T").type() << ")";
    pass = false;
  }
  return pass;
}

NodeConfigMap LocatePlaceholder(const NodeMatch &match,
                                const QuantizeConfig &config,
                                std::set<NodeGroup> &node_groups) {
  NodeConfigMap ops_to_quantize;
  const NodeDef &placeholder_node = match.node;
  if (!CheckDtype(placeholder_node)) return ops_to_quantize;

  bool updated = UpdateNodeConfigMap(placeholder_node, GetInputConfig(config),
                                     ops_to_quantize);
  if (updated) {
    DLOG_INFO(1) << "Quantize placeholder: " << placeholder_node.name() << "("
                 << placeholder_node.op() << ")";
  }
  return ops_to_quantize;
}

NodeConfigMap LocateConvfcBiasIdRelu(const NodeMatch &match,
                                     const QuantizeConfig &config,
                                     std::set<NodeGroup> &node_groups) {
  NodeConfigMap ops_to_quantize;
  const NodeDef &relu_node = match.node;
  const NodeDef &identity_node = match.inputs[0].node;
  const NodeDef &biasadd_node = match.inputs[0].inputs[0].node;
  const NodeDef &bias_node = match.inputs[0].inputs[0].inputs[1].node;
  const NodeDef &convfc_node = match.inputs[0].inputs[0].inputs[0].node;
  const NodeDef &input_node =
      match.inputs[0].inputs[0].inputs[0].inputs[0].node;
  const NodeDef &weight_node =
      match.inputs[0].inputs[0].inputs[0].inputs[1].node;
  if (!CheckDtype(convfc_node)) return ops_to_quantize;

  bool updated =
      UpdateNodeConfigMap(weight_node, GetWtConfig(config), ops_to_quantize);
  updated = updated && UpdateNodeConfigMap(bias_node, GetWtConfig(config),
                                           ops_to_quantize);
  updated = updated && UpdateNodeConfigMap(relu_node, GetActConfig(config),
                                           ops_to_quantize);
  if (updated) {
    DLOG_INFO(1) << "Quantize convfc + bias + identity + relu: "
                 << relu_node.name() << " + " << identity_node.name() << " + "
                 << biasadd_node.name() << "(" << biasadd_node.op() << ") <-- "
                 << convfc_node.name() << " + " << bias_node.name();
    node_groups.insert(std::vector<string>{
        convfc_node.name(), input_node.name(), relu_node.name(),
        weight_node.name(), bias_node.name()});
  }
  return ops_to_quantize;
}

NodeConfigMap LocateConvfcBiasRelu(const NodeMatch &match,
                                   const QuantizeConfig &config,
                                   std::set<NodeGroup> &node_groups) {
  NodeConfigMap ops_to_quantize;
  const NodeDef &relu_node = match.node;
  const NodeDef &biasadd_node = match.inputs[0].node;
  const NodeDef &bias_node = match.inputs[0].inputs[1].node;
  const NodeDef &convfc_node = match.inputs[0].inputs[0].node;
  const NodeDef &input_node = match.inputs[0].inputs[0].inputs[0].node;
  const NodeDef &weight_node = match.inputs[0].inputs[0].inputs[1].node;
  if (!CheckDtype(convfc_node)) return ops_to_quantize;

  bool updated = UpdateNodeConfigMap(
      weight_node, GetWtConfig(config, convfc_node.op()), ops_to_quantize);
  updated = updated && UpdateNodeConfigMap(bias_node, GetWtConfig(config),
                                           ops_to_quantize);
  updated = updated && UpdateNodeConfigMap(relu_node, GetActConfig(config),
                                           ops_to_quantize);
  if (updated) {
    DLOG_INFO(1) << "Quantize convfc + bias + relu: " << relu_node.name()
                 << " + " << biasadd_node.name() << "(" << biasadd_node.op()
                 << ") <-- " << convfc_node.name() << " + " << bias_node.name();
    node_groups.insert(std::vector<string>{
        convfc_node.name(), input_node.name(), relu_node.name(),
        weight_node.name(), bias_node.name()});
  }
  return ops_to_quantize;
}

NodeConfigMap LocateConvfcBias(const NodeMatch &match,
                               const QuantizeConfig &config,
                               std::set<NodeGroup> &node_groups) {
  NodeConfigMap ops_to_quantize;
  const NodeDef &biasadd_node = match.node;
  const NodeDef &bias_node = match.inputs[1].node;
  const NodeDef &convfc_node = match.inputs[0].node;
  const NodeDef &input_node = match.inputs[0].inputs[0].node;
  const NodeDef &weight_node = match.inputs[0].inputs[1].node;
  if (!CheckDtype(convfc_node)) return ops_to_quantize;

  bool updated = UpdateNodeConfigMap(
      weight_node, GetWtConfig(config, convfc_node.op()), ops_to_quantize);
  updated = updated && UpdateNodeConfigMap(bias_node, GetWtConfig(config),
                                           ops_to_quantize);
  updated = updated && UpdateNodeConfigMap(biasadd_node, GetActConfig(config),
                                           ops_to_quantize);
  if (updated) {
    DLOG_INFO(1) << "Quantize convfc + bias: " << biasadd_node.name() << "("
                 << biasadd_node.op() << ") <-- " << convfc_node.name() << " + "
                 << bias_node.name();
    node_groups.insert(std::vector<string>{
        convfc_node.name(), input_node.name(), biasadd_node.name(),
        weight_node.name(), bias_node.name()});
  }
  return ops_to_quantize;
}

NodeConfigMap LocateConvfcRelu(const NodeMatch &match,
                               const QuantizeConfig &config,
                               std::set<NodeGroup> &node_groups) {
  NodeConfigMap ops_to_quantize;
  const NodeDef &relu_node = match.node;
  const NodeDef &convfc_node = match.inputs[0].node;
  const NodeDef &input_node = match.inputs[0].inputs[0].node;
  const NodeDef &weight_node = match.inputs[0].inputs[1].node;
  if (!CheckDtype(convfc_node)) return ops_to_quantize;

  bool updated = UpdateNodeConfigMap(
      weight_node, GetWtConfig(config, convfc_node.op()), ops_to_quantize);
  updated = updated && UpdateNodeConfigMap(relu_node, GetActConfig(config),
                                           ops_to_quantize);
  if (updated) {
    DLOG_INFO(1) << "Quantize convfc + relu: " << relu_node.name() << "("
                 << relu_node.op() << ") <-- " << convfc_node.name();
    node_groups.insert(std::vector<string>{convfc_node.name(),
                                           input_node.name(), relu_node.name(),
                                           weight_node.name(), "NULL"});
  }
  return ops_to_quantize;
}

NodeConfigMap LocateConvfc(const NodeMatch &match, const QuantizeConfig &config,
                           std::set<NodeGroup> &node_groups) {
  NodeConfigMap ops_to_quantize;
  const NodeDef &convfc_node = match.node;
  const NodeDef &input_node = match.inputs[0].node;
  const NodeDef &weight_node = match.inputs[1].node;
  if (!CheckDtype(convfc_node)) return ops_to_quantize;

  bool updated = UpdateNodeConfigMap(
      weight_node, GetWtConfig(config, convfc_node.op()), ops_to_quantize);
  updated = updated && UpdateNodeConfigMap(convfc_node, GetActConfig(config),
                                           ops_to_quantize);
  if (updated) {
    DLOG_INFO(1) << "Quantize convfc: " << convfc_node.name();
    node_groups.insert(
        std::vector<string>{convfc_node.name(), input_node.name(),
                            convfc_node.name(), weight_node.name(), "NULL"});
  }
  return ops_to_quantize;
}

NodeConfigMap LocateConv2dTransposeBiasRelu(const NodeMatch &match,
                                            const QuantizeConfig &config,
                                            std::set<NodeGroup> &node_groups) {
  NodeConfigMap ops_to_quantize;
  const NodeDef &relu_node = match.node;
  const NodeDef &biasadd_node = match.inputs[0].node;
  const NodeDef &bias_node = match.inputs[0].inputs[1].node;
  const NodeDef &conv2d_transpose_node = match.inputs[0].inputs[0].node;
  const NodeDef &input_node = match.inputs[0].inputs[0].inputs[2].node;
  const NodeDef &weight_node = match.inputs[0].inputs[0].inputs[1].node;
  if (!CheckDtype(conv2d_transpose_node)) return ops_to_quantize;

  bool updated =
      UpdateNodeConfigMap(weight_node, GetWtConfig(config), ops_to_quantize);
  updated = updated && UpdateNodeConfigMap(bias_node, GetWtConfig(config),
                                           ops_to_quantize);
  updated = updated && UpdateNodeConfigMap(relu_node, GetActConfig(config),
                                           ops_to_quantize);
  if (updated) {
    DLOG_INFO(1) << "Quantize conv2d_transpose + bias + relu: "
                 << relu_node.name() << " + " << biasadd_node.name() << "("
                 << biasadd_node.op() << ") <-- "
                 << conv2d_transpose_node.name() << " + " << bias_node.name();
    node_groups.insert(std::vector<string>{
        conv2d_transpose_node.name(), input_node.name(), relu_node.name(),
        weight_node.name(), bias_node.name()});
  }
  return ops_to_quantize;
}

NodeConfigMap LocateConv2dTransposeBias(const NodeMatch &match,
                                        const QuantizeConfig &config,
                                        std::set<NodeGroup> &node_groups) {
  NodeConfigMap ops_to_quantize;
  const NodeDef &biasadd_node = match.node;
  const NodeDef &bias_node = match.inputs[1].node;
  const NodeDef &conv2d_transpose_node = match.inputs[0].node;
  const NodeDef &input_node = match.inputs[0].inputs[2].node;
  const NodeDef &weight_node = match.inputs[0].inputs[1].node;
  if (!CheckDtype(conv2d_transpose_node)) return ops_to_quantize;

  bool updated =
      UpdateNodeConfigMap(weight_node, GetWtConfig(config), ops_to_quantize);
  updated = updated && UpdateNodeConfigMap(bias_node, GetWtConfig(config),
                                           ops_to_quantize);
  updated = updated && UpdateNodeConfigMap(biasadd_node, GetActConfig(config),
                                           ops_to_quantize);
  if (updated) {
    DLOG_INFO(1) << "Quantize conv2d_transpose + bias: " << biasadd_node.name()
                 << "(" << biasadd_node.op() << ") <-- "
                 << conv2d_transpose_node.name() << " + " << bias_node.name();
    node_groups.insert(std::vector<string>{
        conv2d_transpose_node.name(), input_node.name(), biasadd_node.name(),
        weight_node.name(), bias_node.name()});
  }
  return ops_to_quantize;
}

NodeConfigMap LocateConv2dTransposeRelu(const NodeMatch &match,
                                        const QuantizeConfig &config,
                                        std::set<NodeGroup> &node_groups) {
  NodeConfigMap ops_to_quantize;
  const NodeDef &relu_node = match.node;
  const NodeDef &conv2d_transpose_node = match.inputs[0].node;
  const NodeDef &input_node = match.inputs[0].inputs[2].node;
  const NodeDef &weight_node = match.inputs[0].inputs[1].node;
  if (!CheckDtype(conv2d_transpose_node)) return ops_to_quantize;

  bool updated =
      UpdateNodeConfigMap(weight_node, GetWtConfig(config), ops_to_quantize);
  updated = updated && UpdateNodeConfigMap(relu_node, GetActConfig(config),
                                           ops_to_quantize);
  if (updated) {
    DLOG_INFO(1) << "Quantize conv2d_transpose + relu: " << relu_node.name()
                 << "(" << relu_node.op() << ") <-- "
                 << conv2d_transpose_node.name();
    node_groups.insert(std::vector<string>{conv2d_transpose_node.name(),
                                           input_node.name(), relu_node.name(),
                                           weight_node.name(), "NULL"});
  }
  return ops_to_quantize;
}

NodeConfigMap LocateConv2dTranspose(const NodeMatch &match,
                                    const QuantizeConfig &config,
                                    std::set<NodeGroup> &node_groups) {
  NodeConfigMap ops_to_quantize;
  const NodeDef &conv2d_transpose_node = match.node;
  const NodeDef &input_node = match.inputs[2].node;
  const NodeDef &weight_node = match.inputs[1].node;
  if (!CheckDtype(conv2d_transpose_node)) return ops_to_quantize;

  bool updated =
      UpdateNodeConfigMap(weight_node, GetWtConfig(config), ops_to_quantize);
  updated =
      updated && UpdateNodeConfigMap(conv2d_transpose_node,
                                     GetActConfig(config), ops_to_quantize);
  if (updated) {
    DLOG_INFO(1) << "Quantize conv2d_transpose: "
                 << conv2d_transpose_node.name() << "("
                 << conv2d_transpose_node.op() << ")";
    node_groups.insert(std::vector<string>{
        conv2d_transpose_node.name(), input_node.name(),
        conv2d_transpose_node.name(), weight_node.name(), "NULL"});
  }
  return ops_to_quantize;
}

NodeConfigMap LocateAtrousConvBiasRelu(const NodeMatch &match,
                                       const QuantizeConfig &config,
                                       std::set<NodeGroup> &node_groups) {
  NodeConfigMap ops_to_quantize;
  const NodeDef &relu_node = match.node;
  const NodeDef &bias_node = match.inputs[0].inputs[1].node;
  const NodeDef &atrous_conv_node = match.inputs[0].inputs[0].inputs[0].node;
  const NodeDef &input_node =
      match.inputs[0].inputs[0].inputs[0].inputs[0].inputs[0].node;
  const NodeDef &weight_node =
      match.inputs[0].inputs[0].inputs[0].inputs[1].node;
  if (!CheckDtype(atrous_conv_node)) return ops_to_quantize;

  bool updated =
      UpdateNodeConfigMap(weight_node, GetWtConfig(config), ops_to_quantize);
  updated = updated & UpdateNodeConfigMap(bias_node, GetWtConfig(config),
                                          ops_to_quantize);
  updated = updated & UpdateNodeConfigMap(relu_node, GetActConfig(config),
                                          ops_to_quantize);
  if (updated) {
    DLOG_INFO(1) << "Quantize atrou_conv_bias_relu: " << relu_node.name()
                 << " <-- " << atrous_conv_node.name() << "("
                 << atrous_conv_node.op() << ")";
    node_groups.insert(std::vector<string>{
        atrous_conv_node.name(), input_node.name(), relu_node.name(),
        weight_node.name(), bias_node.name()});
  }
  return ops_to_quantize;
}

NodeConfigMap LocateAtrousConvBias(const NodeMatch &match,
                                   const QuantizeConfig &config,
                                   std::set<NodeGroup> &node_groups) {
  NodeConfigMap ops_to_quantize;
  const NodeDef &biasadd_node = match.node;
  const NodeDef &bias_node = match.inputs[1].node;
  const NodeDef &atrous_conv_node = match.inputs[0].inputs[0].node;
  const NodeDef &input_node =
      match.inputs[0].inputs[0].inputs[0].inputs[0].node;
  const NodeDef &weight_node = match.inputs[0].inputs[0].inputs[1].node;
  if (!CheckDtype(atrous_conv_node)) return ops_to_quantize;

  bool updated =
      UpdateNodeConfigMap(weight_node, GetWtConfig(config), ops_to_quantize);
  updated = updated & UpdateNodeConfigMap(bias_node, GetWtConfig(config),
                                          ops_to_quantize);
  updated = updated & UpdateNodeConfigMap(biasadd_node, GetActConfig(config),
                                          ops_to_quantize);
  if (updated) {
    DLOG_INFO(1) << "Quantize atrou_conv_bias: " << biasadd_node.name()
                 << " <-- " << atrous_conv_node.name() << "("
                 << atrous_conv_node.op() << ")";
    node_groups.insert(std::vector<string>{
        atrous_conv_node.name(), input_node.name(), biasadd_node.name(),
        weight_node.name(), bias_node.name()});
  }
  return ops_to_quantize;
}

NodeConfigMap LocateAtrousConvRelu(const NodeMatch &match,
                                   const QuantizeConfig &config,
                                   std::set<NodeGroup> &node_groups) {
  NodeConfigMap ops_to_quantize;
  const NodeDef &relu_node = match.node;
  const NodeDef &atrous_conv_node = match.inputs[0].inputs[0].node;
  const NodeDef &input_node =
      match.inputs[0].inputs[0].inputs[0].inputs[0].node;
  const NodeDef &weight_node = match.inputs[0].inputs[0].inputs[1].node;
  if (!CheckDtype(atrous_conv_node)) return ops_to_quantize;

  bool updated =
      UpdateNodeConfigMap(weight_node, GetWtConfig(config), ops_to_quantize);
  updated = updated & UpdateNodeConfigMap(relu_node, GetActConfig(config),
                                          ops_to_quantize);
  if (updated) {
    DLOG_INFO(1) << "Quantize atrou_conv_relu: " << relu_node.name() << " <-- "
                 << atrous_conv_node.name() << "(" << atrous_conv_node.op()
                 << ")";
    node_groups.insert(std::vector<string>{atrous_conv_node.name(),
                                           input_node.name(), relu_node.name(),
                                           weight_node.name(), "NULL"});
  }
  return ops_to_quantize;
}

NodeConfigMap LocateAtrousConv(const NodeMatch &match,
                               const QuantizeConfig &config,
                               std::set<NodeGroup> &node_groups) {
  NodeConfigMap ops_to_quantize;
  const NodeDef &batch_to_space_node = match.node;
  const NodeDef &atrous_conv_node = match.inputs[0].node;
  const NodeDef &input_node = match.inputs[0].inputs[0].inputs[0].node;
  const NodeDef &weight_node = match.inputs[0].inputs[1].node;
  if (!CheckDtype(atrous_conv_node)) return ops_to_quantize;

  bool updated =
      UpdateNodeConfigMap(weight_node, GetWtConfig(config), ops_to_quantize);
  updated =
      updated & UpdateNodeConfigMap(batch_to_space_node, GetActConfig(config),
                                    ops_to_quantize);
  if (updated) {
    DLOG_INFO(1) << "Quantize atrou_conv: " << atrous_conv_node.name() << "("
                 << atrous_conv_node.op() << ")";
    node_groups.insert(std::vector<string>{
        atrous_conv_node.name(), input_node.name(), batch_to_space_node.name(),
        weight_node.name(), "NULL"});
  }
  return ops_to_quantize;
}

NodeConfigMap LocateConvfcBiasLeakyRelu(const NodeMatch &match,
                                        const QuantizeConfig &config,
                                        std::set<NodeGroup> &node_groups) {
  NodeConfigMap ops_to_quantize;
  const NodeDef &leakyrelu_node = match.node;
  const NodeDef &alpha_node = match.inputs[0].inputs[0].node;
  const NodeDef &biasadd_node = match.inputs[1].node;
  const NodeDef &bias_node = match.inputs[1].inputs[1].node;
  const NodeDef &convfc_node = match.inputs[1].inputs[0].node;
  const NodeDef &input_node = match.inputs[1].inputs[0].inputs[0].node;
  const NodeDef &weight_node = match.inputs[1].inputs[0].inputs[1].node;
  if (!CheckDtype(alpha_node) || !CheckDtype(convfc_node))
    return ops_to_quantize;

  bool updated = UpdateNodeConfigMap(
      weight_node, GetWtConfig(config, convfc_node.op()), ops_to_quantize);
  updated = updated && UpdateNodeConfigMap(bias_node, GetWtConfig(config),
                                           ops_to_quantize);
  updated = updated && UpdateNodeConfigMap(alpha_node, GetWtConfig(config),
                                           ops_to_quantize);
  updated = updated && UpdateNodeConfigMap(leakyrelu_node, GetActConfig(config),
                                           ops_to_quantize);
  if (updated) {
    DLOG_INFO(1) << "Quantize convfc + bias + leakyrelu: "
                 << leakyrelu_node.name() << "(" << leakyrelu_node.op()
                 << ") <-- " << biasadd_node.name() << "(" << biasadd_node.op()
                 << ") <-- " << convfc_node.name() << "(" << convfc_node.op()
                 << ")";
    node_groups.insert(std::vector<string>{
        convfc_node.name(), input_node.name(), leakyrelu_node.name(),
        weight_node.name(), bias_node.name()});
  }
  return ops_to_quantize;
}

NodeConfigMap LocateConvfcLeakyRelu(const NodeMatch &match,
                                    const QuantizeConfig &config,
                                    std::set<NodeGroup> &node_groups) {
  NodeConfigMap ops_to_quantize;
  const NodeDef &leakyrelu_node = match.node;
  const NodeDef &alpha_node = match.inputs[0].inputs[0].node;
  const NodeDef &convfc_node = match.inputs[1].node;
  const NodeDef &input_node = match.inputs[1].inputs[0].node;
  const NodeDef &weight_node = match.inputs[1].inputs[1].node;
  if (!CheckDtype(alpha_node) || !CheckDtype(convfc_node))
    return ops_to_quantize;

  bool updated = UpdateNodeConfigMap(
      weight_node, GetWtConfig(config, convfc_node.op()), ops_to_quantize);
  updated = updated && UpdateNodeConfigMap(alpha_node, GetWtConfig(config),
                                           ops_to_quantize);
  updated = updated && UpdateNodeConfigMap(leakyrelu_node, GetActConfig(config),
                                           ops_to_quantize);
  if (updated) {
    DLOG_INFO(1) << "Quantize convfc + leakyrelu: " << leakyrelu_node.name()
                 << "(" << leakyrelu_node.op() << ") <-- " << convfc_node.name()
                 << "(" << convfc_node.op() << ")";
    node_groups.insert(
        std::vector<string>{convfc_node.name(), input_node.name(),
                            leakyrelu_node.name(), weight_node.name(), "NULL"});
  }
  return ops_to_quantize;
}

NodeConfigMap LocateHardSigmoid(const NodeMatch &match,
                                const QuantizeConfig &config,
                                std::set<NodeGroup> &node_groups) {
  NodeConfigMap ops_to_quantize;
  const NodeDef &vitis_scale_node = match.node;
  const NodeDef &mul_node = match.inputs[0].node;
  if (!CheckDtype(mul_node) || !CheckDtype(vitis_scale_node))
    return ops_to_quantize;

  bool updated = UpdateNodeConfigMap(vitis_scale_node, GetActConfig(config),
                                     ops_to_quantize);
  if (updated) {
    DLOG_INFO(1) << "Quantize hard sigmoid: " << mul_node.name();
  }
  return ops_to_quantize;
}

NodeConfigMap LocateHardSwish(const NodeMatch &match,
                              const QuantizeConfig &config,
                              std::set<NodeGroup> &node_groups) {
  NodeConfigMap ops_to_quantize;
  const NodeDef &mul_x_node = match.node;
  const NodeDef &vitis_scale_node = match.inputs[1].node;
  const NodeDef &input_node = match.inputs[0].node;
  if (!CheckDtype(mul_x_node)) return ops_to_quantize;

  bool updated =
      UpdateNodeConfigMap(mul_x_node, GetActConfig(config), ops_to_quantize);
  updated =
      updated && UpdateNodeConfigMap(vitis_scale_node, GetActConfig(config),
                                     ops_to_quantize);
  if (updated) {
    DLOG_INFO(1) << "Quantize hard swish: " << mul_x_node.name();
  }
  return ops_to_quantize;
}

NodeConfigMap LocateSwish(const NodeMatch &match, const QuantizeConfig &config,
                          std::set<NodeGroup> &node_groups) {
  NodeConfigMap ops_to_quantize;
  const NodeDef &mul_node = match.node;
  const NodeDef &sigmoid_node = match.inputs[1].node;
  const NodeDef &input_node = match.inputs[0].node;
  if (!CheckDtype(sigmoid_node)) return ops_to_quantize;

  bool updated =
      UpdateNodeConfigMap(mul_node, GetActConfig(config), ops_to_quantize);
  updated = updated && UpdateNodeConfigMap(sigmoid_node, GetActConfig(config),
                                           ops_to_quantize);
  updated = updated && UpdateNodeConfigMap(input_node, GetActConfig(config),
                                           ops_to_quantize);
  if (updated) {
    DLOG_INFO(1) << "Quantize swish: " << mul_node.name();
  }
  return ops_to_quantize;
}

NodeConfigMap LocateLeakyRelu(const NodeMatch &match,
                              const QuantizeConfig &config,
                              std::set<NodeGroup> &node_groups) {
  NodeConfigMap ops_to_quantize;
  const NodeDef &leakyrelu_node = match.node;
  const NodeDef &alpha_node = match.inputs[0].inputs[0].node;
  if (!CheckDtype(alpha_node)) return ops_to_quantize;

  Tensor alpha = GetNodeTensorAttr(alpha_node, "value");

  bool updated =
      UpdateNodeConfigMap(alpha_node, GetWtConfig(config), ops_to_quantize);
  updated = updated && UpdateNodeConfigMap(leakyrelu_node, GetActConfig(config),
                                           ops_to_quantize);
  if (updated) {
    DLOG_INFO(1) << "Quantize leakyrelu: " << leakyrelu_node.name();
  }
  return ops_to_quantize;
}

NodeConfigMap LocateConvfcBiasFusedLeakyRelu(const NodeMatch &match,
                                             const QuantizeConfig &config,
                                             std::set<NodeGroup> &node_groups) {
  NodeConfigMap ops_to_quantize;
  const NodeDef &leakyrelu_node = match.node;
  const NodeDef &biasadd_node = match.inputs[0].node;
  const NodeDef &bias_node = match.inputs[0].inputs[1].node;
  const NodeDef &convfc_node = match.inputs[0].inputs[0].node;
  const NodeDef &input_node = match.inputs[0].inputs[0].inputs[0].node;
  const NodeDef &weight_node = match.inputs[0].inputs[0].inputs[1].node;
  if (!CheckDtype(convfc_node)) return ops_to_quantize;

  bool updated =
      UpdateNodeConfigMap(weight_node, GetWtConfig(config), ops_to_quantize);
  updated = updated && UpdateNodeConfigMap(bias_node, GetWtConfig(config),
                                           ops_to_quantize);
  updated = updated && UpdateNodeConfigMap(leakyrelu_node, GetActConfig(config),
                                           ops_to_quantize);
  if (updated) {
    DLOG_INFO(1) << "Quantize convfc + bias + leakyrelu: "
                 << leakyrelu_node.name() << "(" << leakyrelu_node.op()
                 << ") <-- " << biasadd_node.name() << "(" << biasadd_node.op()
                 << ") <-- " << convfc_node.name() << "(" << convfc_node.op()
                 << ")";
    node_groups.insert(std::vector<string>{
        convfc_node.name(), input_node.name(), leakyrelu_node.name(),
        weight_node.name(), bias_node.name()});
  }
  return ops_to_quantize;
}

NodeConfigMap LocateConvfcFusedLeakyRelu(const NodeMatch &match,
                                         const QuantizeConfig &config,
                                         std::set<NodeGroup> &node_groups) {
  NodeConfigMap ops_to_quantize;
  const NodeDef &leakyrelu_node = match.node;
  const NodeDef &convfc_node = match.inputs[0].node;
  const NodeDef &input_node = match.inputs[0].inputs[0].node;
  const NodeDef &weight_node = match.inputs[0].inputs[1].node;
  if (!CheckDtype(convfc_node)) return ops_to_quantize;

  bool updated =
      UpdateNodeConfigMap(weight_node, GetWtConfig(config), ops_to_quantize);
  updated = updated && UpdateNodeConfigMap(leakyrelu_node, GetActConfig(config),
                                           ops_to_quantize);
  if (updated) {
    DLOG_INFO(1) << "Quantize convfc + leakyrelu: " << leakyrelu_node.name()
                 << "(" << leakyrelu_node.op() << ") <-- " << convfc_node.name()
                 << "(" << convfc_node.op() << ")";
    node_groups.insert(
        std::vector<string>{convfc_node.name(), input_node.name(),
                            leakyrelu_node.name(), weight_node.name(), "NULL"});
  }
  return ops_to_quantize;
}

NodeConfigMap LocateFusedLeakyRelu(const NodeMatch &match,
                                   const QuantizeConfig &config,
                                   std::set<NodeGroup> &node_groups) {
  NodeConfigMap ops_to_quantize;
  const NodeDef &leakyrelu_node = match.node;
  bool updated = UpdateNodeConfigMap(leakyrelu_node, GetActConfig(config),
                                     ops_to_quantize);
  if (updated) {
    DLOG_INFO(1) << "Quantize leakyrelu: " << leakyrelu_node.name();
  }
  return ops_to_quantize;
}

NodeConfigMap LocateConvfcBiasKerasLeakyRelu(const NodeMatch &match,
                                             const QuantizeConfig &config,
                                             std::set<NodeGroup> &node_groups) {
  NodeConfigMap ops_to_quantize;
  const NodeDef &leakyrelu_node = match.node;
  const NodeDef &alpha_node = match.inputs[1].inputs[0].node;
  const NodeDef &biasadd_node = match.inputs[0].inputs[0].node;
  const NodeDef &bias_node = match.inputs[0].inputs[0].inputs[1].node;
  const NodeDef &convfc_node = match.inputs[0].inputs[0].inputs[0].node;
  const NodeDef &input_node =
      match.inputs[0].inputs[0].inputs[0].inputs[0].node;
  const NodeDef &weight_node =
      match.inputs[0].inputs[0].inputs[0].inputs[1].node;
  if (!CheckDtype(alpha_node) || !CheckDtype(convfc_node))
    return ops_to_quantize;

  bool updated = UpdateNodeConfigMap(
      weight_node, GetWtConfig(config, convfc_node.op()), ops_to_quantize);
  updated = updated && UpdateNodeConfigMap(bias_node, GetWtConfig(config),
                                           ops_to_quantize);
  updated = updated && UpdateNodeConfigMap(alpha_node, GetWtConfig(config),
                                           ops_to_quantize);
  updated = updated && UpdateNodeConfigMap(leakyrelu_node, GetActConfig(config),
                                           ops_to_quantize);
  if (updated) {
    DLOG_INFO(1) << "Quantize convfc + bias + keras_leakyrelu: "
                 << leakyrelu_node.name() << "(" << leakyrelu_node.op()
                 << ") <-- " << biasadd_node.name() << "(" << biasadd_node.op()
                 << ") <-- " << convfc_node.name() << "(" << convfc_node.op()
                 << ")";
    node_groups.insert(std::vector<string>{
        convfc_node.name(), input_node.name(), leakyrelu_node.name(),
        weight_node.name(), bias_node.name()});
  }
  return ops_to_quantize;
}

NodeConfigMap LocateConvfcKerasLeakyRelu(const NodeMatch &match,
                                         const QuantizeConfig &config,
                                         std::set<NodeGroup> &node_groups) {
  NodeConfigMap ops_to_quantize;
  const NodeDef &leakyrelu_node = match.node;
  const NodeDef &alpha_node = match.inputs[1].inputs[0].node;
  const NodeDef &convfc_node = match.inputs[0].inputs[0].node;
  const NodeDef &input_node = match.inputs[0].inputs[0].inputs[0].node;
  const NodeDef &weight_node = match.inputs[0].inputs[0].inputs[1].node;
  if (!CheckDtype(alpha_node) || !CheckDtype(convfc_node))
    return ops_to_quantize;

  bool updated = UpdateNodeConfigMap(
      weight_node, GetWtConfig(config, convfc_node.op()), ops_to_quantize);
  updated = updated && UpdateNodeConfigMap(alpha_node, GetWtConfig(config),
                                           ops_to_quantize);
  updated = updated && UpdateNodeConfigMap(leakyrelu_node, GetActConfig(config),
                                           ops_to_quantize);
  if (updated) {
    DLOG_INFO(1) << "Quantize convfc + keras_leakyrelu: "
                 << leakyrelu_node.name() << "(" << leakyrelu_node.op()
                 << ") <-- " << convfc_node.name() << "(" << convfc_node.op()
                 << ")";
    node_groups.insert(
        std::vector<string>{convfc_node.name(), input_node.name(),
                            leakyrelu_node.name(), weight_node.name(), "NULL"});
  }
  return ops_to_quantize;
}

NodeConfigMap LocateKerasLeakyRelu(const NodeMatch &match,
                                   const QuantizeConfig &config,
                                   std::set<NodeGroup> &node_groups) {
  NodeConfigMap ops_to_quantize;
  const NodeDef &leakyrelu_node = match.node;
  const NodeDef &alpha_node = match.inputs[1].inputs[0].node;
  if (!CheckDtype(alpha_node)) return ops_to_quantize;

  bool updated =
      UpdateNodeConfigMap(alpha_node, GetWtConfig(config), ops_to_quantize);
  updated = updated && UpdateNodeConfigMap(leakyrelu_node, GetActConfig(config),
                                           ops_to_quantize);
  if (updated) {
    DLOG_INFO(1) << "Quantize keras_leakyrelu: " << leakyrelu_node.name();
  }
  return ops_to_quantize;
}

NodeConfigMap LocateUpsampling(const NodeMatch &match,
                               const QuantizeConfig &config,
                               std::set<NodeGroup> &node_groups) {
  NodeConfigMap ops_to_quantize;
  const NodeDef &resize_node = match.node;
  if (!CheckDtype(resize_node)) return ops_to_quantize;

  bool updated =
      UpdateNodeConfigMap(resize_node, GetActConfig(config), ops_to_quantize);
  if (updated) {
    DLOG_INFO(1) << "Quantize upsampling: " << resize_node.name() << "("
                 << resize_node.op() << ")";
  }
  return ops_to_quantize;
}

NodeConfigMap LocateResize(const NodeMatch &match, const QuantizeConfig &config,
                           std::set<NodeGroup> &node_groups) {
  NodeConfigMap ops_to_quantize;
  const NodeDef &resize_node = match.node;
  if (!CheckDtype(resize_node)) return ops_to_quantize;

  bool updated =
      UpdateNodeConfigMap(resize_node, GetActConfig(config), ops_to_quantize);
  if (updated) {
    DLOG_INFO(1) << "Quantize resize: " << resize_node.name() << "("
                 << resize_node.op() << ")";
  }
  return ops_to_quantize;
}

NodeConfigMap LocateDepthToSpace(const NodeMatch &match,
                                 const QuantizeConfig &config,
                                 std::set<NodeGroup> &node_groups) {
  NodeConfigMap ops_to_quantize;
  const NodeDef &depth_to_space_node = match.node;
  if (!CheckDtype(depth_to_space_node)) return ops_to_quantize;

  bool updated = UpdateNodeConfigMap(depth_to_space_node, GetActConfig(config),
                                     ops_to_quantize);
  if (updated) {
    DLOG_INFO(1) << "Quantize DepthToSpace: " << depth_to_space_node.name()
                 << "(" << depth_to_space_node.op() << ")";
  }
  return ops_to_quantize;
}

NodeConfigMap LocateTPUNearestNeighborUpsampling(
    const NodeMatch &match, const QuantizeConfig &config,
    std::set<NodeGroup> &node_groups) {
  NodeConfigMap ops_to_quantize;
  const NodeDef &output_reshape_node = match.node;
  if (!CheckDtype(output_reshape_node)) return ops_to_quantize;

  bool updated = UpdateNodeConfigMap(output_reshape_node, GetActConfig(config),
                                     ops_to_quantize);
  if (updated) {
    DLOG_INFO(1) << "Quantize tpu nearest neighbor upsampling: "
                 << output_reshape_node.name() << "("
                 << output_reshape_node.op() << ")";
  }
  return ops_to_quantize;
}

NodeConfigMap LocateBatchNormRelu(const NodeMatch &match,
                                  const QuantizeConfig &config,
                                  std::set<NodeGroup> &node_groups) {
  NodeConfigMap ops_to_quantize;
  const NodeDef &relu_node = match.node;
  const NodeDef &add_node = match.inputs[0].node;
  const NodeDef &offset_node = match.inputs[0].inputs[1].node;
  const NodeDef &mul_node = match.inputs[0].inputs[0].node;
  const NodeDef &input_node = match.inputs[0].inputs[0].inputs[0].node;
  const NodeDef &scale_node = match.inputs[0].inputs[0].inputs[1].node;
  if (!CheckDtype(mul_node)) return ops_to_quantize;

  bool updated =
      UpdateNodeConfigMap(scale_node, GetWtConfig(config), ops_to_quantize);
  updated = updated & UpdateNodeConfigMap(offset_node, GetWtConfig(config),
                                          ops_to_quantize);
  updated = updated & UpdateNodeConfigMap(relu_node, GetActConfig(config),
                                          ops_to_quantize);
  if (updated) {
    DLOG_INFO(1) << "Quantize batchnorm + relu: " << relu_node.name() << " <-- "
                 << add_node.name() << "(" << add_node.op() << ")"
                 << " <-- " << mul_node.name() << "(" << mul_node.op() << ")";
    node_groups.insert(std::vector<string>{mul_node.name(), input_node.name(),
                                           relu_node.name(), scale_node.name(),
                                           offset_node.name()});
  }
  return ops_to_quantize;
}

NodeConfigMap LocateBatchNorm(const NodeMatch &match,
                              const QuantizeConfig &config,
                              std::set<NodeGroup> &node_groups) {
  NodeConfigMap ops_to_quantize;
  const NodeDef &add_node = match.node;
  const NodeDef &offset_node = match.inputs[1].node;
  const NodeDef &mul_node = match.inputs[0].node;
  const NodeDef &scale_node = match.inputs[0].inputs[1].node;
  const NodeDef &input_node = match.inputs[0].inputs[0].node;
  if (!CheckDtype(mul_node)) return ops_to_quantize;

  bool updated =
      UpdateNodeConfigMap(scale_node, GetWtConfig(config), ops_to_quantize);
  updated = updated & UpdateNodeConfigMap(offset_node, GetWtConfig(config),
                                          ops_to_quantize);
  updated = updated & UpdateNodeConfigMap(add_node, GetActConfig(config),
                                          ops_to_quantize);
  if (updated) {
    DLOG_INFO(1) << "Quantize batchnorm: " << add_node.name() << " <-- "
                 << mul_node.name() << "(" << mul_node.op() << ")";
    node_groups.insert(std::vector<string>{mul_node.name(), input_node.name(),
                                           add_node.name(), scale_node.name(),
                                           offset_node.name()});
  }
  return ops_to_quantize;
}

NodeConfigMap LocateArrayRelu(const NodeMatch &match,
                              const QuantizeConfig &config,
                              std::set<NodeGroup> &node_groups) {
  NodeConfigMap ops_to_quantize;
  const NodeDef &relu_node = match.node;
  const NodeDef &array_node = match.inputs[0].node;
  if (!CheckDtype(array_node)) return ops_to_quantize;

  bool updated =
      UpdateNodeConfigMap(relu_node, GetActConfig(config), ops_to_quantize);
  if (updated) {
    DLOG_INFO(1) << "Quantize array + relu: " << relu_node.name() << " <-- "
                 << array_node.name() << "(" << array_node.op() << ")";
  }
  return ops_to_quantize;
}

NodeConfigMap LocateArray(const NodeMatch &match, const QuantizeConfig &config,
                          std::set<NodeGroup> &node_groups) {
  NodeConfigMap ops_to_quantize;
  const NodeDef &array_node = match.node;
  if (!CheckDtype(array_node)) return ops_to_quantize;

  bool updated =
      UpdateNodeConfigMap(array_node, GetActConfig(config), ops_to_quantize);
  if (updated) {
    DLOG_INFO(1) << "Quantize array: " << array_node.name() << "("
                 << array_node.op() << ")";
  }
  return ops_to_quantize;
}

NodeConfigMap LocateMul_v1(const NodeMatch &match, const QuantizeConfig &config,
                           std::set<NodeGroup> &node_groups) {
  NodeConfigMap ops_to_quantize;
  const NodeDef &mul_node = match.node;
  const NodeDef &scale_node = match.inputs[1].node;
  if (!CheckDtype(mul_node)) return ops_to_quantize;

  bool updated =
      UpdateNodeConfigMap(mul_node, GetActConfig(config), ops_to_quantize);
  updated = updated & UpdateNodeConfigMap(scale_node, GetActConfig(config),
                                          ops_to_quantize);
  if (updated) {
    DLOG_INFO(1) << "Quantize mul v1: " << mul_node.name() << "("
                 << mul_node.op() << ")  +  " << scale_node.name() << "("
                 << scale_node.op() << ")";
  }
  return ops_to_quantize;
}

NodeConfigMap LocateMul_v2(const NodeMatch &match, const QuantizeConfig &config,
                           std::set<NodeGroup> &node_groups) {
  NodeConfigMap ops_to_quantize;
  const NodeDef &mul_node = match.node;
  const NodeDef &scale_node = match.inputs[0].node;
  if (!CheckDtype(mul_node)) return ops_to_quantize;

  bool updated =
      UpdateNodeConfigMap(mul_node, GetActConfig(config), ops_to_quantize);
  updated = updated & UpdateNodeConfigMap(scale_node, GetActConfig(config),
                                          ops_to_quantize);
  if (updated) {
    DLOG_INFO(1) << "Quantize mul v2: " << mul_node.name() << "("
                 << mul_node.op() << ")  +  " << scale_node.name() << "("
                 << scale_node.op() << ")";
  }
  return ops_to_quantize;
}

NodeConfigMap LocateAvgpoolMul(const NodeMatch &match,
                               const QuantizeConfig &config,
                               std::set<NodeGroup> &node_groups) {
  NodeConfigMap ops_to_quantize;
  const NodeDef &mul_node = match.node;
  const NodeDef &avgpool_node = match.inputs[0].node;
  if (!CheckDtype(mul_node)) return ops_to_quantize;

  bool updated =
      UpdateNodeConfigMap(mul_node, GetActConfig(config), ops_to_quantize);
  if (updated) {
    DLOG_INFO(1) << "Quantize avgpool_mul: " << mul_node.name() << "("
                 << avgpool_node.op() << ")";
  }
  return ops_to_quantize;
}

NodeConfigMap LocateClipByValue(const NodeMatch &match,
                                const QuantizeConfig &config,
                                std::set<NodeGroup> &node_groups) {
  NodeConfigMap ops_to_quantize;
  const NodeDef &maximum_node = match.node;
  const NodeDef &minimum_node = match.inputs[0].node;
  if (!CheckDtype(maximum_node)) return ops_to_quantize;

  bool updated =
      UpdateNodeConfigMap(maximum_node, GetActConfig(config), ops_to_quantize);
  if (updated) {
    DLOG_INFO(1) << "Quantize clip_by_value: " << maximum_node.name() << "("
                 << minimum_node.op() << ")";
  }
  return ops_to_quantize;
}

NodeConfigMap LocateOtherRelu(const NodeMatch &match,
                              const QuantizeConfig &config,
                              std::set<NodeGroup> &node_groups) {
  NodeConfigMap ops_to_quantize;
  const NodeDef &relu_node = match.node;
  const NodeDef &other_node = match.inputs[0].node;
  if (!CheckDtype(other_node)) return ops_to_quantize;

  bool updated =
      UpdateNodeConfigMap(relu_node, GetActConfig(config), ops_to_quantize);
  if (updated) {
    DLOG_INFO(1) << "Quantize other + relu: " << relu_node.name() << " <-- "
                 << other_node.name() << "(" << other_node.op() << ")";
  }
  return ops_to_quantize;
}

NodeConfigMap LocateOther(const NodeMatch &match, const QuantizeConfig &config,
                          std::set<NodeGroup> &node_groups) {
  NodeConfigMap ops_to_quantize;
  const NodeDef &other_node = match.node;
  if (!CheckDtype(other_node)) return ops_to_quantize;

  bool updated =
      UpdateNodeConfigMap(other_node, GetActConfig(config), ops_to_quantize);
  if (updated) {
    DLOG_INFO(1) << "Quantize other: " << other_node.name() << "("
                 << other_node.op() << ")";
  }
  return ops_to_quantize;
}

NodeDef *CreateFixNeuronOp(const string &producer_name,
                           const QuantizeConfig &config, const string &name,
                           GraphDef &input_graph_def) {
  NodeDef *fn_node = input_graph_def.mutable_node()->Add();
  fn_node->set_name(name);
  fn_node->set_op("FixNeuron");
  AddNodeInput(producer_name, fn_node);
  SetNodeAttr("method", config.method, fn_node);
  SetNodeAttr("phase", config.phase, fn_node);
  SetNodeAttr("T", DT_FLOAT, fn_node);
  SetNodeAttr("mode", config.mode, fn_node);
  if (config.mode == QuantizeMode::WEIGHT ||
      config.mode == QuantizeMode::DW_WEIGHT) {
    SetNodeAttr("bit_width", config.weight_bit, fn_node);
  } else if (config.mode == QuantizeMode::ACTIVATION) {
    SetNodeAttr("bit_width", config.activation_bit, fn_node);
  } else {
    LOG(FATAL) << "Invalid mode(" << config.mode << ") for node:" << name;
  }
  SetNodeAttr("output_dir", config.output_dir, fn_node);
  return fn_node;
}

NodeDef *CreateActivationFixNeuronOp(const string &producer_name,
                                     const QuantizeConfig &config,
                                     GraphDef &input_graph_def) {
  auto act_config(config);
  string name = producer_name + "/aquant";
  return CreateFixNeuronOp(producer_name, act_config, name, input_graph_def);
}

NodeDef *CreateWeightFixNeuronOp(const string &producer_name,
                                 const QuantizeConfig &config,
                                 GraphDef &input_graph_def) {
  auto wt_config(config);
  string name = producer_name + "/wquant";
  return CreateFixNeuronOp(producer_name, wt_config, name, input_graph_def);
}

string GenerateDecentCommand(const string graph_path,
                             const std::vector<const NodeDef *> inputs,
                             const std::vector<const NodeDef *> outputs) {
  std::vector<string> input_nodes;
  std::vector<string> output_nodes;
  for (const NodeDef *input : inputs) {
    input_nodes.push_back(input->name());
  }
  for (const NodeDef *output : outputs) {
    output_nodes.push_back(output->name());
  }

  string decent_command = "  decent_q quantize";
  decent_command += " --input_frozen_graph " + graph_path;
  decent_command += " --input_nodes " + str_util::Join(input_nodes, ",");
  decent_command += " --output_nodes " + str_util::Join(output_nodes, ",");
  decent_command += " --input_fn `default|random or customized input_fn`";
  decent_command += " [Other Options]\n";
  return decent_command;
}
}  // namespace

Status QuantizeConfig::FromString(const string config_string) {
  std::vector<string> params = str_util::Split(config_string, ",");
  for (int i = 0; i < params.size() - 1; i += 2) {
    string param = params[i];
    string value = params[i + 1];
    if (param == "phase") {
      phase = QuantizePhase(std::stoi(value));
    } else if (param == "weight_bit") {
      weight_bit = std::stoi(value);
    } else if (param == "activation_bit") {
      activation_bit = std::stoi(value);
    } else if (param == "nodes_bit") {
      std::vector<string> node_bit_params = str_util::Split(value, ":");
      nodes_bit.emplace(node_bit_params[0], std::stoi(node_bit_params[1]));
    } else if (param == "nodes_method") {
      std::vector<string> node_method_params = str_util::Split(value, ":");
      nodes_method.emplace(node_method_params[0],
                           std::stoi(node_method_params[1]));
    } else if (param == "method") {
      method = QuantizeMethod(std::stoi(value));
    } else if (param == "mode") {
      mode = QuantizeMode(std::stoi(value));
    } else if (param == "input_nodes") {
      input_nodes.push_back(value);
    } else if (param == "output_nodes") {
      output_nodes.push_back(value);
    } else if (param == "input_shapes") {
      value = str_util::StringReplace(value, "*", ",", true);
      input_shapes.push_back(value);
    } else if (param == "quant_input_dtypes") {
      quant_input_dtypes.push_back(value);
    } else if (param == "ignore_nodes") {
      ignore_nodes.insert(value);
    } else if (param == "calib_iter") {
      calib_iter = std::stoi(value);
    } else if (param == "output_dir") {
      output_dir = value;
    } else if (param == "align_concat") {
      align_concat = std::stoi(value);
    } else if (param == "align_pool") {
      align_pool = std::stoi(value);
    } else if (param == "adjust_shift_bias") {
      adjust_shift_bias = std::stoi(value);
    } else if (param == "adjust_shift_cut") {
      adjust_shift_cut = std::stoi(value);
    } else if (param == "simulate_dpu") {
      simulate_dpu = std::stoi(value);
    } else if (param == "do_cle") {
      do_cle = std::stoi(value);
    } else if (param == "scale_all_avgpool") {
      scale_all_avgpool = std::stoi(value);
    } else if (param == "replace_relu6") {
      replace_relu6 = std::stoi(value);
    } else if (param == "replace_sigmoid") {
      replace_sigmoid = std::stoi(value);
    } else if (param == "fold_bn_only") {
      fold_bn_only = std::stoi(value);
    } else if (param == "replace_softmax") {
      replace_softmax = std::stoi(value);
    } else {
      return errors::InvalidArgument("Wrong QuantizeConfig Parameter: " +
                                     param);
    }
  }
  return Status::OK();
}

Status GraphQuantizer::_LocateOpsToQuantize(const GraphDef &input_graph_def) {
  typedef NodeConfigMap (*LocationFuncHandle)(
      const NodeMatch &, const QuantizeConfig &, std::set<NodeGroup> &);

  // Lookup table to map pattern_id to function, refer to known_patterns.cc for
  // deatails of pattern_id
  std::map<string, LocationFuncHandle> map_pattern_func({
      {"placeholder", &LocatePlaceholder},
      {"atrous_conv_bias_relu", &LocateAtrousConvBiasRelu},
      {"atrous_conv_bias", &LocateAtrousConvBias},
      {"atrous_conv_relu", &LocateAtrousConvRelu},
      {"atrous_conv", &LocateAtrousConv},
      {"convfc_bias_id_relu", &LocateConvfcBiasIdRelu},
      {"convfc_bias_relu", &LocateConvfcBiasRelu},
      {"convfc_bias", &LocateConvfcBias},
      {"convfc_relu", &LocateConvfcRelu},
      {"convfc", &LocateConvfc},
      {"conv2d_transpose_bias_relu", &LocateConv2dTransposeBiasRelu},
      {"conv2d_transpose_bias", &LocateConv2dTransposeBias},
      {"conv2d_transpose_relu", &LocateConv2dTransposeRelu},
      {"conv2d_transpose", &LocateConv2dTranspose},
      {"keras_conv2d_transpose_bias_relu",
       &LocateConv2dTransposeBiasRelu},  // reuse function of Conv2dTranspose
                                         // for KerasConv2dTranspose
      {"keras_conv2d_transpose_bias", &LocateConv2dTransposeBias},
      {"keras_conv2d_transpose_relu", &LocateConv2dTransposeRelu},
      {"keras_conv2d_transpose", &LocateConv2dTranspose},
      {"conv2d_backprop_input_bias_relu",
       &LocateConv2dTransposeBiasRelu},  // reuse function of Conv2dTranspose
                                         // for Conv2d_backprop_input
      {"conv2d_backprop_input_bias", &LocateConv2dTransposeBias},
      {"conv2d_backprop_input_relu", &LocateConv2dTransposeRelu},
      {"conv2d_backprop_input", &LocateConv2dTranspose},
      {"convfc_bias_leakyrelu", &LocateConvfcBiasLeakyRelu},
      {"convfc_bias_fused_leakyrelu", &LocateConvfcBiasFusedLeakyRelu},
      {"convfc_bias_keras_leakyrelu", &LocateConvfcBiasKerasLeakyRelu},
      {"convfc_leakyrelu", &LocateConvfcLeakyRelu},
      {"convfc_fused_leakyrelu", &LocateConvfcFusedLeakyRelu},
      {"convfc_keras_leakyrelu", &LocateConvfcKerasLeakyRelu},
      {"hard_swish", &LocateHardSwish},
      {"hard_sigmoid", &LocateHardSigmoid},
      {"swish", &LocateSwish},
      {"leakyrelu", &LocateLeakyRelu},
      {"fused_leakyrelu", &LocateFusedLeakyRelu},
      {"keras_leakyrelu", &LocateKerasLeakyRelu},
      {"upsampling", &LocateUpsampling},
      {"resize_bilinear", &LocateResize},
      {"depth_to_space", &LocateDepthToSpace},
      {"tpu_nearest_neighbor_upsampling", &LocateTPUNearestNeighborUpsampling},
      {"batchnorm_relu", &LocateBatchNormRelu},
      {"batchnorm", &LocateBatchNorm},
      {"array_relu", &LocateArrayRelu},
      {"mul_v1", &LocateMul_v1},
      {"mul_v2", &LocateMul_v2},
      {"array", &LocateArray},
      {"avgpool_mul", &LocateAvgpoolMul},
      {"clip_by_value", &LocateClipByValue},
      {"other_relu", &LocateOtherRelu},
      {"other", &LocateOther},
  });

  // Run
  bool did_graph_change;
  do {
    did_graph_change = false;
    for (auto match_id = 0; match_id < _matched_node_patterns.size();
         ++match_id) {
      auto pattern_id = std::get<0>(_matched_node_patterns[match_id]);
      auto pattern_name = get_pattern_name_from_id(pattern_id);
      auto match = std::get<1>(_matched_node_patterns[match_id]);

      std::vector<const NodeDef *> input_nodes =
          get_input_nodes(match, pattern_name);
      std::set<string> input_node_names;
      for (int i = 0; i < input_nodes.size(); i++) {
        if (input_nodes[i]->op() == "Identity") {
          std::vector<string> split_input_names =
              str_util::Split(input_nodes[i]->input(0), ":");
          input_node_names.insert(split_input_names[0]);
        } else {
          input_node_names.insert(input_nodes[i]->name());
        }
      }

      // For other pattern, the input lenght is variant
      if (pattern_name == "other" || pattern_name == "other_relu") {
        std::map<string, const NodeDef *> names_to_nodes;
        MapNamesToNodes(input_graph_def, &names_to_nodes);
        const NodeDef &current_node = match.node;
        for (auto i = 0; i < current_node.input_size() - 1; i++) {
          std::vector<string> split_input_names =
              str_util::Split(current_node.input(i), ":");
          string input_node_name = split_input_names[0];
          if (!names_to_nodes.count(input_node_name)) {
            DLOG_WARNING << "Not found current_node in graph_def, node name: "
                         << input_node_name;
            continue;
          }
          const NodeDef *input_node = names_to_nodes[input_node_name];
          if (input_node->op() == "Identity") {
            input_node_names.insert(input_node->input(0));
          } else {
            input_node_names.insert(input_node->name());
          }
        }
      }

      if (CheckAnyIgnoredNodes(_config.ignore_nodes, match, input_node_names)) {
        DLOG_WARNING << "Found ignored match: ";
        PrintNodeMatch(match, 2);
        continue;
      }

      bool all_inputs_quantized = true;
      string unquantized_input;
      for (auto name : input_node_names) {
        if (!_ops_to_quantize.count(name)) {
          all_inputs_quantized = false;
          unquantized_input = name;
        }
      }

      if ((!all_inputs_quantized) && (!compute_patterns.count(pattern_name))) {
        DLOG_INFO(1)
            << "Found unquantized inputs and skip perform locate func: "
            << pattern_name << " for " << match.node.name()
            << " unquantized input node: " << unquantized_input;
        continue;
      }

      if (!map_pattern_func.count(pattern_name)) {
        LOG(FATAL) << "Invalid pattern name: " << pattern_name;
      }
      DLOG_INFO(1) << "Perform locate func: " << pattern_name;
      // PrintNodeMatch(match, 2);
      NodeConfigMap new_ops_to_quantize =
          (*map_pattern_func[pattern_name])(match, _config, _node_groups);
      if (new_ops_to_quantize.size() > 0) {
        did_graph_change |=
            MergeNodeConfigMap(_ops_to_quantize, new_ops_to_quantize);
      }
    }
  } while (did_graph_change);

  // For compute_patterns, insert fix neuron for unquantized input nodes of
  // compute_patterns to make sure all inputs are quantized.
  // For non-compute_pattern, only insert fix neuron for non-compute_patterns
  // if there inputs are all quantized.
  for (auto match_id = 0; match_id < _matched_node_patterns.size();
       ++match_id) {
    auto pattern_id = std::get<0>(_matched_node_patterns[match_id]);
    auto pattern_name = get_pattern_name_from_id(pattern_id);
    auto match = std::get<1>(_matched_node_patterns[match_id]);
    std::vector<const NodeDef *> input_nodes =
        get_input_nodes(match, pattern_name);
    std::set<string> input_node_names;
    for (int i = 0; i < input_nodes.size(); i++) {
      if (input_nodes[i]->op() == "Identity") {
        std::vector<string> split_input_names =
            str_util::Split(input_nodes[i]->input(0), ":");
        input_node_names.insert(split_input_names[0]);
      } else {
        input_node_names.insert(input_nodes[i]->name());
      }
    }

    // For other pattern, the input lenght is variant
    if (pattern_name == "other" || pattern_name == "other_relu") {
      std::map<string, const NodeDef *> names_to_nodes;
      MapNamesToNodes(input_graph_def, &names_to_nodes);
      const NodeDef &current_node = match.node;
      for (auto i = 0; i < current_node.input_size() - 1; i++) {
        std::vector<string> split_input_names =
            str_util::Split(current_node.input(i), ":");
        string input_node_name = split_input_names[0];
        if (!names_to_nodes.count(input_node_name)) {
          DLOG_WARNING << "Not found current_node in graph_def, node name: "
                       << input_node_name;
          continue;
        }
        const NodeDef *input_node = names_to_nodes[input_node_name];
        if (input_node->op() == "Identity") {
          input_node_names.insert(input_node->input(0));
        } else {
          input_node_names.insert(input_node->name());
        }
      }
    }

    if (CheckAnyIgnoredNodes(_config.ignore_nodes, match, input_node_names)) {
      continue;
    }

    // For concat nodes with constant inputs, treat them as constant activation
    // and add quantization
    if (pattern_name == "other" && match.node.op() == "ConcatV2") {
      const NodeDef &concat_node = match.node;
      for (auto i = 0; i < concat_node.input_size() - 1; i++) {
        if (!_ops_to_quantize.count(concat_node.input(i))) {
          DLOG_INFO(1) << "Add quantize op for unquantized input node: "
                       << concat_node.input(i)
                       << " of node: " << match.node.name()
                       << "(op: " << match.node.op()
                       << ", pattern: " << pattern_name << ")";
          _ops_to_quantize.insert(
              std::make_pair(concat_node.input(i), GetActConfig(_config)));
        }
      }
    }

    bool all_inputs_quantized = true;
    string unquantized_input;
    for (auto name : input_node_names) {
      if (!_ops_to_quantize.count(name)) {
        all_inputs_quantized = false;
        unquantized_input = name;
      }
    }

    if (all_inputs_quantized) continue;

    if (!compute_patterns.count(pattern_name)) {
      DLOG_INFO(1)
          << "Node " << match.node.name() << "(" << match.node.op()
          << ") cannot be quantized, because it has unquantized input node: "
          << unquantized_input;
      continue;
    } else {
      for (auto name : input_node_names) {
        if (!_ops_to_quantize.count(name)) {
          const NodeDef &current_node = match.node;
          if (CheckDtype(current_node)) {
            DLOG_INFO(1) << "Add quantize op for unquantized input node: "
                         << name << " of node: " << match.node.name()
                         << "(op: " << match.node.op()
                         << ", pattern: " << pattern_name << ")";
            _ops_to_quantize.insert(
                std::make_pair(name, GetActConfig(_config)));
          } else {
            DLOG_INFO(1) << "dytpe is not float, so skip add quantize op for "
                            "unquantized input node: "
                         << name << " of node: " << match.node.name()
                         << "(op: " << match.node.op()
                         << ", pattern: " << pattern_name << ")";
          }
        }
      }
    }
  }

  // Save node groups information
  SaveNodeGroupsToFile(_node_groups, _config.output_dir);

  return Status::OK();
}

Status GraphQuantizer::_MatchQuantizedNodeName(const GraphDef &input_graph_def,
                                               const std::string &node_name,
                                               std::string &matched_node_name) {
  // specified node name exactly equal node name in _ops_to_quantize
  if (_ops_to_quantize.count(node_name)) {
    DLOG_INFO(1) << "Found specified node " << node_name
                 << " in ops_to_quantize";
    matched_node_name = node_name;
    return Status::OK();
  }
  // if specified node name in _node_groups then return weigths name
  for (const auto &node_group : _node_groups) {
    if (node_name == node_group[0]) {
      // node_group[0] is compute node; node_group[2] is output node;
      // node_group[3] is weighs node
      matched_node_name =
          node_group[3] == "NULL" ? node_group[2] : node_group[3];
      DLOG_INFO(1) << "Found specified node " << node_name
                   << " in node_groups, map it to the corresponding node "
                   << matched_node_name;
    }
  }
  // TODO: if name of node in _node_groups endwith _Fold, then it is
  // generated by foldBNtrain
  //
  // TODO: if specified node is conv, and conv/conv_biasadd is the output of
  // this node_group, and want to specify out aquant bit_width or method
  //
  // TODO: Batchnorm without conv

  return Status::OK();
}

Status GraphQuantizer::_ModifyFixNeuronConfig(const GraphDef &input_graph_def) {
  for (auto &item : _config.nodes_bit) {
    const string node_name = item.first;
    string matched_node_name = item.first;
    TF_RETURN_IF_ERROR(
        _MatchQuantizedNodeName(input_graph_def, node_name, matched_node_name));
    if (CheckSpecifiedNodeName(input_graph_def, _ops_to_quantize,
                               matched_node_name)) {
      _ops_to_quantize[matched_node_name].weight_bit = item.second;
      _ops_to_quantize[matched_node_name].activation_bit = item.second;
    }
  }

  for (auto &item : _config.nodes_method) {
    const string node_name = item.first;
    string matched_node_name = item.first;
    TF_RETURN_IF_ERROR(
        _MatchQuantizedNodeName(input_graph_def, node_name, matched_node_name));
    if (CheckSpecifiedNodeName(input_graph_def, _ops_to_quantize,
                               matched_node_name)) {
      if (item.second == 2 && _ops_to_quantize[matched_node_name].mode != 2) {
        item.second = 1;
      }
      QuantizeMethod method = static_cast<QuantizeMethod>(item.second);
      _ops_to_quantize[matched_node_name].method = method;
    }
  }
  return Status::OK();
}

Status GraphQuantizer::_InsertFixNeuronOps(const GraphDef &input_graph_def,
                                           GraphDef &output_graph_def) {
  output_graph_def.Clear();
  std::map<string, string> inputs_to_rename;
  std::unordered_set<string> nodes_to_ignore;
  GraphDef fn_inserted_graph_def;
  for (const NodeDef &node : input_graph_def.node()) {
    NodeDef *new_node = fn_inserted_graph_def.mutable_node()->Add();
    *new_node = node;
    if (_ops_to_quantize.count(node.name())) {
      auto config = _ops_to_quantize.at(node.name());
      DLOG_INFO(1) << "Insert FixNeuron: " << node.name() << "(" << node.op()
                   << ") Quantize_mode: " << config.mode;
      NodeDef *new_fn_node;
      if (config.mode == QuantizeMode::WEIGHT ||
          config.mode == QuantizeMode::DW_WEIGHT) {
        new_fn_node = CreateWeightFixNeuronOp(new_node->name(), config,
                                              fn_inserted_graph_def);
      } else {
        new_fn_node = CreateActivationFixNeuronOp(new_node->name(), config,
                                                  fn_inserted_graph_def);
      }
      inputs_to_rename[new_node->name()] = new_fn_node->name();
      nodes_to_ignore.insert(new_fn_node->name());
    }
  }
  TF_RETURN_IF_ERROR(RenameNodeInputs(fn_inserted_graph_def, inputs_to_rename,
                                      nodes_to_ignore, &output_graph_def));
  SaveGraphForDebugging(output_graph_def, "fn_inserted.pb", _config.output_dir);
  return Status::OK();
}

Status GraphQuantizer::_FreezeFixNeuronOps(const GraphDef &input_graph_def,
                                           GraphDef &output_graph_def) {
  output_graph_def.Clear();
  std::unordered_map<string, std::vector<int>> quantize_info_map;
  LoadQuantizeInfoFromFile(input_graph_def, &quantize_info_map,
                           _config.output_dir);

  for (const NodeDef &node : input_graph_def.node()) {
    NodeDef *new_node = output_graph_def.mutable_node()->Add();
    new_node->CopyFrom(node);
    if (node.op() == "FixNeuron") {
      if (_ops_to_quantize.count(node.input(0))) {
        LOG(FATAL) << "Found invalid fix_neuron: " << node.name();
      }
      if (quantize_info_map.count(node.name())) {
        int p = quantize_info_map.at(node.name())[1];

        // Check all zeros
        if (p == 127) {
          DLOG_WARNING
              << "Node " << node.input(0)
              << "'s output values are all zeros. This may cause error for DPU "
                 "compiler, please check your float model.";
        }

        SetNodeAttr("quantize_pos", quantize_info_map.at(node.name())[1],
                    new_node);
        SetNodeAttr("phase", 1, new_node);
        _node_to_fn.insert(
            std::pair<string, string>(node.input(0), node.name()));
        DLOG_INFO(1) << "Freeze FixNeuron: " << node.name() << "(" << node.op()
                     << ")";
      } else {
        LOG(FATAL) << "[QUANTIZE_TF1_MISSING_QUANTIZE_INFO]"
                   << "[Not found]"
                   << " Cannot find quantize info for op: " << node.name()
                   << ", please check your quantize info file.";
      }
    }
  }
  SaveGraphForDebugging(output_graph_def, "freeze_fix_neuron_ops.pb",
                        _config.output_dir);
  return Status::OK();
}

// Align quantize_pos for concat's input nodes, backtracing through
// concat transparent_ops
Status AlignConcat(const GraphDef &input_graph_def, GraphDef &output_graph_def,
                   const QuantizeConfig &config) {
  GraphDef current_graph_def;

  current_graph_def = input_graph_def;
  std::map<string, const NodeDef *> names_to_nodes;
  MapNamesToNodes(current_graph_def, &names_to_nodes);
  std::map<string, int> nodes_to_align;
  std::set<string> concat_transparent_ops{"Reshape", "ExpandDims"};

  for (auto i = 0; i < current_graph_def.node_size(); i++) {
    if (current_graph_def.node(i).op() == "FixNeuron") {
      const NodeDef &fn_node = current_graph_def.node(i);
      const NodeDef cur_node = *(names_to_nodes[fn_node.input(0)]);
      if (cur_node.op() == "Concat" || cur_node.op() == "ConcatV2") {
        bool need_align = false;
        if (config.align_concat == 0) {
          need_align = true;
        } else if (config.align_concat == 1 &&
                   (std::find(config.output_nodes.begin(),
                              config.output_nodes.end(),
                              cur_node.name()) != config.output_nodes.end())) {
          need_align = true;
        } else if (config.align_concat == 2) {
          need_align = false;
        }

        if (need_align) {
          int concat_pos = fn_node.attr().at("quantize_pos").i();
          for (auto j = 0; j < cur_node.input_size() - 1; j++) {
            const NodeDef *cur_input = names_to_nodes[cur_node.input(j)];
            while (cur_input->op() == "FixNeuron" ||
                   concat_transparent_ops.count(cur_input->op())) {
              if (cur_input->op() == "FixNeuron") {
                int pos = cur_input->attr().at("quantize_pos").i();
                if (pos != concat_pos) {
                  DLOG_INFO(1)
                      << "Align quantize pos for concat node "
                      << cur_node.name() << " <-- input: " << cur_input->name()
                      << "(from " << pos << " to " << concat_pos << ").";
                  nodes_to_align[cur_input->name()] = concat_pos;
                }
              }
              cur_input = names_to_nodes[cur_input->input(0)];
            }
          }
        }
      }
    }
  }

  // Do align
  output_graph_def = current_graph_def;
  for (auto i = 0; i < output_graph_def.node_size(); i++) {
    NodeDef *cur_node = output_graph_def.mutable_node(i);
    if (nodes_to_align.count(cur_node->name())) {
      SetNodeAttr("quantize_pos", nodes_to_align[cur_node->name()], cur_node);
    }
  }

  return Status::OK();
}

// Align quantize_pos for maxpool avgpool input nodes
Status AlignPool(const GraphDef &input_graph_def, GraphDef &output_graph_def,
                 const QuantizeConfig &config) {
  GraphDef current_graph_def;

  current_graph_def = input_graph_def;
  std::map<string, const NodeDef *> names_to_nodes;
  MapNamesToNodes(current_graph_def, &names_to_nodes);
  std::map<string, int> nodes_to_align;

  for (auto i = 0; i < current_graph_def.node_size(); i++) {
    if (current_graph_def.node(i).op() == "FixNeuron") {
      const NodeDef &fn_node = current_graph_def.node(i);
      NodeDef cur_node = *(names_to_nodes[fn_node.input(0)]);
      if (cur_node.op() == "Mul") {
        const NodeDef cur_input_node = *(names_to_nodes[cur_node.input(0)]);
        if (cur_input_node.op() != "AvgPool") {
          continue;
        }
        const NodeDef in_pos_node = *(names_to_nodes[cur_input_node.input(0)]);
        cur_node = cur_input_node;
      } else if (cur_node.op() == "MaxPool") {
      } else {
        continue;
      }

      bool need_align = false;
      if (config.align_pool == 0) {
        need_align = true;
      } else if (config.align_pool == 1 &&
                 (std::find(config.output_nodes.begin(),
                            config.output_nodes.end(),
                            cur_node.name()) != config.output_nodes.end())) {
        need_align = true;
      } else if (config.align_pool == 2) {
        need_align = false;
      }

      if (need_align) {
        int pool_pos = fn_node.attr().at("quantize_pos").i();
        const NodeDef *cur_input = names_to_nodes[cur_node.input(0)];
        if (cur_input->op() == "FixNeuron") {
          int pos = cur_input->attr().at("quantize_pos").i();
          if (pos != pool_pos) {
            DLOG_INFO(1) << "Align quantize pos for pool node "
                         << cur_node.name()
                         << " <-- input: " << cur_input->name() << "(from "
                         << pos << " to " << pool_pos << ").";
            nodes_to_align[cur_input->name()] = pool_pos;
          }
        }
      }
    }
  }

  // Do align
  output_graph_def = current_graph_def;
  for (auto i = 0; i < output_graph_def.node_size(); i++) {
    NodeDef *cur_node = output_graph_def.mutable_node(i);
    if (nodes_to_align.count(cur_node->name())) {
      SetNodeAttr("quantize_pos", nodes_to_align[cur_node->name()], cur_node);
    }
  }

  return Status::OK();
}

// Helper functions for AdjustShiftBias
// Calculate the integer decomposition
std::tuple<bool, std::vector<std::uint32_t>> IntegerDecomposition(
    const std::uint32_t &input, const std::uint32_t &num) {
  std::vector<std::uint32_t> ret;
  auto n = input;
  while (n % 2 == 0) {
    ret.push_back(2);
    n /= 2;
  }
  std::uint32_t f = 3;
  while (f * f <= n) {
    if (n % f == 0) {
      ret.push_back(f);
      n /= f;
    } else {
      f += 2;
    }
  }
  if (1 != n) {
    ret.push_back(n);
  }
  bool if_enough = ret.size() >= num;
  return std::make_tuple(if_enough, ret);
}

// Calculate the integer composition
std::tuple<bool, std::vector<std::uint32_t>> IntegerComposition(
    const std::vector<std::uint32_t> &integers,
    const std::vector<std::tuple<std::uint32_t, std::uint32_t>> &bounds) {
  // pre-check all the bounds
  std::vector<std::uint32_t> all_integers(integers);
  std::sort(all_integers.begin(), all_integers.end());
  std::uint32_t all_integer_mul = 1;
  std::for_each(
      all_integers.begin(), all_integers.end(),
      [&all_integer_mul](const std::uint32_t i) { all_integer_mul *= i; });
  std::uint32_t ret_size = bounds.size();
  std::vector<std::uint32_t> ret(ret_size, 1);
  for (auto idx = 0U; idx < ret_size; idx++) {
    ret[idx] = std::get<0>(bounds[idx]);
  }
  bool one_round_failed = true;
  auto integer_it = all_integers.begin();
  std::uint32_t idx = 0U;
  while (all_integers.end() != integer_it) {
    auto top = std::get<1>(bounds[idx]);
    if ((ret[idx] < top) && (ret[idx] * (*integer_it) < top)) {
      ret[idx] *= (*integer_it);
      integer_it++;
      one_round_failed = false;
    }
    idx++;
    // top up the idx
    if (!(idx < ret_size)) {
      if (one_round_failed) {
        break;
      } else {
        one_round_failed = true;
        idx = 0U;
      }
    }
  }
  bool if_success = (integer_it == all_integers.end());
  return std::make_tuple(if_success, ret);
}

// Adjust shift_bias for convolution like ops
// DPU constraints for shift_bias:
//  shift_bias = w + i - b
//  mode 1:
//    shift_bias >= min(0, -( 24 - (8 + shift_cut)))
//    shfit_bias <= 16
//  mode 2:
//    shift_bias >= min(0, -( 24 - (8 + ceil(log2(k_h * k_w * c_i)))))
//    shift_bais <=16
Status AdjustShiftBias(const GraphDef &input_graph_def,
                       GraphDef &output_graph_def, const QuantizeConfig &config,
                       const std::unordered_map<string, string> &node_to_fn) {
  GraphDef current_graph_def, processed_graph_def;

  current_graph_def = input_graph_def;
  std::map<string, const NodeDef *> names_to_nodes;
  MapNamesToNodes(current_graph_def, &names_to_nodes);
  std::map<string, int> nodes_to_adjust;

  if (config.adjust_shift_bias == 0) {
    output_graph_def = current_graph_def;
    return Status::OK();
  }

  std::set<NodeGroup> node_groups;
  LoadNodeGroupsFromFile(node_groups, config.output_dir);
  for (auto g : node_groups) {
    DLOG_INFO(1) << "Found group: " << g[0] << " " << g[1] << " " << g[2] << " "
                 << g[3] << " " << g[4];
    // Skip group without bias
    if (g[4] == "NULL") {
      continue;
    }

    // Get quantize positions
    std::vector<int> p(5);
    bool skip = false;
    for (int i = 1; i <= 4; i++) {
      if (!node_to_fn.count(g[i])) {
        DLOG_WARNING << "Skip adjust shift cut for " << g[0]
                     << " because the quantize info of " << g[i]
                     << " is missing.";
        skip = true;
        break;
      }
      p[i] =
          names_to_nodes.at(node_to_fn.at(g[i]))->attr().at("quantize_pos").i();
    }
    if (skip) {
      continue;
    }
    DLOG_INFO(1) << "pi: " << p[1] << " po: " << p[2] << " pw: " << p[3]
                 << " pb: " << p[4];

    // Default min, max
    int min = 0;
    int max = 0;

    if (config.adjust_shift_bias == 1) {
      int shift_cut = p[3] + p[1] - p[2];
      min = std::min(0, -(24 - (8 + shift_cut)));
      max = 16;
    } else if (config.adjust_shift_bias == 2) {
      // Get kh, kw, ic
      int kh = 0;
      int kw = 0;
      int ic = 0;
      const NodeDef *compute_node = names_to_nodes.at(g[0]);
      const NodeDef *weight_node = names_to_nodes.at(g[3]);
      if (weight_node->op() == "Const") {
        Tensor weight = GetNodeTensorAttr(*weight_node, "value");
        DLOG_INFO(1) << "shape: " << weight.shape();
        if (compute_node->op() == "Conv2D" ||
            compute_node->op() == "DepthwiseConv2d" ||
            compute_node->op() == "DepthwiseConv2dNative" ||
            compute_node->op() == "Conv2DBackpropInput") {
          kh = weight.dim_size(0);
          kw = weight.dim_size(1);
          ic = weight.dim_size(2);
        } else if (compute_node->op() == "MatMul") {
          // decompose the input colum into {height, width, channel} 3
          // dimensions,
          // size limitations are {16, 16, 4096}
          auto tuple_size_decomposition =
              IntegerDecomposition(weight.dim_size(0), 3);
          if (!std::get<0>(tuple_size_decomposition)) {
            DLOG_WARNING << "Skip shift bias check for MatMul node: "
                         << compute_node->name() << " because its size "
                         << weight.shape() << " exceed DPU limitations.";
            continue;
          }
          auto tuple_size_composition = IntegerComposition(
              std::get<1>(tuple_size_decomposition),
              {std::make_tuple(1, 16), std::make_tuple(1, 16),
               std::make_tuple(1, 4096)});
          if (!std::get<0>(tuple_size_composition)) {
            DLOG_WARNING << "Skip shift bias check for MatMul node: "
                         << compute_node->name() << " because its size "
                         << weight.shape() << " exceed DPU limitations.";
            continue;
          }

          kh = std::get<1>(tuple_size_composition)[0];
          kw = std::get<1>(tuple_size_composition)[1];
          ic = std::get<1>(tuple_size_composition)[2];
        } else {
          DLOG_WARNING << "Skip shift bias check for compute node: "
                       << compute_node->name()
                       << " with type: " << compute_node->op();
          continue;
        }
      }
      DLOG_INFO(1) << "kh: " << kh << " kw: " << kw << " ic: " << ic;

      // Calc min, max
      min = std::min(0, -(24 - (8 + int(std::ceil(std::log2(kh * kw * ic))))));
      max = 16;
    }
    DLOG_INFO(1) << "min: " << min << " max: " << max;

    int shift_bias = p[3] + p[1] - p[4];
    DLOG_INFO(1) << "shift bias: " << shift_bias;

    // Adjust bias pos to satisfy shift_cut constraints.
    if (shift_bias < min) {
      nodes_to_adjust[node_to_fn.at(g[4])] = p[3] + p[1] - min;
      DLOG_WARNING << "Shift bias of node " << g[0] << " is " << shift_bias
                   << ". It exceed range [" << min << ", " << max
                   << "], modify quantize pos from " << p[4] << " to "
                   << p[3] + p[1] - min;
    } else if (shift_bias > max) {
      nodes_to_adjust[node_to_fn.at(g[4])] = p[3] + p[1] - max;
      DLOG_WARNING << "Shift bias of node " << g[0] << " is " << shift_bias
                   << ". It exceed range [" << min << ", " << max
                   << "], modify quantize pos from " << p[4] << " to "
                   << p[3] + p[1] - max;
    }
  }

  // Do Adjust
  output_graph_def = current_graph_def;
  for (auto i = 0; i < output_graph_def.node_size(); i++) {
    NodeDef *cur_node = output_graph_def.mutable_node(i);
    if (nodes_to_adjust.count(cur_node->name())) {
      SetNodeAttr("quantize_pos", nodes_to_adjust[cur_node->name()], cur_node);
    }
  }

  return Status::OK();
}

// Adjust shift_cut for convolution like ops
// DPU constraints for shift_cut:
//  shift_cut = w + i - o
//  shift_cut >= 0
//  shfit_cut <= 16
Status AdjustShiftCut(const GraphDef &input_graph_def,
                      GraphDef &output_graph_def, const QuantizeConfig &config,
                      const std::unordered_map<string, string> &node_to_fn) {
  GraphDef current_graph_def, processed_graph_def;

  current_graph_def = input_graph_def;
  std::map<string, const NodeDef *> names_to_nodes;
  MapNamesToNodes(current_graph_def, &names_to_nodes);
  std::map<string, int> nodes_to_adjust;

  if (config.adjust_shift_cut == 0) {
    output_graph_def = current_graph_def;
    return Status::OK();
  }

  std::set<NodeGroup> node_groups;
  LoadNodeGroupsFromFile(node_groups, config.output_dir);
  for (auto g : node_groups) {
    DLOG_INFO(1) << "Found group: " << g[0] << " " << g[1] << " " << g[2] << " "
                 << g[3] << " " << g[4];
    int min = 0;
    int max = 16;
    DLOG_INFO(1) << "min: " << min << " max: " << max;

    // Get quantize positions
    std::vector<int> p(4);
    bool skip = false;
    for (int i = 1; i <= 3; i++) {
      if (!node_to_fn.count(g[i])) {
        DLOG_WARNING << "Skip adjust shift cut for " << g[0]
                     << " because the quantize info of " << g[i]
                     << " is missing.";
        skip = true;
        break;
      }
      p[i] =
          names_to_nodes.at(node_to_fn.at(g[i]))->attr().at("quantize_pos").i();
    }
    if (skip) {
      continue;
    }
    DLOG_INFO(1) << "pi: " << p[1] << " po: " << p[2] << " pw: " << p[3];

    int shift_cut = p[3] + p[1] - p[2];
    DLOG_INFO(1) << "shift cut: " << shift_cut;

    // Adjust weight pos to satisfy shift_cut constraints.
    if (shift_cut < min) {
      nodes_to_adjust[node_to_fn.at(g[3])] = min + p[2] - p[1];
      DLOG_WARNING << "Shift cut of node " << g[0] << " is " << shift_cut
                   << ". It exceed range [" << min << ", " << max
                   << "], modify quantize pos from " << p[3] << " to "
                   << min + p[2] - p[1];
    } else if (shift_cut > max) {
      nodes_to_adjust[node_to_fn.at(g[3])] = max + p[2] - p[1];
      DLOG_WARNING << "Shift cut of node " << g[0] << " is " << shift_cut
                   << ". It exceed range [" << min << ", " << max
                   << "], modify quantize pos from " << p[3] << " to "
                   << max + p[2] - p[1];
    }
  }

  // Do Adjust
  output_graph_def = current_graph_def;
  for (auto i = 0; i < output_graph_def.node_size(); i++) {
    NodeDef *cur_node = output_graph_def.mutable_node(i);
    if (nodes_to_adjust.count(cur_node->name())) {
      SetNodeAttr("quantize_pos", nodes_to_adjust[cur_node->name()], cur_node);
    }
  }

  return Status::OK();
}

// for hard_sigmoid pattern quantize_pos, need input pos >= 0, output pos >= 7
Status AdjustHardSigmoidPos(const GraphDef &input_graph_def,
                            GraphDef &output_graph_def) {
  GraphDef current_graph_def;

  current_graph_def = input_graph_def;
  std::map<string, const NodeDef *> names_to_nodes;
  MapNamesToNodes(current_graph_def, &names_to_nodes);
  std::map<string, int> nodes_to_adjust;

  for (auto i = 0; i < current_graph_def.node_size(); i++) {
    if (current_graph_def.node(i).op() == "FixNeuron") {
      const NodeDef &out_fn_node = current_graph_def.node(i);
      const NodeDef &vitis_mul_node = *(names_to_nodes[out_fn_node.input(0)]);
      const string &viti_mul_name = vitis_mul_node.name();
      const string &hardsigmoid_mul_posfix = "vitis_hard_sigmoid_mul";
      // if vitis hardsigmoid mul node end with "vitis_hard_sigmoid_mul"
      bool is_vitis_scale_node =
          vitis_mul_node.op() == "Mul" &&
          std::mismatch(hardsigmoid_mul_posfix.rbegin(),
                        hardsigmoid_mul_posfix.rend(), viti_mul_name.rbegin())
                  .first == hardsigmoid_mul_posfix.rend();
      if (is_vitis_scale_node) {
        const NodeDef &mul_1_6_node =
            *(names_to_nodes[vitis_mul_node.input(0)]);
        const NodeDef &relu6_node = *(names_to_nodes[mul_1_6_node.input(0)]);
        const NodeDef &add_3_node = *(names_to_nodes[relu6_node.input(0)]);
        const NodeDef &in_fn_node = *(names_to_nodes[add_3_node.input(0)]);
        int in_pos = in_fn_node.attr().at("quantize_pos").i();
        int out_pos = out_fn_node.attr().at("quantize_pos").i();
        if (in_pos < 0) {
          DLOG_WARNING << "Adjust input quantize_pos of hard sigmoid node "
                       << mul_1_6_node.name() << "  from " << in_pos
                       << " to 0.";
          nodes_to_adjust[in_fn_node.name()] = 0;
        }
        if (out_pos < 7) {
          DLOG_WARNING << "Adjust output quantize_pos of hard sigmoid node "
                       << mul_1_6_node.name() << "  from " << out_pos
                       << " to 7.";
          nodes_to_adjust[out_fn_node.name()] = 7;
        }
      }
    }
  }

  output_graph_def = current_graph_def;
  for (auto i = 0; i < output_graph_def.node_size(); i++) {
    NodeDef *cur_node = output_graph_def.mutable_node(i);
    if (nodes_to_adjust.count(cur_node->name())) {
      SetNodeAttr("quantize_pos", nodes_to_adjust[cur_node->name()], cur_node);
    }
  }
  return Status::OK();
}

Status GraphQuantizer::_AdjustQuantizePos(const GraphDef &input_graph_def,
                                          GraphDef &output_graph_def) {
  GraphDef current_graph_def, processed_graph_def;

  current_graph_def = input_graph_def;
  TF_RETURN_IF_ERROR(
      AlignConcat(current_graph_def, processed_graph_def, _config));

  current_graph_def = processed_graph_def;
  TF_RETURN_IF_ERROR(
      AlignPool(current_graph_def, processed_graph_def, _config));

  current_graph_def = processed_graph_def;
  TF_RETURN_IF_ERROR(AdjustShiftCut(current_graph_def, processed_graph_def,
                                    _config, _node_to_fn));

  current_graph_def = processed_graph_def;
  TF_RETURN_IF_ERROR(AdjustShiftBias(current_graph_def, processed_graph_def,
                                     _config, _node_to_fn));

  current_graph_def = processed_graph_def;
  TF_RETURN_IF_ERROR(
      AdjustHardSigmoidPos(current_graph_def, processed_graph_def));

  SaveGraphForDebugging(processed_graph_def, "adjust_quantize_pos.pb",
                        _config.output_dir);
  output_graph_def = processed_graph_def;
  return Status::OK();
}

Status GraphQuantizer::CheckGraph(const GraphDef &input_graph_def,
                                  const string graph_path) {
  TF_RETURN_IF_ERROR(IsGraphValid(input_graph_def));
  std::map<string, int> op_count = GetOpCount(input_graph_def);
  std::cout << std::endl;

  std::vector<const NodeDef *> inputs = FindInputs(input_graph_def);
  std::vector<const NodeDef *> outputs = FindOutputs(input_graph_def);
  return Status::OK();
}

Status GraphQuantizer::ReplaceSigmoidWithHardSigmoid(
    const GraphDef &input_graph_def, GraphDef *output_graph_def) {
  GraphDef current_graph_def, processed_graph_def;
  output_graph_def->Clear();
  // GraphDef current_graph_def, processed_graph_def;
  std::map<string, string> inputs_to_rename;
  std::unordered_set<string> nodes_to_ignore;

  current_graph_def = input_graph_def;
  TF_RETURN_IF_ERROR(ReplaceMatchingOpTypes(
      current_graph_def,  // clang-format off
      {"Sigmoid",
      },  // clang-format on
      [&inputs_to_rename, &nodes_to_ignore](
          const NodeMatch &match, const std::set<string> &input_nodes,
          const std::set<string> &output_nodes,
          std::vector<NodeDef> *new_nodes) {
        const NodeDef &sigmoid_node = match.node;

        DLOG_WARNING << "Replace Sigmoid node (" << sigmoid_node.name()
                     << ") with HardSigmoid ";

        NodeDef offset_node;
        offset_node.set_op("Const");
        offset_node.set_name(sigmoid_node.name());
        SetNodeAttr("dtype", DT_FLOAT, &offset_node);
        Tensor offset_tensor(DT_FLOAT, {1});
        offset_tensor.flat<float>()(0) = 3.f;
        SetNodeTensorAttr<float>("value", offset_tensor, &offset_node);
        new_nodes->push_back(offset_node);

        NodeDef add_node;
        add_node.set_op("Add");
        add_node.set_name(sigmoid_node.name() + "/h_sigmoid/add");
        SetNodeAttr("T", DT_FLOAT, &add_node);
        AddNodeInput(sigmoid_node.input(0), &add_node);
        AddNodeInput(offset_node.name(), &add_node);
        new_nodes->push_back(add_node);

        NodeDef relu6_node;
        relu6_node.set_name(sigmoid_node.name() + "/h_sigmoid/relu6");
        relu6_node.set_op("Relu6");
        AddNodeInput(add_node.name(), &relu6_node);
        SetNodeAttr("T", DT_FLOAT, &relu6_node);
        new_nodes->push_back(relu6_node);

        NodeDef scale_node;
        scale_node.set_op("Const");
        scale_node.set_name(sigmoid_node.name() + "/h_sigmoid/scale");
        SetNodeAttr("dtype", DT_FLOAT, &scale_node);
        Tensor scale_tensor(DT_FLOAT, {1});
        scale_tensor.flat<float>()(0) = 1.f / 6;
        SetNodeTensorAttr<float>("value", scale_tensor, &scale_node);
        new_nodes->push_back(scale_node);

        NodeDef mul_node;
        mul_node.set_op("Mul");
        mul_node.set_name(sigmoid_node.name() + "/h_sigmoid/mul");
        SetNodeAttr("T", DT_FLOAT, &mul_node);
        AddNodeInput(relu6_node.name(), &mul_node);
        AddNodeInput(scale_node.name(), &mul_node);
        new_nodes->push_back(mul_node);

        inputs_to_rename[sigmoid_node.name()] = mul_node.name();
        nodes_to_ignore.insert(add_node.name());
        return Status::OK();
      },
      {}, &processed_graph_def));
  TF_RETURN_IF_ERROR(RenameNodeInputs(processed_graph_def, inputs_to_rename,
                                      nodes_to_ignore, output_graph_def));
  return Status::OK();
}

Status GraphQuantizer::ConvertConstantsToVariables(
    const GraphDef &input_graph_def, GraphDef &output_graph_def) {
  GraphDef current_graph_def, processed_graph_def;
  current_graph_def = input_graph_def;

  // Convert constants to variables
  TF_RETURN_IF_ERROR(
      _ConvertConstantsToVariables(current_graph_def, &processed_graph_def));
  SaveGraphForDebugging(processed_graph_def, "convert_to_variables.pb",
                        _config.output_dir);

  output_graph_def = processed_graph_def;
  return Status::OK();
}

Status GraphQuantizer::CreateOptimizedGraph(const GraphDef &input_graph_def,
                                            GraphDef &output_graph_def) {
  output_graph_def.Clear();
  GraphDef current_graph_def, processed_graph_def;

  current_graph_def = input_graph_def;
  processed_graph_def = input_graph_def;
  // Remove Identity|CheckNumerics
  TransformFuncContext context_remove_nodes;
  context_remove_nodes.input_names = _config.input_nodes;
  context_remove_nodes.output_names = _config.output_nodes;
  context_remove_nodes.params.insert(
      std::pair<string, std::vector<string>>({"op", {string("Identity")}}));
  context_remove_nodes.params.insert(std::pair<string, std::vector<string>>(
      {"op", {string("CheckNumerics")}}));
  TF_RETURN_IF_ERROR(RemoveNodes(current_graph_def, context_remove_nodes,
                                 &processed_graph_def));
  // remove identityN
  current_graph_def = processed_graph_def;
  TF_RETURN_IF_ERROR(
      RemoveIdentityNNode(current_graph_def, &processed_graph_def));
  SaveGraphForDebugging(processed_graph_def, "remove_identityN_nodes.pb",
                        _config.output_dir);

  if (!_config.fold_bn_only) {
    // Separate shared constants
    current_graph_def = processed_graph_def;
    TF_RETURN_IF_ERROR(
        SeparateSharedConstants(current_graph_def, &processed_graph_def));
    SaveGraphForDebugging(processed_graph_def, "separate_shared_constants.pb",
                          _config.output_dir);

    // replace relu min6 with relu6
    current_graph_def = processed_graph_def;
    TF_RETURN_IF_ERROR(
        ReplaceReluMin6WithRelu6(current_graph_def, &processed_graph_def));
    SaveGraphForDebugging(processed_graph_def,
                          "replace_relu_min6_with_relu6.pb",
                          _config.output_dir);

    // adjust hard swish compute order if the pattern is
    // add + relu6 + mul_x + mul_1/6
    current_graph_def = processed_graph_def;
    TF_RETURN_IF_ERROR(
        AdjustHardSwishComputeOrder(current_graph_def, &processed_graph_def));
    SaveGraphForDebugging(processed_graph_def,
                          "adjust_hard_swish_compute_order.pb",
                          _config.output_dir);
  }

  // Fold constants
  // current_graph_def = processed_graph_def;
  // TransformFuncContext context_fold_constants;
  // context_fold_constants.input_names = _config.input_nodes;
  // context_fold_constants.output_names = _config.output_nodes;
  // TF_RETURN_IF_ERROR(FoldConstants(current_graph_def, context_fold_constants,
  //&processed_graph_def));
  // SaveGraphForDebugging(processed_graph_def, "fold_constants.pb",
  //_config.output_dir);

  // Update Old Batchnorms
  current_graph_def = processed_graph_def;
  TF_RETURN_IF_ERROR(
      UpdateOldBatchNorms(current_graph_def, &processed_graph_def));
  SaveGraphForDebugging(processed_graph_def, "update_old_batchnorms.pb",
                        _config.output_dir);

  // Fold Batchnorms
  current_graph_def = processed_graph_def;
  bool is_training = false;
  TF_RETURN_IF_ERROR(
      FoldBatchNorms(current_graph_def, &processed_graph_def, is_training));
  SaveGraphForDebugging(processed_graph_def, "fold_batchnorms.pb",
                        _config.output_dir);

  // Fold Conv+Mul
  current_graph_def = processed_graph_def;
  TF_RETURN_IF_ERROR(
      FoldConvMulInference(current_graph_def, &processed_graph_def));
  SaveGraphForDebugging(processed_graph_def, "fold_convmul.pb",
                        _config.output_dir);

  current_graph_def = processed_graph_def;
  // cross layers equalization
  if (_config.do_cle == 1) {
    TF_RETURN_IF_ERROR(
        CrossLayersEqualization(current_graph_def, processed_graph_def));
  }

  // Simulate DPU
  if (_config.simulate_dpu == 1 && !_config.fold_bn_only) {
    current_graph_def = processed_graph_def;
    TF_RETURN_IF_ERROR(SimulateDPU(
        current_graph_def, &processed_graph_def, _config.scale_all_avgpool,
        _config.replace_softmax, _config.replace_sigmoid));
    SaveGraphForDebugging(processed_graph_def, "simulate_dpu.pb",
                          _config.output_dir);
  }

  output_graph_def = processed_graph_def;
  SaveGraphForDebugging(output_graph_def, "optimized.pb", _config.output_dir);
  return Status::OK();
}

Status GraphQuantizer::CrossLayersEqualization(const GraphDef &input_graph_def,
                                               GraphDef &output_graph_def) {
  output_graph_def.Clear();
  GraphDef current_graph_def, processed_graph_def;

  current_graph_def = input_graph_def;
  if (_config.replace_relu6 == 1) {
    TF_RETURN_IF_ERROR(
        ReplaceRelu6WithRelu(current_graph_def, &processed_graph_def));
    SaveGraphForDebugging(processed_graph_def, "replaced_relu6.pb",
                          _config.output_dir);
    current_graph_def = processed_graph_def;
  }

  std::vector<std::vector<ConvBiasPair>> conv_group;
  TF_RETURN_IF_ERROR(ParseConvPairs(current_graph_def, conv_group,
                                    _config.input_nodes, _config.output_nodes));

  const int cle_iter = 10;
  DLOG_INFO(1) << "Implementing cross layer equalization for iteration number: "
               << cle_iter;
  for (auto iter = 0; iter < cle_iter; ++iter) {
    std::cout << "Implementing cross layer equalization for " << iter + 1 << "/"
              << cle_iter << std::endl;
    for (auto i = 0; i < conv_group.size(); ++i) {
      const auto &consecutive_conv = conv_group[i];
      for (auto j = 0; j < consecutive_conv.size() - 1; ++j) {
        TF_RETURN_IF_ERROR(
            EqualizeConvPair(current_graph_def, processed_graph_def,
                             consecutive_conv[j], consecutive_conv[j + 1]));
        current_graph_def = processed_graph_def;
      }
    }
  }
  output_graph_def = current_graph_def;
  SaveGraphForDebugging(output_graph_def, "cross_layers_equalization.pb",
                        _config.output_dir);
  return Status::OK();
}

Status GraphQuantizer::PartitionGraph(
    const GraphDef &input_graph_def, GraphDef &main_graph_def,
    GraphDef &aux_graph_def, std::map<string, NodeDef> &origin_input_nodes) {
  // Get subgraph between input_nodes and output_nodes
  TransformFuncContext context_strip;
  context_strip.input_names = _config.input_nodes;
  context_strip.output_names = _config.output_nodes;
  TransformFuncParameters *param_strip = &context_strip.params;
  for (int i = 0; i < _config.input_nodes.size(); i++) {
    string node_name = _config.input_nodes[i];
    (*param_strip)["name"].push_back(node_name);

    (*param_strip)["type_for_name"].push_back("float32");
    (*param_strip)["shape_for_name"].push_back(_config.input_shapes[i]);
  }
  TF_RETURN_IF_ERROR(
      StripUnusedNodes(input_graph_def, context_strip, &main_graph_def));

  std::unordered_set<string> main_nodes;
  for (auto node : main_graph_def.node()) {
    main_nodes.insert(node.name());
  }
  FilterGraphDef(
      input_graph_def,
      [&](const NodeDef &node) { return main_nodes.count(node.name()) == 0; },
      &aux_graph_def);

  // Save original input_nodes for merging
  std::map<string, const NodeDef *> node_map;
  MapNamesToNodes(input_graph_def, &node_map);

  for (auto name : _config.input_nodes) {
    origin_input_nodes[name] = *(node_map[name]);
  }

  SaveGraphForDebugging(main_graph_def, "main_graph.pb", _config.output_dir);
  SaveGraphForDebugging(aux_graph_def, "aux_graph.pb", _config.output_dir);
  return Status::OK();
}

Status GraphQuantizer::MergeGraph(
    const GraphDef &main_graph_def, const GraphDef &aux_graph_def,
    const std::map<string, NodeDef> origin_input_nodes,
    GraphDef &output_graph_def) {
  std::map<string, string> inputs_to_rename;
  std::unordered_set<string> nodes_to_ignore;
  for (auto node : main_graph_def.node()) {
    if (node.op() == "FixNeuron") {
      inputs_to_rename[node.input(0)] = node.name();
      nodes_to_ignore.insert(node.name());
    }
  }

  GraphDef processed_graph_def;
  for (auto node : main_graph_def.node()) {
    if (origin_input_nodes.count(node.name())) {
      *(processed_graph_def.mutable_node()->Add()) =
          origin_input_nodes.at(node.name());
    } else {
      *(processed_graph_def.mutable_node()->Add()) = node;
    }
  }

  for (auto node : aux_graph_def.node()) {
    *(processed_graph_def.mutable_node()->Add()) = node;
  }

  GraphDef renamed_graph_def;
  RenameNodeInputs(processed_graph_def, inputs_to_rename, nodes_to_ignore,
                   &renamed_graph_def);

  SortByExecutionOrder(renamed_graph_def, &output_graph_def);
  return Status::OK();
}

Status GraphQuantizer::CreateQuantizeCalibrationGraph(
    const GraphDef &input_graph_def, GraphDef &output_graph_def) {
  SaveGraphForDebugging(input_graph_def, "input_graph_def.pb",
                        _config.output_dir);
  _config.phase = QuantizePhase::CALIB;
  output_graph_def.Clear();
  GraphDef current_graph_def, processed_graph_def;

  // Partition
  GraphDef main_graph_def, aux_graph_def;
  std::map<string, NodeDef> origin_input_nodes;
  TF_RETURN_IF_ERROR(PartitionGraph(input_graph_def, main_graph_def,
                                    aux_graph_def, origin_input_nodes));
  processed_graph_def = main_graph_def;

  // Optimize main graph
  current_graph_def = processed_graph_def;
  TF_RETURN_IF_ERROR(
      CreateOptimizedGraph(current_graph_def, processed_graph_def));

  // Parse main graph for pattern matching and locate ops to quantize
  current_graph_def = processed_graph_def;
  TF_RETURN_IF_ERROR(ParseGraph(current_graph_def, _matched_node_patterns,
                                _matched_nodes, _config.ignore_nodes,
                                _unmatched_nodes));
  for (const NodeDef &node : current_graph_def.node()) {
    if (_unmatched_nodes.count(node.name())) {
      DLOG_INFO(1) << "Found un-quantizable node: " << node.name() << "("
                   << node.op() << ")";
    }
  }
  TF_RETURN_IF_ERROR(_LocateOpsToQuantize(current_graph_def));

  // modify _ops_to_quantize according to _config.nodes_bit and
  // _config.nodes_method Insert fix_neuron in the main graph
  TF_RETURN_IF_ERROR(_ModifyFixNeuronConfig(current_graph_def));

  // Insert fix_neuron in the graph
  TF_RETURN_IF_ERROR(
      _InsertFixNeuronOps(current_graph_def, processed_graph_def));

  // Merge
  current_graph_def = processed_graph_def;
  TF_RETURN_IF_ERROR(MergeGraph(current_graph_def, aux_graph_def,
                                origin_input_nodes, processed_graph_def));

  output_graph_def = processed_graph_def;
  SaveGraphForDebugging(output_graph_def, "quantize_calibration_model.pb",
                        _config.output_dir);
  return Status::OK();
}

Status GraphQuantizer::CreateQuantizeTrainingGraph(
    const GraphDef &input_graph_def, GraphDef &output_graph_def) {
  _config.phase = QuantizePhase::TRAIN;
  output_graph_def.Clear();
  GraphDef current_graph_def, processed_graph_def;

  SaveGraphForDebugging(input_graph_def, "input_model.pb", _config.output_dir);

  processed_graph_def = input_graph_def;

  // Fold Batchnorms(is_training=true)
  current_graph_def = processed_graph_def;
  bool is_training = true;
  TF_RETURN_IF_ERROR(
      FoldBatchNorms(current_graph_def, &processed_graph_def, is_training));
  SaveGraphForDebugging(processed_graph_def, "fold_batchnorms_train.pb",
                        _config.output_dir);

  // Partition
  current_graph_def = processed_graph_def;
  GraphDef main_graph_def, aux_graph_def;
  std::map<string, NodeDef> origin_input_nodes;
  TF_RETURN_IF_ERROR(PartitionGraph(current_graph_def, main_graph_def,
                                    aux_graph_def, origin_input_nodes));
  processed_graph_def = main_graph_def;

  // // adjust hard swish compute order if the pattern is
  // // add + relu6 + mul_x + mul_1/6
  // current_graph_def = processed_graph_def;
  // TF_RETURN_IF_ERROR(
  //     AdjustHardSwishComputeOrder(current_graph_def, &processed_graph_def));
  // SaveGraphForDebugging(processed_graph_def,
  //                       "adjust_hard_swish_compute_order.pb",
  //                       _config.output_dir);

  // Simulate DPU
  if (_config.simulate_dpu == 1) {
    current_graph_def = processed_graph_def;
    TF_RETURN_IF_ERROR(SimulateDPU(
        current_graph_def, &processed_graph_def, _config.scale_all_avgpool,
        _config.replace_softmax, _config.replace_sigmoid));
    SaveGraphForDebugging(processed_graph_def, "simulate_dpu_train.pb",
                          _config.output_dir);
  }

  // Parse main graph for pattern matching and locate ops to quantize
  current_graph_def = processed_graph_def;
  TF_RETURN_IF_ERROR(ParseGraph(current_graph_def, _matched_node_patterns,
                                _matched_nodes, _config.ignore_nodes,
                                _unmatched_nodes));
  for (const NodeDef &node : current_graph_def.node()) {
    if (_unmatched_nodes.count(node.name())) {
      DLOG_INFO(1) << "Found un-quantizable node: " << node.name() << "("
                   << node.op() << ")";
    }
  }
  TF_RETURN_IF_ERROR(_LocateOpsToQuantize(current_graph_def));

  // modify _ops_to_quantize according to _config.nodes_bit and
  // _config.nodes_method Insert fix_neuron in the main graph
  TF_RETURN_IF_ERROR(_ModifyFixNeuronConfig(current_graph_def));

  // Insert fix_neuron in the graph
  TF_RETURN_IF_ERROR(
      _InsertFixNeuronOps(current_graph_def, processed_graph_def));

  // Merge
  current_graph_def = processed_graph_def;
  TF_RETURN_IF_ERROR(MergeGraph(current_graph_def, aux_graph_def,
                                origin_input_nodes, processed_graph_def));

  output_graph_def = processed_graph_def;
  SaveGraphForDebugging(output_graph_def, "quantize_train_model.pb",
                        _config.output_dir);
  return Status::OK();
}

Status GraphQuantizer::CreateQuantizeEvaluationGraph(
    const GraphDef &input_graph_def, GraphDef &output_graph_def) {
  _config.phase = QuantizePhase::EVAL;
  output_graph_def.Clear();
  GraphDef current_graph_def, processed_graph_def;

  current_graph_def = input_graph_def;
  TF_RETURN_IF_ERROR(
      _FreezeFixNeuronOps(current_graph_def, processed_graph_def));

  current_graph_def = processed_graph_def;
  TF_RETURN_IF_ERROR(
      _AdjustQuantizePos(current_graph_def, processed_graph_def));

  output_graph_def = processed_graph_def;
  SaveGraphForDebugging(output_graph_def, "quantize_evaluation_model.pb",
                        _config.output_dir);
  return Status::OK();
}

Status GraphQuantizer::ConvertFoldedBatchnorms(const GraphDef &input_graph_def,
                                               GraphDef &output_graph_def) {
  output_graph_def.Clear();
  GraphDef current_graph_def, processed_graph_def;

  // First remove Identity|CheckNumerics
  current_graph_def = input_graph_def;
  TransformFuncContext context_remove_nodes;
  context_remove_nodes.input_names = _config.input_nodes;
  context_remove_nodes.output_names = _config.output_nodes;
  context_remove_nodes.params.insert(
      std::pair<string, std::vector<string>>({"op", {string("Identity")}}));
  context_remove_nodes.params.insert(std::pair<string, std::vector<string>>(
      {"op", {string("CheckNumerics")}}));
  TF_RETURN_IF_ERROR(RemoveNodes(current_graph_def, context_remove_nodes,
                                 &processed_graph_def));
  // Fold fix neuron
  current_graph_def = processed_graph_def;
  bool fold_only = true;
  TF_RETURN_IF_ERROR(
      FoldFixNeuron(current_graph_def, &processed_graph_def, fold_only));
  SaveGraphForDebugging(processed_graph_def,
                        "fold_fix_neuron_in_convert_fold_bn.pb",
                        _config.output_dir);

  // Separate shared constants
  current_graph_def = processed_graph_def;
  bool is_quantized = true;
  TF_RETURN_IF_ERROR(SeparateSharedConstants(
      current_graph_def, &processed_graph_def, is_quantized));
  SaveGraphForDebugging(processed_graph_def,
                        "separate_shared_constants_in_train.pb",
                        _config.output_dir);

  // Do convert for Conv2D + bias + bn
  current_graph_def = processed_graph_def;
  std::map<string, string> inputs_to_rename;
  bool did_graph_change = false;
  do {
    did_graph_change = false;
    inputs_to_rename.clear();
    TF_RETURN_IF_ERROR(ReplaceMatchingOpTypes(
        current_graph_def,
        // clang-format off
        {"BiasAdd|Add|AddV2",
          {
            {"Conv2D",
              {
                {"FixNeuron", // input_quant_node
                  {
                    {"*"}, // input_node
                  }
                },
                {"FixNeuron", // weights_quant_node
                  {
                    {"Mul",
                      {
                        {"Mul"},
                        {"Const"}, // weights_node
                      }
                    }
                  }
                }
              }
            },
            {"FixNeuron", // bias_quant_node
              {
                {"Add|AddV2",
                  {
                    {"Sub",
                      {
                        {"Const"}, // beta_ndoe
                        {"Mul",
                          {
                            {"Mul",
                              {
                                {"Rsqrt",
                                  {
                                    {"Add|AddV2",
                                      {
                                        {"Const"}, // eps_node
                                        {"Const"}, // variance_node
                                      }
                                    },
                                  }
                                },
                                {"Const"},  // gamma_node,
                              }
                            },
                            {"Const"}, // mean_node
                          }
                        },
                      }
                    },
                    {"Mul",
                      {
                        {"Mul"},
                        {"Const"}, // bias
                      }
                    },
                  }
                },
              }
            },
          }
        },
        // clang-format on
        [&did_graph_change, &inputs_to_rename](
            const NodeMatch &match, const std::set<string> &input_nodes,
            const std::set<string> &output_nodes,
            std::vector<NodeDef> *new_nodes) {
          const NodeDef &biasadd_node = match.node;
          const NodeDef &conv_node = match.inputs[0].node;
          const NodeDef &bias_quant_node = match.inputs[1].node;
          const NodeDef &add_1_node = match.inputs[1].inputs[0].node;
          const NodeDef &sub_node = match.inputs[1].inputs[0].inputs[0].node;
          const NodeDef &input_quant_node = match.inputs[0].inputs[0].node;
          const NodeDef &input_node = match.inputs[0].inputs[0].inputs[0].node;
          const NodeDef &weights_quant_node = match.inputs[0].inputs[1].node;
          const NodeDef &mul_node_1 = match.inputs[0].inputs[1].inputs[0].node;
          const NodeDef &mul_node_3 = match.inputs[1].inputs[0].inputs[1].node;
          const NodeDef &org_bias_node =
              match.inputs[1].inputs[0].inputs[1].inputs[1].node;
          const NodeDef &weights_node =
              match.inputs[0].inputs[1].inputs[0].inputs[1].node;
          const NodeDef &beta_node =
              match.inputs[1].inputs[0].inputs[0].inputs[0].node;
          const NodeDef &gamma_node = match.inputs[1]
                                          .inputs[0]
                                          .inputs[0]
                                          .inputs[1]
                                          .inputs[0]
                                          .inputs[1]
                                          .node;
          const NodeDef &mean_node =
              match.inputs[1].inputs[0].inputs[0].inputs[1].inputs[1].node;
          const NodeDef &variance_node = match.inputs[1]
                                             .inputs[0]
                                             .inputs[0]
                                             .inputs[1]
                                             .inputs[0]
                                             .inputs[0]
                                             .inputs[0]
                                             .inputs[1]
                                             .node;
          const NodeDef &eps_node = match.inputs[1]
                                        .inputs[0]
                                        .inputs[0]
                                        .inputs[1]
                                        .inputs[0]
                                        .inputs[0]
                                        .inputs[0]
                                        .inputs[0]
                                        .node;

          Tensor mean = GetNodeTensorAttr(mean_node, "value");
          Tensor variance = GetNodeTensorAttr(variance_node, "value");
          Tensor beta = GetNodeTensorAttr(beta_node, "value");
          Tensor gamma = GetNodeTensorAttr(gamma_node, "value");
          Tensor bias = GetNodeTensorAttr(org_bias_node, "value");
          float variance_epsilon =
              GetNodeTensorAttr(eps_node, "value").flat<float>()(0);

          // Calulate the scale and offset
          Tensor scale(DT_FLOAT, mean.shape());
          auto scale_flatten = scale.flat<float>();
          Tensor offset(DT_FLOAT, mean.shape());
          auto offset_flatten = offset.flat<float>();

          // Calculate the scale and offset values to apply.
          const int64 num_cols = mean.shape().dim_size(0);
          bool scale_after_normalization = true;
          if (scale_after_normalization) {
            for (int i = 0; i < num_cols; ++i) {
              scale_flatten(i) =
                  (1.0f / sqrtf(variance.flat<float>()(i) + variance_epsilon)) *
                  gamma.flat<float>()(i);
            }
          } else {
            for (int i = 0; i < num_cols; ++i) {
              scale_flatten(i) =
                  (1.0f / sqrtf(variance.flat<float>()(i) + variance_epsilon));
            }
          }
          for (int i = 0; i < num_cols; ++i) {
            offset_flatten(i) =
                ((bias.flat<float>()(i) - mean.flat<float>()(i)) *
                 scale_flatten(i)) +
                beta.flat<float>()(i);
          }

          // Construct the scale and biases nodes.
          NodeDef scale_node;
          scale_node.set_op("Const");
          scale_node.set_name(conv_node.name() + "/scale");
          SetNodeAttr("dtype", DT_FLOAT, &scale_node);
          SetNodeTensorAttr<float>("value", scale, &scale_node);

          NodeDef bias_node;
          bias_node.set_op("Const");
          bias_node.set_name(conv_node.name() + "/biases");
          SetNodeAttr("dtype", DT_FLOAT, &bias_node);
          SetNodeTensorAttr<float>("value", offset, &bias_node);

          // Get merged weights, using functions from fold_batch_norms.h
          NodeDef scaled_weights_node;
          TF_RETURN_IF_ERROR(GetMergedConvWeights(
              conv_node, weights_node, scale_node, &scaled_weights_node));

          // Construct the final nodes
          new_nodes->push_back(input_node);
          new_nodes->push_back(input_quant_node);
          new_nodes->push_back(scaled_weights_node);

          NodeDef new_weights_quant_node = weights_quant_node;
          new_weights_quant_node.mutable_input()->Clear();
          AddNodeInput(scaled_weights_node.name(), &new_weights_quant_node);
          new_nodes->push_back(new_weights_quant_node);

          NodeDef new_conv_node = conv_node;
          new_conv_node.mutable_input()->Clear();
          AddNodeInput(input_quant_node.name(), &new_conv_node);
          AddNodeInput(new_weights_quant_node.name(), &new_conv_node);
          new_nodes->push_back(new_conv_node);
          new_nodes->push_back(bias_node);

          NodeDef new_bias_quant_node = bias_quant_node;
          new_bias_quant_node.mutable_input()->Clear();
          AddNodeInput(bias_node.name(), &new_bias_quant_node);
          new_nodes->push_back(new_bias_quant_node);

          NodeDef new_biasadd_node = biasadd_node;
          new_biasadd_node.mutable_input()->Clear();
          AddNodeInput(new_conv_node.name(), &new_biasadd_node);
          AddNodeInput(new_bias_quant_node.name(), &new_biasadd_node);
          new_nodes->push_back(new_biasadd_node);

          did_graph_change = true;

          return Status::OK();
        },
        {true}, &processed_graph_def));
    TF_RETURN_IF_ERROR(RenameNodeInputs(processed_graph_def, inputs_to_rename,
                                        std::unordered_set<string>(),
                                        &current_graph_def));
  } while (did_graph_change);

  // Do convert for DepthwiseConv2D + bias + BN
  do {
    did_graph_change = false;
    inputs_to_rename.clear();
    TF_RETURN_IF_ERROR(ReplaceMatchingOpTypes(
        current_graph_def,
        // clang-format off
        {"BiasAdd|Add|AddV2",
          {
            {"DepthwiseConv2dNative",
              {
                {"FixNeuron", // input_quant_node
                  {
                    {"*"}, // input_node
                  }
                },
                {"FixNeuron", // weights_quant_node
                  {
                    {"Mul",
                      {
                        {"Reshape",
                          {
                            {"Mul"},
                            {"Const"},
                          }
                        },
                        {"Const"}, // weights_node
                      }
                    },
                  }
                },
              }
            },
            {"FixNeuron", // bias_quant_node
              {
                {"Add|AddV2",
                  {
                    {"Sub",
                      {
                        {"Const"}, // beta_ndoe
                        {"Mul",
                          {
                            {"Mul",
                              {
                                {"Rsqrt",
                                  {
                                    {"Add|AddV2",
                                      {
                                        {"Const"}, // eps_node
                                        {"Const"}, // variance_node
                                      }
                                    },
                                  }
                                },
                                {"Const"},  // gamma_node,
                              }
                            },
                            {"Const"}, // mean_node
                          }
                        },
                      }
                    },
                    {"Mul",
                      {
                        {"Mul"},
                        {"Const"}, // bias
                      }
                    },
                  }
                },
              }
            },
          }
        },
        // clang-format on
        [&did_graph_change, &inputs_to_rename](
            const NodeMatch &match, const std::set<string> &input_nodes,
            const std::set<string> &output_nodes,
            std::vector<NodeDef> *new_nodes) {
          const NodeDef &biasadd_node = match.node;
          const NodeDef &conv_node = match.inputs[0].node;
          const NodeDef &add_1_node = match.inputs[1].inputs[0].node;
          const NodeDef &bias_quant_node = match.inputs[1].node;
          const NodeDef &sub_node = match.inputs[1].inputs[0].inputs[0].node;
          const NodeDef &input_quant_node = match.inputs[0].inputs[0].node;
          const NodeDef &input_node = match.inputs[0].inputs[0].inputs[0].node;
          const NodeDef &weights_quant_node = match.inputs[0].inputs[1].node;
          const NodeDef &mul_node_1 = match.inputs[0].inputs[1].inputs[0].node;
          const NodeDef &mul_node_3 = match.inputs[1].inputs[0].inputs[1].node;
          const NodeDef &weights_node =
              match.inputs[0].inputs[1].inputs[0].inputs[1].node;
          const NodeDef &org_bias_node =
              match.inputs[1].inputs[0].inputs[1].inputs[1].node;
          const NodeDef &reshape_quant_node =
              match.inputs[0].inputs[1].inputs[0].inputs[0].node;
          const NodeDef &reshape_node =
              match.inputs[0].inputs[1].inputs[0].inputs[0].inputs[0].node;
          const NodeDef &beta_node =
              match.inputs[1].inputs[0].inputs[0].inputs[0].node;
          const NodeDef &gamma_node = match.inputs[1]
                                          .inputs[0]
                                          .inputs[0]
                                          .inputs[1]
                                          .inputs[0]
                                          .inputs[1]
                                          .node;
          const NodeDef &mean_node =
              match.inputs[1].inputs[0].inputs[0].inputs[1].inputs[1].node;
          const NodeDef &variance_node = match.inputs[1]
                                             .inputs[0]
                                             .inputs[0]
                                             .inputs[1]
                                             .inputs[0]
                                             .inputs[0]
                                             .inputs[0]
                                             .inputs[1]
                                             .node;
          const NodeDef &eps_node = match.inputs[1]
                                        .inputs[0]
                                        .inputs[0]
                                        .inputs[1]
                                        .inputs[0]
                                        .inputs[0]
                                        .inputs[0]
                                        .inputs[0]
                                        .node;

          Tensor mean = GetNodeTensorAttr(mean_node, "value");
          Tensor variance = GetNodeTensorAttr(variance_node, "value");
          Tensor beta = GetNodeTensorAttr(beta_node, "value");
          Tensor gamma = GetNodeTensorAttr(gamma_node, "value");
          Tensor bias = GetNodeTensorAttr(org_bias_node, "value");
          float variance_epsilon =
              GetNodeTensorAttr(eps_node, "value").flat<float>()(0);

          // Calulate the scale and offset
          Tensor scale(DT_FLOAT, mean.shape());
          auto scale_flatten = scale.flat<float>();
          Tensor offset(DT_FLOAT, mean.shape());
          auto offset_flatten = offset.flat<float>();

          // Calculate the scale and offset values to apply.
          const int64 num_cols = mean.shape().dim_size(0);
          bool scale_after_normalization = true;
          if (scale_after_normalization) {
            for (int i = 0; i < num_cols; ++i) {
              scale_flatten(i) =
                  (1.0f / sqrtf(variance.flat<float>()(i) + variance_epsilon)) *
                  gamma.flat<float>()(i);
            }
          } else {
            for (int i = 0; i < num_cols; ++i) {
              scale_flatten(i) =
                  (1.0f / sqrtf(variance.flat<float>()(i) + variance_epsilon));
            }
          }
          for (int i = 0; i < num_cols; ++i) {
            offset_flatten(i) =
                ((bias.flat<float>()(i) - mean.flat<float>()(i)) *
                 scale_flatten(i)) +
                beta.flat<float>()(i);
          }

          // Construct the scale and biases nodes.
          NodeDef scale_node;
          scale_node.set_op("Const");
          scale_node.set_name(conv_node.name() + "/scale");
          SetNodeAttr("dtype", DT_FLOAT, &scale_node);
          SetNodeTensorAttr<float>("value", scale, &scale_node);

          NodeDef bias_node;
          bias_node.set_op("Const");
          bias_node.set_name(conv_node.name() + "/biases");
          SetNodeAttr("dtype", DT_FLOAT, &bias_node);
          SetNodeTensorAttr<float>("value", offset, &bias_node);

          // Get merged weights, using functions from fold_batch_norms.h
          NodeDef scaled_weights_node;
          TF_RETURN_IF_ERROR(GetMergedConvWeights(
              conv_node, weights_node, scale_node, &scaled_weights_node));

          // Construct the final nodes
          new_nodes->push_back(input_node);
          new_nodes->push_back(input_quant_node);
          new_nodes->push_back(scaled_weights_node);

          NodeDef new_weights_quant_node = weights_quant_node;
          new_weights_quant_node.mutable_input()->Clear();
          AddNodeInput(scaled_weights_node.name(), &new_weights_quant_node);
          new_nodes->push_back(new_weights_quant_node);

          NodeDef new_conv_node = conv_node;
          new_conv_node.mutable_input()->Clear();
          AddNodeInput(input_quant_node.name(), &new_conv_node);
          AddNodeInput(new_weights_quant_node.name(), &new_conv_node);
          new_nodes->push_back(new_conv_node);
          new_nodes->push_back(bias_node);

          NodeDef new_bias_quant_node = bias_quant_node;
          new_bias_quant_node.mutable_input()->Clear();
          AddNodeInput(bias_node.name(), &new_bias_quant_node);
          new_nodes->push_back(new_bias_quant_node);

          NodeDef new_biasadd_node = biasadd_node;
          new_biasadd_node.mutable_input()->Clear();
          AddNodeInput(new_conv_node.name(), &new_biasadd_node);
          AddNodeInput(new_bias_quant_node.name(), &new_biasadd_node);
          new_nodes->push_back(new_biasadd_node);

          did_graph_change = true;

          return Status::OK();
        },
        {true}, &processed_graph_def));
    TF_RETURN_IF_ERROR(RenameNodeInputs(processed_graph_def, inputs_to_rename,
                                        std::unordered_set<string>(),
                                        &current_graph_def));
  } while (did_graph_change);

  // Do convert for Conv2D
  do {
    did_graph_change = false;
    inputs_to_rename.clear();
    TF_RETURN_IF_ERROR(ReplaceMatchingOpTypes(
        current_graph_def,
        // clang-format off
        {"BiasAdd|Add|AddV2",
          {
            {"Conv2D",
              {
                {"FixNeuron", // input_quant_node
                  {
                    {"*"}, // input_node
                  }
                },
                {"FixNeuron", // weights_quant_node
                  {
                    {"Mul",
                      {
                        {"Mul"},
                        {"Const"}, // weights_node
                      }
                    }
                  }
                }
              }
            },
            {"FixNeuron", // bias_quant_node
              {
                {"Sub",
                  {
                    {"Const"}, // beta_ndoe
                    {"Mul",
                      {
                        {"Mul",
                          {
                            {"Rsqrt",
                              {
                                {"Add|AddV2",
                                  {
                                    {"Const"}, // eps_node
                                    {"Const"}, // variance_node
                                  }
                                }
                              }
                            },
                            {"Const"}  // gamma_node,
                          }
                        },
                        {"Const"}, // mean_node
                      }
                    },
                  }
                },
              }
            },
          }
        },
        // clang-format on
        [&did_graph_change, &inputs_to_rename](
            const NodeMatch &match, const std::set<string> &input_nodes,
            const std::set<string> &output_nodes,
            std::vector<NodeDef> *new_nodes) {
          const NodeDef &biasadd_node = match.node;
          const NodeDef &conv_node = match.inputs[0].node;
          const NodeDef &bias_quant_node = match.inputs[1].node;
          const NodeDef &sub_node = match.inputs[1].inputs[0].node;
          const NodeDef &input_quant_node = match.inputs[0].inputs[0].node;
          const NodeDef &input_node = match.inputs[0].inputs[0].inputs[0].node;
          const NodeDef &weights_quant_node = match.inputs[0].inputs[1].node;
          const NodeDef &mul_node_1 = match.inputs[0].inputs[1].inputs[0].node;
          const NodeDef &weights_node =
              match.inputs[0].inputs[1].inputs[0].inputs[1].node;
          const NodeDef &beta_node = match.inputs[1].inputs[0].inputs[0].node;
          const NodeDef &gamma_node =
              match.inputs[1].inputs[0].inputs[1].inputs[0].inputs[1].node;
          const NodeDef &mean_node =
              match.inputs[1].inputs[0].inputs[1].inputs[1].node;
          const NodeDef &variance_node = match.inputs[1]
                                             .inputs[0]
                                             .inputs[1]
                                             .inputs[0]
                                             .inputs[0]
                                             .inputs[0]
                                             .inputs[1]
                                             .node;
          const NodeDef &eps_node = match.inputs[1]
                                        .inputs[0]
                                        .inputs[1]
                                        .inputs[0]
                                        .inputs[0]
                                        .inputs[0]
                                        .inputs[0]
                                        .node;

          Tensor mean = GetNodeTensorAttr(mean_node, "value");
          Tensor variance = GetNodeTensorAttr(variance_node, "value");
          Tensor beta = GetNodeTensorAttr(beta_node, "value");
          Tensor gamma = GetNodeTensorAttr(gamma_node, "value");
          float variance_epsilon =
              GetNodeTensorAttr(eps_node, "value").flat<float>()(0);

          // Calulate the scale and offset
          Tensor scale(DT_FLOAT, mean.shape());
          auto scale_flatten = scale.flat<float>();
          Tensor offset(DT_FLOAT, mean.shape());
          auto offset_flatten = offset.flat<float>();

          // Calculate the scale and offset values to apply.
          const int64 num_cols = mean.shape().dim_size(0);
          bool scale_after_normalization = true;
          if (scale_after_normalization) {
            for (int i = 0; i < num_cols; ++i) {
              scale_flatten(i) =
                  (1.0f / sqrtf(variance.flat<float>()(i) + variance_epsilon)) *
                  gamma.flat<float>()(i);
            }
          } else {
            for (int i = 0; i < num_cols; ++i) {
              scale_flatten(i) =
                  (1.0f / sqrtf(variance.flat<float>()(i) + variance_epsilon));
            }
          }
          for (int i = 0; i < num_cols; ++i) {
            offset_flatten(i) = (-mean.flat<float>()(i) * scale_flatten(i)) +
                                beta.flat<float>()(i);
          }

          // Construct the scale and biases nodes.
          NodeDef scale_node;
          scale_node.set_op("Const");
          scale_node.set_name(conv_node.name() + "/scale");
          SetNodeAttr("dtype", DT_FLOAT, &scale_node);
          SetNodeTensorAttr<float>("value", scale, &scale_node);

          NodeDef bias_node;
          bias_node.set_op("Const");
          bias_node.set_name(conv_node.name() + "/biases");
          SetNodeAttr("dtype", DT_FLOAT, &bias_node);
          SetNodeTensorAttr<float>("value", offset, &bias_node);

          // Get merged weights, using functions from fold_batch_norms.h
          NodeDef scaled_weights_node;
          TF_RETURN_IF_ERROR(GetMergedConvWeights(
              conv_node, weights_node, scale_node, &scaled_weights_node));

          // Construct the final nodes
          new_nodes->push_back(input_node);
          new_nodes->push_back(input_quant_node);
          new_nodes->push_back(scaled_weights_node);

          NodeDef new_weights_quant_node = weights_quant_node;
          new_weights_quant_node.mutable_input()->Clear();
          AddNodeInput(scaled_weights_node.name(), &new_weights_quant_node);
          new_nodes->push_back(new_weights_quant_node);

          NodeDef new_conv_node = conv_node;
          new_conv_node.mutable_input()->Clear();
          AddNodeInput(input_quant_node.name(), &new_conv_node);
          AddNodeInput(new_weights_quant_node.name(), &new_conv_node);
          new_nodes->push_back(new_conv_node);
          new_nodes->push_back(bias_node);

          NodeDef new_bias_quant_node = bias_quant_node;
          new_bias_quant_node.mutable_input()->Clear();
          AddNodeInput(bias_node.name(), &new_bias_quant_node);
          new_nodes->push_back(new_bias_quant_node);

          NodeDef new_biasadd_node = biasadd_node;
          new_biasadd_node.mutable_input()->Clear();
          AddNodeInput(new_conv_node.name(), &new_biasadd_node);
          AddNodeInput(new_bias_quant_node.name(), &new_biasadd_node);
          new_nodes->push_back(new_biasadd_node);

          did_graph_change = true;

          return Status::OK();
        },
        {true}, &processed_graph_def));
    TF_RETURN_IF_ERROR(RenameNodeInputs(processed_graph_def, inputs_to_rename,
                                        std::unordered_set<string>(),
                                        &current_graph_def));
  } while (did_graph_change);

  // Do convert for DepthwiseConv2D
  do {
    did_graph_change = false;
    inputs_to_rename.clear();
    TF_RETURN_IF_ERROR(ReplaceMatchingOpTypes(
        current_graph_def,
        // clang-format off
        {"BiasAdd|Add|AddV2",
          {
            {"DepthwiseConv2dNative",
              {
                {"FixNeuron", // input_quant_node
                  {
                    {"*"}, // input_node
                  }
                },
                {"FixNeuron", // weights_quant_node
                  {
                    {"Mul",
                      {
                        {"Reshape",
                          {
                            {"Mul"},
                            {"Const"},
                          }
                        },
                        {"Const"}, // weights_node
                      }
                    },
                  }
                },
              }
            },
            {"FixNeuron", // bias_quant_node
              {
                {"Sub",
                  {
                    {"Const"}, // beta_ndoe
                    {"Mul",
                      {
                        {"Mul",
                          {
                            {"Rsqrt",
                              {
                                {"Add|AddV2",
                                  {
                                    {"Const"}, // eps_node
                                    {"Const"}, // variance_node
                                  }
                                },
                              }
                            },
                            {"Const"}  // gamma_node,
                          }
                        },
                        {"Const"}, // mean_node
                      }
                    },
                  }
                },
              }
            },
          }
        },
        // clang-format on
        [&did_graph_change, &inputs_to_rename](
            const NodeMatch &match, const std::set<string> &input_nodes,
            const std::set<string> &output_nodes,
            std::vector<NodeDef> *new_nodes) {
          const NodeDef &biasadd_node = match.node;
          const NodeDef &conv_node = match.inputs[0].node;
          const NodeDef &bias_quant_node = match.inputs[1].node;
          const NodeDef &sub_node = match.inputs[1].inputs[0].node;
          const NodeDef &input_quant_node = match.inputs[0].inputs[0].node;
          const NodeDef &input_node = match.inputs[0].inputs[0].inputs[0].node;
          const NodeDef &weights_quant_node = match.inputs[0].inputs[1].node;
          const NodeDef &mul_node_1 = match.inputs[0].inputs[1].inputs[0].node;
          const NodeDef &weights_node =
              match.inputs[0].inputs[1].inputs[0].inputs[1].node;
          const NodeDef &reshape_quant_node =
              match.inputs[0].inputs[1].inputs[0].inputs[0].node;
          const NodeDef &reshape_node =
              match.inputs[0].inputs[1].inputs[0].inputs[0].inputs[0].node;
          const NodeDef &beta_node = match.inputs[1].inputs[0].inputs[0].node;
          const NodeDef &gamma_node =
              match.inputs[1].inputs[0].inputs[1].inputs[0].inputs[1].node;
          const NodeDef &mean_node =
              match.inputs[1].inputs[0].inputs[1].inputs[1].node;
          const NodeDef &variance_node = match.inputs[1]
                                             .inputs[0]
                                             .inputs[1]
                                             .inputs[0]
                                             .inputs[0]
                                             .inputs[0]
                                             .inputs[1]
                                             .node;
          const NodeDef &eps_node = match.inputs[1]
                                        .inputs[0]
                                        .inputs[1]
                                        .inputs[0]
                                        .inputs[0]
                                        .inputs[0]
                                        .inputs[0]
                                        .node;

          Tensor mean = GetNodeTensorAttr(mean_node, "value");
          Tensor variance = GetNodeTensorAttr(variance_node, "value");
          Tensor beta = GetNodeTensorAttr(beta_node, "value");
          Tensor gamma = GetNodeTensorAttr(gamma_node, "value");
          float variance_epsilon =
              GetNodeTensorAttr(eps_node, "value").flat<float>()(0);

          // Calulate the scale and offset
          Tensor scale(DT_FLOAT, mean.shape());
          auto scale_flatten = scale.flat<float>();
          Tensor offset(DT_FLOAT, mean.shape());
          auto offset_flatten = offset.flat<float>();

          // Calculate the scale and offset values to apply.
          const int64 num_cols = mean.shape().dim_size(0);
          bool scale_after_normalization = true;
          if (scale_after_normalization) {
            for (int i = 0; i < num_cols; ++i) {
              scale_flatten(i) =
                  (1.0f / sqrtf(variance.flat<float>()(i) + variance_epsilon)) *
                  gamma.flat<float>()(i);
            }
          } else {
            for (int i = 0; i < num_cols; ++i) {
              scale_flatten(i) =
                  (1.0f / sqrtf(variance.flat<float>()(i) + variance_epsilon));
            }
          }
          for (int i = 0; i < num_cols; ++i) {
            offset_flatten(i) = (-mean.flat<float>()(i) * scale_flatten(i)) +
                                beta.flat<float>()(i);
          }

          // Construct the scale and biases nodes.
          NodeDef scale_node;
          scale_node.set_op("Const");
          scale_node.set_name(conv_node.name() + "/scale");
          SetNodeAttr("dtype", DT_FLOAT, &scale_node);
          SetNodeTensorAttr<float>("value", scale, &scale_node);

          NodeDef bias_node;
          bias_node.set_op("Const");
          bias_node.set_name(conv_node.name() + "/biases");
          SetNodeAttr("dtype", DT_FLOAT, &bias_node);
          SetNodeTensorAttr<float>("value", offset, &bias_node);

          // Get merged weights, using functions from fold_batch_norms.h
          NodeDef scaled_weights_node;
          TF_RETURN_IF_ERROR(GetMergedConvWeights(
              conv_node, weights_node, scale_node, &scaled_weights_node));

          // Construct the final nodes
          new_nodes->push_back(input_node);
          new_nodes->push_back(input_quant_node);
          new_nodes->push_back(scaled_weights_node);

          NodeDef new_weights_quant_node = weights_quant_node;
          new_weights_quant_node.mutable_input()->Clear();
          AddNodeInput(scaled_weights_node.name(), &new_weights_quant_node);
          new_nodes->push_back(new_weights_quant_node);

          NodeDef new_conv_node = conv_node;
          new_conv_node.mutable_input()->Clear();
          AddNodeInput(input_quant_node.name(), &new_conv_node);
          AddNodeInput(new_weights_quant_node.name(), &new_conv_node);
          new_nodes->push_back(new_conv_node);
          new_nodes->push_back(bias_node);

          NodeDef new_bias_quant_node = bias_quant_node;
          new_bias_quant_node.mutable_input()->Clear();
          AddNodeInput(bias_node.name(), &new_bias_quant_node);
          new_nodes->push_back(new_bias_quant_node);

          NodeDef new_biasadd_node = biasadd_node;
          new_biasadd_node.mutable_input()->Clear();
          AddNodeInput(new_conv_node.name(), &new_biasadd_node);
          AddNodeInput(new_bias_quant_node.name(), &new_biasadd_node);
          new_nodes->push_back(new_biasadd_node);

          did_graph_change = true;

          return Status::OK();
        },
        {true}, &processed_graph_def));
    TF_RETURN_IF_ERROR(RenameNodeInputs(processed_graph_def, inputs_to_rename,
                                        std::unordered_set<string>(),
                                        &current_graph_def));
  } while (did_graph_change);

  output_graph_def = current_graph_def;
  return Status::OK();
}

Status GraphQuantizer::CreateQuantizeDeployGraph(
    const GraphDef &input_graph_def, GraphDef &output_graph_def) {
  output_graph_def.Clear();
  GraphDef current_graph_def, processed_graph_def;

  // Remove Identity|CheckNumerics
  current_graph_def = input_graph_def;
  TransformFuncContext context_remove_nodes;
  context_remove_nodes.input_names = _config.input_nodes;
  context_remove_nodes.output_names = _config.output_nodes;
  context_remove_nodes.params.insert(
      std::pair<string, std::vector<string>>({"op", {string("Identity")}}));
  context_remove_nodes.params.insert(std::pair<string, std::vector<string>>(
      {"op", {string("CheckNumerics")}}));
  TF_RETURN_IF_ERROR(RemoveNodes(current_graph_def, context_remove_nodes,
                                 &processed_graph_def));
  SaveGraphForDebugging(processed_graph_def, "remove_nodes_in_deploy.pb",
                        _config.output_dir);

  // Get subgraph between input_nodes and output_nodes
  current_graph_def = processed_graph_def;
  TransformFuncContext context_strip;
  context_strip.input_names = _config.input_nodes;
  context_strip.output_names = _config.output_nodes;
  TransformFuncParameters *param_strip = &context_strip.params;
  for (int i = 0; i < _config.input_nodes.size(); i++) {
    string node_name = _config.input_nodes[i];
    (*param_strip)["name"].push_back(node_name);

    (*param_strip)["type_for_name"].push_back("float32");
    (*param_strip)["shape_for_name"].push_back(_config.input_shapes[i]);
  }
  TF_RETURN_IF_ERROR(
      StripUnusedNodes(current_graph_def, context_strip, &processed_graph_def));
  SaveGraphForDebugging(processed_graph_def, "deploy_sub_graph.pb",
                        _config.output_dir);

  // Inference Reshapes
  current_graph_def = processed_graph_def;
  int deploy_batch_size = 1;
  TF_RETURN_IF_ERROR(InferenceShape(current_graph_def, &processed_graph_def,
                                    deploy_batch_size));
  current_graph_def = processed_graph_def;
  TF_RETURN_IF_ERROR(
      StripUnusedNodes(current_graph_def, context_strip, &processed_graph_def));
  SaveGraphForDebugging(processed_graph_def, "inference_reshapes.pb",
                        _config.output_dir);

  // Fold fix neuron
  current_graph_def = processed_graph_def;
  TF_RETURN_IF_ERROR(FoldFixNeuron(current_graph_def, &processed_graph_def));
  SaveGraphForDebugging(processed_graph_def, "fold_fix_neuron.pb",
                        _config.output_dir);

  // Transform to deploy model
  current_graph_def = processed_graph_def;
  TF_RETURN_IF_ERROR(
      DeployQuantizedGraph(current_graph_def, &processed_graph_def, _config));

  // Remove Identity|CheckNumerics if they're at the head or tail
  current_graph_def = processed_graph_def;
  context_remove_nodes.input_names = {};
  context_remove_nodes.output_names = {};
  TF_RETURN_IF_ERROR(
      RemoveNodes(current_graph_def, context_remove_nodes, &output_graph_def));

  // Dump quantize info
  SaveQuantizeInfoForDebugging(output_graph_def, _config.output_dir);
  SaveGraphForDebugging(output_graph_def, "deployed.pb", _config.output_dir);

  return Status::OK();
}

}  // namespace decent_q
}  // namespace tensorflow
