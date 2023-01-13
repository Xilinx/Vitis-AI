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

#include "deploy_quantized_graph.h"
#include "fold_batch_norms.h"
#include "known_patterns.h"
#include "quantize_utils.h"
#include "tensorflow/core/framework/node_def_util.h"

namespace tensorflow {
namespace decent_q {

namespace {

std::vector<NodeDef> ConvertPlaceholder(
    const NodeMatch &match, std::map<string, string> &inputs_to_rename,
    std::unordered_set<string> &nodes_to_ignore, const bool &quantize_ignored) {
  std::vector<NodeDef> new_nodes;
  const NodeDef &placeholder_node = match.node;
  NodeDef converted_node = placeholder_node;
  if (quantize_ignored) {
    SetNodeAttr("quantize_ignored", true, &converted_node);
  }
  // Set batchsize to 1
  TensorShapeProto &shape =
      ((*converted_node.mutable_attr())["shape"].mutable_shape())[0];
  if (shape.dim(0).size() == -1) {
    shape.mutable_dim(0)->set_size(1);
  }
  // Don't need to copy opos, because it's already there
  new_nodes.push_back(converted_node);
  return new_nodes;
}

std::vector<NodeDef> ConvertConvfcBiasRelu(
    const NodeMatch &match, std::map<string, string> &inputs_to_rename,
    std::unordered_set<string> &nodes_to_ignore, const bool &quantize_ignored) {
  std::vector<NodeDef> new_nodes;

  const NodeDef &relu_node = match.node;
  const NodeDef &biasadd_node = match.inputs[0].node;
  const NodeDef &bias_node = match.inputs[0].inputs[1].node;
  const NodeDef &convfc_node = match.inputs[0].inputs[0].node;
  const NodeDef &weight_node = match.inputs[0].inputs[0].inputs[1].node;
  const NodeDef &input_node = match.inputs[0].inputs[0].inputs[0].node;
  CHECK(weight_node.op() == "Const")
      << "[DEPLOY ERROR] Expect const node for weight, but " << weight_node.op()
      << "(" << weight_node.name() << ") found.";
  CHECK(bias_node.op() == "Const")
      << "[DEPLOY ERROR] Expect const node for bias, but " << bias_node.op()
      << "(" << bias_node.name() << ") found.";

  DLOG_INFO(1) << "Convert convfc + bias + relu: " << relu_node.name() << " + "
               << biasadd_node.name() << " <-- " << convfc_node.name() << " + "
               << bias_node.name();

  NodeDef converted_node = convfc_node;
  converted_node.mutable_input()->Clear();
  AddNodeInput(input_node.name(), &converted_node);
  CopyNodeAttr(weight_node, "value", "weights", &converted_node);
  CopyNodeAttr(bias_node, "value", "bias", &converted_node);

  NodeDef new_relu_node = relu_node;
  new_relu_node.set_input(0, converted_node.name());

  if (quantize_ignored) {
    SetNodeAttr("quantize_ignored", true, &converted_node);
    SetNodeAttr("quantize_ignored", true, &new_relu_node);
  } else {
    CHECK(weight_node.attr().count("wpos"))
        << "[DEPLOY ERROR] Cannot find quantize info for weights: "
        << weight_node.name();
    CopyNodeAttr(weight_node, "wpos", "wpos", &converted_node);

    CHECK(bias_node.attr().count("wpos"))
        << "[DEPLOY ERROR] Cannot find quantize info for bias: "
        << bias_node.name();
    CopyNodeAttr(bias_node, "wpos", "bpos", &converted_node);

    CHECK(relu_node.attr().count("opos"))
        << "[DEPLOY ERROR] Cannot find quantize info for relu: "
        << relu_node.name();
    CopyNodeAttr(relu_node, "opos", "opos", &converted_node);
  }

  new_nodes.push_back(converted_node);
  new_nodes.push_back(new_relu_node);
  return new_nodes;
}

std::vector<NodeDef> ConvertConvfcBiasIdRelu(
    const NodeMatch &match, std::map<string, string> &inputs_to_rename,
    std::unordered_set<string> &nodes_to_ignore, const bool &quantize_ignored) {
  std::vector<NodeDef> new_nodes;

  const NodeDef &relu_node = match.node;
  const NodeDef &identity_node = match.inputs[0].node;
  const NodeDef &biasadd_node = match.inputs[0].inputs[0].node;
  const NodeDef &bias_node = match.inputs[0].inputs[0].inputs[1].node;
  const NodeDef &convfc_node = match.inputs[0].inputs[0].inputs[0].node;
  const NodeDef &weight_node =
      match.inputs[0].inputs[0].inputs[0].inputs[1].node;
  const NodeDef &input_node =
      match.inputs[0].inputs[0].inputs[0].inputs[0].node;
  CHECK(weight_node.op() == "Const")
      << "[DEPLOY ERROR] Expect const node for weight, but " << weight_node.op()
      << "(" << weight_node.name() << ") found.";
  CHECK(bias_node.op() == "Const")
      << "[DEPLOY ERROR] Expect const node for bias, but " << bias_node.op()
      << "(" << bias_node.name() << ") found.";

  DLOG_INFO(1) << "Convert convfc + bias + identity + relu: "
               << relu_node.name() << " + " << identity_node.name() << " + "
               << biasadd_node.name() << " <-- " << convfc_node.name() << " + "
               << bias_node.name();

  NodeDef converted_node = convfc_node;
  converted_node.mutable_input()->Clear();
  AddNodeInput(input_node.name(), &converted_node);
  CopyNodeAttr(weight_node, "value", "weights", &converted_node);
  CopyNodeAttr(bias_node, "value", "bias", &converted_node);

  NodeDef new_relu_node = relu_node;
  new_relu_node.set_input(0, converted_node.name());

  if (quantize_ignored) {
    SetNodeAttr("quantize_ignored", true, &converted_node);
    SetNodeAttr("quantize_ignored", true, &new_relu_node);
  } else {
    CHECK(weight_node.attr().count("wpos"))
        << "[DEPLOY ERROR] Cannot find quantize info for weights: "
        << weight_node.name();
    CopyNodeAttr(weight_node, "wpos", "wpos", &converted_node);

    CHECK(bias_node.attr().count("wpos"))
        << "[DEPLOY ERROR] Cannot find quantize info for bias: "
        << bias_node.name();
    CopyNodeAttr(bias_node, "wpos", "bpos", &converted_node);

    CHECK(relu_node.attr().count("opos"))
        << "[DEPLOY ERROR] Cannot find quantize info for relu: "
        << relu_node.name();
    CopyNodeAttr(relu_node, "opos", "opos", &converted_node);
  }

  new_nodes.push_back(converted_node);
  new_nodes.push_back(new_relu_node);
  return new_nodes;
}

std::vector<NodeDef> ConvertConvfcBias(
    const NodeMatch &match, std::map<string, string> &inputs_to_rename,
    std::unordered_set<string> &nodes_to_ignore, const bool &quantize_ignored) {
  std::vector<NodeDef> new_nodes;

  const NodeDef &biasadd_node = match.node;
  const NodeDef &bias_node = match.inputs[1].node;
  const NodeDef &convfc_node = match.inputs[0].node;
  const NodeDef &weight_node = match.inputs[0].inputs[1].node;
  const NodeDef &input_node = match.inputs[0].inputs[0].node;
  CHECK(weight_node.op() == "Const")
      << "[DEPLOY ERROR] Expect const node for weight, but " << weight_node.op()
      << "(" << weight_node.name() << ") found.";
  CHECK(bias_node.op() == "Const")
      << "[DEPLOY ERROR] Expect const node for bias, but " << bias_node.op()
      << "(" << bias_node.name() << ") found.";

  DLOG_INFO(1) << "Convert convfc + bias: " << biasadd_node.name() << " <-- "
               << convfc_node.name() << " + " << bias_node.name();

  NodeDef converted_node = convfc_node;
  converted_node.mutable_input()->Clear();
  AddNodeInput(input_node.name(), &converted_node);
  CopyNodeAttr(weight_node, "value", "weights", &converted_node);
  CopyNodeAttr(bias_node, "value", "bias", &converted_node);

  if (quantize_ignored) {
    SetNodeAttr("quantize_ignored", true, &converted_node);
  } else {
    CHECK(weight_node.attr().count("wpos"))
        << "[DEPLOY ERROR] Cannot find quantize info for weights: "
        << weight_node.name();
    CopyNodeAttr(weight_node, "wpos", "wpos", &converted_node);

    CHECK(bias_node.attr().count("wpos"))
        << "[DEPLOY ERROR] Cannot find quantize info for bias: "
        << bias_node.name();
    CopyNodeAttr(bias_node, "wpos", "bpos", &converted_node);

    CHECK(biasadd_node.attr().count("opos"))
        << "[DEPLOY ERROR] Cannot find quantize info for biasadd: "
        << biasadd_node.name();
    CopyNodeAttr(biasadd_node, "opos", "opos", &converted_node);
  }

  new_nodes.push_back(converted_node);
  inputs_to_rename[biasadd_node.name()] = converted_node.name();
  return new_nodes;
}

std::vector<NodeDef> ConvertConvfcRelu(
    const NodeMatch &match, std::map<string, string> &inputs_to_rename,
    std::unordered_set<string> &nodes_to_ignore, const bool &quantize_ignored) {
  std::vector<NodeDef> new_nodes;

  const NodeDef &relu_node = match.node;
  const NodeDef &convfc_node = match.inputs[0].node;
  const NodeDef &weight_node = match.inputs[0].inputs[1].node;
  const NodeDef &input_node = match.inputs[0].inputs[0].node;
  CHECK(weight_node.op() == "Const")
      << "[DEPLOY ERROR] Expect const node for weight, but " << weight_node.op()
      << "(" << weight_node.name() << ") found.";

  DLOG_INFO(1) << "Convert convfc + relu: " << relu_node.name() << " <-- "
               << convfc_node.name();

  NodeDef converted_node = convfc_node;
  converted_node.mutable_input()->Clear();
  AddNodeInput(input_node.name(), &converted_node);
  CopyNodeAttr(weight_node, "value", "weights", &converted_node);

  NodeDef new_relu_node = relu_node;
  new_relu_node.set_input(0, converted_node.name());

  if (quantize_ignored) {
    SetNodeAttr("quantize_ignored", true, &converted_node);
    SetNodeAttr("quantize_ignored", true, &new_relu_node);
  } else {
    CHECK(weight_node.attr().count("wpos"))
        << "[DEPLOY ERROR] Cannot find quantize info for weights: "
        << weight_node.name();
    CopyNodeAttr(weight_node, "wpos", "wpos", &converted_node);

    CHECK(relu_node.attr().count("opos"))
        << "[DEPLOY ERROR] Cannot find quantize info for relu: "
        << relu_node.name();
    CopyNodeAttr(relu_node, "opos", "opos", &converted_node);
  }

  new_nodes.push_back(converted_node);
  new_nodes.push_back(new_relu_node);
  return new_nodes;
}

std::vector<NodeDef> ConvertConvfc(const NodeMatch &match,
                                   std::map<string, string> &inputs_to_rename,
                                   std::unordered_set<string> &nodes_to_ignore,
                                   const bool &quantize_ignored) {
  std::vector<NodeDef> new_nodes;

  const NodeDef &convfc_node = match.node;
  const NodeDef &weight_node = match.inputs[1].node;
  const NodeDef &input_node = match.inputs[0].node;
  CHECK(weight_node.op() == "Const")
      << "[DEPLOY ERROR] Expect const node for weight, but " << weight_node.op()
      << "(" << weight_node.name() << ") found.";

  DLOG_INFO(1) << "Convert convfc: " << convfc_node.name();

  NodeDef converted_node = convfc_node;
  converted_node.mutable_input()->Clear();
  AddNodeInput(input_node.name(), &converted_node);
  CopyNodeAttr(weight_node, "value", "weights", &converted_node);

  if (quantize_ignored) {
    SetNodeAttr("quantize_ignored", true, &converted_node);
  } else {
    CHECK(weight_node.attr().count("wpos"))
        << "[DEPLOY ERROR] Cannot find quantize info for weights: "
        << weight_node.name();
    CopyNodeAttr(weight_node, "wpos", "wpos", &converted_node);
    // Don't need to copy opos, because it's already there
  }

  new_nodes.push_back(converted_node);
  return new_nodes;
}

std::vector<NodeDef> ConvertConv2dTransposeBiasRelu(
    const NodeMatch &match, std::map<string, string> &inputs_to_rename,
    std::unordered_set<string> &nodes_to_ignore, const bool &quantize_ignored) {
  std::vector<NodeDef> new_nodes;

  const NodeDef &relu_node = match.node;
  const NodeDef &biasadd_node = match.inputs[0].node;
  const NodeDef &bias_node = match.inputs[0].inputs[1].node;
  const NodeDef &conv2d_transpose_node = match.inputs[0].inputs[0].node;
  const NodeDef &weight_node = match.inputs[0].inputs[0].inputs[1].node;
  const NodeDef &input_node = match.inputs[0].inputs[0].inputs[2].node;

  CHECK(weight_node.op() == "Const")
      << "[DEPLOY ERROR] Expect const node for weight, but " << weight_node.op()
      << "(" << weight_node.name() << ") found.";
  CHECK(bias_node.op() == "Const")
      << "[DEPLOY ERROR] Expect const node for bias, but " << bias_node.op()
      << "(" << bias_node.name() << ") found.";

  DLOG_INFO(1) << "Convert conv2d_transpose + bias + relu: " << relu_node.name()
               << " + " << biasadd_node.name() << " <-- "
               << conv2d_transpose_node.name() << " + " << bias_node.name();

  NodeDef converted_node = conv2d_transpose_node;
  converted_node.set_op("Deconv2d");
  converted_node.mutable_input()->Clear();
  AddNodeInput(input_node.name(), &converted_node);
  CopyNodeAttr(weight_node, "value", "weights", &converted_node);
  CopyNodeAttr(bias_node, "value", "bias", &converted_node);

  NodeDef new_relu_node = relu_node;
  new_relu_node.set_input(0, converted_node.name());

  if (quantize_ignored) {
    SetNodeAttr("quantize_ignored", true, &converted_node);
    SetNodeAttr("quantize_ignored", true, &new_relu_node);
  } else {
    CHECK(weight_node.attr().count("wpos"))
        << "[DEPLOY ERROR] Cannot find quantize info for weights: "
        << weight_node.name();
    CopyNodeAttr(weight_node, "wpos", "wpos", &converted_node);

    CHECK(bias_node.attr().count("wpos"))
        << "[DEPLOY ERROR] Cannot find quantize info for bias: "
        << bias_node.name();
    CopyNodeAttr(bias_node, "wpos", "bpos", &converted_node);

    CHECK(relu_node.attr().count("opos"))
        << "[DEPLOY ERROR] Cannot find quantize info for relu: "
        << relu_node.name();
    CopyNodeAttr(relu_node, "opos", "opos", &converted_node);
  }

  new_nodes.push_back(converted_node);
  new_nodes.push_back(new_relu_node);
  return new_nodes;
}

std::vector<NodeDef> ConvertConv2dTransposeBias(
    const NodeMatch &match, std::map<string, string> &inputs_to_rename,
    std::unordered_set<string> &nodes_to_ignore, const bool &quantize_ignored) {
  std::vector<NodeDef> new_nodes;
  const NodeDef &biasadd_node = match.node;
  const NodeDef &bias_node = match.inputs[1].node;
  const NodeDef &conv2d_transpose_node = match.inputs[0].node;
  const NodeDef &weight_node = match.inputs[0].inputs[1].node;
  const NodeDef &input_node = match.inputs[0].inputs[2].node;

  CHECK(weight_node.op() == "Const")
      << "[DEPLOY ERROR] Expect const node for weight, but " << weight_node.op()
      << "(" << weight_node.name() << ") found.";
  CHECK(bias_node.op() == "Const")
      << "[DEPLOY ERROR] Expect const node for bias, but " << bias_node.op()
      << "(" << bias_node.name() << ") found.";

  DLOG_INFO(1) << "Convert convfc + bias: " << biasadd_node.name() << " <-- "
               << conv2d_transpose_node.name() << " + " << bias_node.name();

  NodeDef converted_node = conv2d_transpose_node;
  converted_node.set_op("Deconv2d");
  converted_node.mutable_input()->Clear();
  AddNodeInput(input_node.name(), &converted_node);
  CopyNodeAttr(weight_node, "value", "weights", &converted_node);
  CopyNodeAttr(bias_node, "value", "bias", &converted_node);

  if (quantize_ignored) {
    SetNodeAttr("quantize_ignored", true, &converted_node);
  } else {
    CHECK(weight_node.attr().count("wpos"))
        << "[DEPLOY ERROR] Cannot find quantize info for weights: "
        << weight_node.name();
    CopyNodeAttr(weight_node, "wpos", "wpos", &converted_node);

    CHECK(bias_node.attr().count("wpos"))
        << "[DEPLOY ERROR] Cannot find quantize info for bias: "
        << bias_node.name();
    CopyNodeAttr(bias_node, "wpos", "bpos", &converted_node);

    CHECK(biasadd_node.attr().count("opos"))
        << "[DEPLOY ERROR] Cannot find quantize info for biasadd: "
        << biasadd_node.name();
    CopyNodeAttr(biasadd_node, "opos", "opos", &converted_node);
  }

  new_nodes.push_back(converted_node);
  inputs_to_rename[biasadd_node.name()] = converted_node.name();
  return new_nodes;
}

std::vector<NodeDef> ConvertConv2dTransposeRelu(
    const NodeMatch &match, std::map<string, string> &inputs_to_rename,
    std::unordered_set<string> &nodes_to_ignore, const bool &quantize_ignored) {
  std::vector<NodeDef> new_nodes;

  const NodeDef &relu_node = match.node;
  const NodeDef &conv2d_transpose_node = match.inputs[0].node;
  const NodeDef &weight_node = match.inputs[0].inputs[1].node;
  const NodeDef &input_node = match.inputs[0].inputs[2].node;

  CHECK(weight_node.op() == "Const")
      << "[DEPLOY ERROR] Expect const node for weight, but " << weight_node.op()
      << "(" << weight_node.name() << ") found.";

  DLOG_INFO(1) << "Convert conv2d_transpose + relu: " << relu_node.name()
               << " <-- " << conv2d_transpose_node.name();

  NodeDef converted_node = conv2d_transpose_node;
  converted_node.set_op("Deconv2d");
  converted_node.mutable_input()->Clear();
  AddNodeInput(input_node.name(), &converted_node);
  CopyNodeAttr(weight_node, "value", "weights", &converted_node);

  NodeDef new_relu_node = relu_node;
  new_relu_node.set_input(0, converted_node.name());

  if (quantize_ignored) {
    SetNodeAttr("quantize_ignored", true, &converted_node);
    SetNodeAttr("quantize_ignored", true, &new_relu_node);
  } else {
    CHECK(weight_node.attr().count("wpos"))
        << "[DEPLOY ERROR] Cannot find quantize info for weights: "
        << weight_node.name();
    CopyNodeAttr(weight_node, "wpos", "wpos", &converted_node);

    CHECK(relu_node.attr().count("opos"))
        << "[DEPLOY ERROR] Cannot find quantize info for relu: "
        << relu_node.name();
    CopyNodeAttr(relu_node, "opos", "opos", &converted_node);
  }

  new_nodes.push_back(converted_node);
  new_nodes.push_back(new_relu_node);
  return new_nodes;
}

std::vector<NodeDef> ConvertConv2dTranspose(
    const NodeMatch &match, std::map<string, string> &inputs_to_rename,
    std::unordered_set<string> &nodes_to_ignore, const bool &quantize_ignored) {
  std::vector<NodeDef> new_nodes;
  const NodeDef &conv2d_transpose_node = match.node;
  const NodeDef &weight_node = match.inputs[1].node;
  const NodeDef &input_node = match.inputs[2].node;

  DLOG_INFO(1) << "Convert conv2d_transpose: " << conv2d_transpose_node.name()
               << " <-- " << input_node.name();

  NodeDef converted_node = conv2d_transpose_node;
  converted_node.set_op("Deconv2d");
  converted_node.mutable_input()->Clear();
  AddNodeInput(input_node.name(), &converted_node);
  CopyNodeAttr(weight_node, "value", "weights", &converted_node);

  if (quantize_ignored) {
    SetNodeAttr("quantize_ignored", true, &converted_node);
  } else {
    CHECK(weight_node.attr().count("wpos"))
        << "[DEPLOY ERROR] Cannot find quantize info for weights: "
        << weight_node.name();
    CopyNodeAttr(weight_node, "wpos", "wpos", &converted_node);
    // Don't need to copy opos, because it's already there
  }

  new_nodes.push_back(converted_node);
  return new_nodes;
}

std::vector<NodeDef> ConvertConvfcBiasFusedLeakyRelu(
    const NodeMatch &match, std::map<string, string> &inputs_to_rename,
    std::unordered_set<string> &nodes_to_ignore, const bool &quantize_ignored) {
  std::vector<NodeDef> new_nodes;

  const NodeDef &leakyrelu_node = match.node;
  const NodeDef &biasadd_node = match.inputs[0].node;
  const NodeDef &bias_node = match.inputs[0].inputs[1].node;
  const NodeDef &convfc_node = match.inputs[0].inputs[0].node;
  const NodeDef &weight_node = match.inputs[0].inputs[0].inputs[1].node;
  const NodeDef &input_node = match.inputs[0].inputs[0].inputs[0].node;
  CHECK(weight_node.op() == "Const")
      << "[DEPLOY ERROR] Expect const node for weight, but " << weight_node.op()
      << "(" << weight_node.name() << ") found.";
  CHECK(bias_node.op() == "Const")
      << "[DEPLOY ERROR] Expect const node for bias, but " << bias_node.op()
      << "(" << bias_node.name() << ") found.";

  DLOG_INFO(1) << "Convert convfc + bias + leakyrelu: " << leakyrelu_node.name()
               << "(" << leakyrelu_node.op() << ") <-- " << biasadd_node.name()
               << "(" << biasadd_node.op() << ") <-- " << convfc_node.name()
               << "(" << convfc_node.op() << ")";

  NodeDef converted_convfc_node = convfc_node;
  converted_convfc_node.mutable_input()->Clear();
  AddNodeInput(input_node.name(), &converted_convfc_node);
  CopyNodeAttr(weight_node, "value", "weights", &converted_convfc_node);
  CopyNodeAttr(bias_node, "value", "bias", &converted_convfc_node);

  if (quantize_ignored) {
    SetNodeAttr("quantize_ignored", true, &converted_convfc_node);
  } else {
    CHECK(weight_node.attr().count("wpos"))
        << "[DEPLOY ERROR] Cannot find quantize info for weights: "
        << weight_node.name();
    CopyNodeAttr(weight_node, "wpos", "wpos", &converted_convfc_node);

    CHECK(bias_node.attr().count("wpos"))
        << "[DEPLOY ERROR] Cannot find quantize info for bias: "
        << bias_node.name();
    CopyNodeAttr(bias_node, "wpos", "bpos", &converted_convfc_node);

    CHECK(leakyrelu_node.attr().count("opos"))
        << "[DEPLOY ERROR] Cannot find quantize info for leakyrelu: "
        << bias_node.name();
    CopyNodeAttr(leakyrelu_node, "opos", "opos", &converted_convfc_node);
  }

  NodeDef converted_leakyrelu_node = leakyrelu_node;
  converted_leakyrelu_node.set_op("LeakyReLU");
  converted_leakyrelu_node.mutable_input()->Clear();
  AddNodeInput(converted_convfc_node.name(), &converted_leakyrelu_node);

  if (quantize_ignored) {
    SetNodeAttr("quantize_ignored", true, &converted_leakyrelu_node);
  }
  // Don't need to copy opos, because it's already there

  new_nodes.push_back(converted_convfc_node);
  new_nodes.push_back(converted_leakyrelu_node);
  return new_nodes;
}

std::vector<NodeDef> ConvertConvfcFusedLeakyRelu(
    const NodeMatch &match, std::map<string, string> &inputs_to_rename,
    std::unordered_set<string> &nodes_to_ignore, const bool &quantize_ignored) {
  std::vector<NodeDef> new_nodes;

  const NodeDef &leakyrelu_node = match.node;
  const NodeDef &convfc_node = match.inputs[0].node;
  const NodeDef &weight_node = match.inputs[0].inputs[1].node;
  const NodeDef &input_node = match.inputs[0].inputs[0].node;
  CHECK(weight_node.op() == "Const")
      << "[DEPLOY ERROR] Expect const node for weight, but " << weight_node.op()
      << "(" << weight_node.name() << ") found.";

  DLOG_INFO(1) << "Convert convfc + leakyrelu: " << leakyrelu_node.name() << "("
               << leakyrelu_node.op() << ") <-- " << convfc_node.name() << "("
               << convfc_node.op() << ")";

  NodeDef converted_convfc_node = convfc_node;
  converted_convfc_node.mutable_input()->Clear();
  AddNodeInput(input_node.name(), &converted_convfc_node);
  CopyNodeAttr(weight_node, "value", "weights", &converted_convfc_node);

  if (quantize_ignored) {
    SetNodeAttr("quantize_ignored", true, &converted_convfc_node);
  } else {
    CHECK(weight_node.attr().count("wpos"))
        << "[DEPLOY ERROR] Cannot find quantize info for weights: "
        << weight_node.name();
    CopyNodeAttr(weight_node, "wpos", "wpos", &converted_convfc_node);

    CHECK(leakyrelu_node.attr().count("opos"))
        << "[DEPLOY ERROR] Cannot find quantize info for leakyrelu: "
        << leakyrelu_node.name();
    CopyNodeAttr(leakyrelu_node, "opos", "opos", &converted_convfc_node);
  }

  NodeDef converted_leakyrelu_node = leakyrelu_node;
  converted_leakyrelu_node.set_op("LeakyReLU");
  converted_leakyrelu_node.mutable_input()->Clear();
  AddNodeInput(converted_convfc_node.name(), &converted_leakyrelu_node);

  if (quantize_ignored) {
    SetNodeAttr("quantize_ignored", true, &converted_leakyrelu_node);
  }
  // Don't need to copy opos, because it's already there

  new_nodes.push_back(converted_convfc_node);
  new_nodes.push_back(converted_leakyrelu_node);
  return new_nodes;
}

std::vector<NodeDef> ConvertFusedLeakyRelu(
    const NodeMatch &match, std::map<string, string> &inputs_to_rename,
    std::unordered_set<string> &nodes_to_ignore, const bool &quantize_ignored) {
  std::vector<NodeDef> new_nodes;
  const NodeDef &leakyrelu_node = match.node;
  const NodeDef &input_node = match.inputs[0].node;

  NodeDef converted_node = leakyrelu_node;
  converted_node.set_op("LeakyReLU");
  converted_node.mutable_input()->Clear();
  AddNodeInput(input_node.name(), &converted_node);

  if (quantize_ignored) {
    SetNodeAttr("quantize_ignored", true, &converted_node);
  }
  // Don't need to copy opos, because it's already there

  new_nodes.push_back(converted_node);
  return new_nodes;
}

std::vector<NodeDef> ConvertConvfcBiasKerasLeakyRelu(
    const NodeMatch &match, std::map<string, string> &inputs_to_rename,
    std::unordered_set<string> &nodes_to_ignore, const bool &quantize_ignored) {
  std::vector<NodeDef> new_nodes;
  const NodeDef &leakyrelu_node = match.node;
  const NodeDef &alpha_node = match.inputs[1].inputs[0].node;
  const NodeDef &biasadd_node = match.inputs[0].inputs[0].node;
  const NodeDef &bias_node = match.inputs[0].inputs[0].inputs[1].node;
  const NodeDef &convfc_node = match.inputs[0].inputs[0].inputs[0].node;
  const NodeDef &weight_node =
      match.inputs[0].inputs[0].inputs[0].inputs[1].node;
  const NodeDef &input_node =
      match.inputs[0].inputs[0].inputs[0].inputs[0].node;
  CHECK(weight_node.op() == "Const")
      << "[DEPLOY ERROR] Expect const node for weight, but " << weight_node.op()
      << "(" << weight_node.name() << ") found.";
  CHECK(bias_node.op() == "Const")
      << "[DEPLOY ERROR] Expect const node for bias, but " << bias_node.op()
      << "(" << bias_node.name() << ") found.";

  DLOG_INFO(1) << "Convert convfc + bias + keras_leakyrelu: "
               << leakyrelu_node.name() << "(" << leakyrelu_node.op()
               << ") <-- " << biasadd_node.name() << "(" << biasadd_node.op()
               << ") <-- " << convfc_node.name() << "(" << convfc_node.op()
               << ")";

  NodeDef converted_convfc_node = convfc_node;
  converted_convfc_node.mutable_input()->Clear();
  AddNodeInput(input_node.name(), &converted_convfc_node);
  CopyNodeAttr(weight_node, "value", "weights", &converted_convfc_node);
  CopyNodeAttr(bias_node, "value", "bias", &converted_convfc_node);

  if (quantize_ignored) {
    SetNodeAttr("quantize_ignored", true, &converted_convfc_node);
  } else {
    CHECK(weight_node.attr().count("wpos"))
        << "[DEPLOY ERROR] Cannot find quantize info for weights: "
        << weight_node.name();
    CopyNodeAttr(weight_node, "wpos", "wpos", &converted_convfc_node);

    CHECK(bias_node.attr().count("wpos"))
        << "[DEPLOY ERROR] Cannot find quantize info for bias: "
        << bias_node.name();
    CopyNodeAttr(bias_node, "wpos", "bpos", &converted_convfc_node);

    CHECK(leakyrelu_node.attr().count("opos"))
        << "[DEPLOY ERROR] Cannot find quantize info for leakyrelu: "
        << bias_node.name();
    CopyNodeAttr(leakyrelu_node, "opos", "opos", &converted_convfc_node);
  }

  NodeDef converted_leakyrelu_node = leakyrelu_node;
  converted_leakyrelu_node.set_op("LeakyReLU");
  converted_leakyrelu_node.mutable_input()->Clear();
  AddNodeInput(converted_convfc_node.name(), &converted_leakyrelu_node);
  CopyNodeAttr(alpha_node, "value", "alpha", &converted_leakyrelu_node);

  if (quantize_ignored) {
    SetNodeAttr("quantize_ignored", true, &converted_leakyrelu_node);
  } else {
    CHECK(alpha_node.attr().count("wpos"))
        << "[DEPLOY ERROR] Cannot find quantize info for alpha: "
        << alpha_node.name();
    CopyNodeAttr(alpha_node, "wpos", "wpos", &converted_leakyrelu_node);
    // Don't need to copy opos, because it's already there
  }
  new_nodes.push_back(converted_convfc_node);
  new_nodes.push_back(converted_leakyrelu_node);
  return new_nodes;
}

std::vector<NodeDef> ConvertConvfcKerasLeakyRelu(
    const NodeMatch &match, std::map<string, string> &inputs_to_rename,
    std::unordered_set<string> &nodes_to_ignore, const bool &quantize_ignored) {
  std::vector<NodeDef> new_nodes;
  const NodeDef &leakyrelu_node = match.node;
  const NodeDef &alpha_node = match.inputs[1].inputs[0].node;
  const NodeDef &convfc_node = match.inputs[0].inputs[0].node;
  const NodeDef &weight_node = match.inputs[0].inputs[0].inputs[1].node;
  const NodeDef &input_node = match.inputs[0].inputs[0].inputs[0].node;
  CHECK(weight_node.op() == "Const")
      << "[DEPLOY ERROR] Expect const node for weight, but " << weight_node.op()
      << "(" << weight_node.name() << ") found.";

  DLOG_INFO(1) << "Convert convfc + keras_leakyrelu: " << leakyrelu_node.name()
               << "(" << leakyrelu_node.op() << ") <-- " << convfc_node.name()
               << "(" << convfc_node.op() << ")";

  NodeDef converted_convfc_node = convfc_node;
  converted_convfc_node.mutable_input()->Clear();
  AddNodeInput(input_node.name(), &converted_convfc_node);
  CopyNodeAttr(weight_node, "value", "weights", &converted_convfc_node);

  if (quantize_ignored) {
    SetNodeAttr("quantize_ignored", true, &converted_convfc_node);
  } else {
    CHECK(weight_node.attr().count("wpos"))
        << "[DEPLOY ERROR] Cannot find quantize info for weights: "
        << weight_node.name();
    CopyNodeAttr(weight_node, "wpos", "wpos", &converted_convfc_node);

    CHECK(leakyrelu_node.attr().count("opos"))
        << "[DEPLOY ERROR] Cannot find quantize info for leakyrelu: "
        << leakyrelu_node.name();
    CopyNodeAttr(leakyrelu_node, "opos", "opos", &converted_convfc_node);
  }

  NodeDef converted_leakyrelu_node = leakyrelu_node;
  converted_leakyrelu_node.set_op("LeakyReLU");
  converted_leakyrelu_node.mutable_input()->Clear();
  AddNodeInput(converted_convfc_node.name(), &converted_leakyrelu_node);
  CopyNodeAttr(alpha_node, "value", "alpha", &converted_leakyrelu_node);

  if (quantize_ignored) {
    SetNodeAttr("quantize_ignored", true, &converted_leakyrelu_node);
  } else {
    CHECK(alpha_node.attr().count("wpos"))
        << "[DEPLOY ERROR] Cannot find quantize info for alpha: "
        << alpha_node.name();
    CopyNodeAttr(alpha_node, "wpos", "wpos", &converted_leakyrelu_node);
    // Don't need to copy opos, because it's already there
  }
  new_nodes.push_back(converted_convfc_node);
  new_nodes.push_back(converted_leakyrelu_node);
  return new_nodes;
}

std::vector<NodeDef> ConvertKerasLeakyRelu(
    const NodeMatch &match, std::map<string, string> &inputs_to_rename,
    std::unordered_set<string> &nodes_to_ignore, const bool &quantize_ignored) {
  std::vector<NodeDef> new_nodes;
  const NodeDef &leakyrelu_node = match.node;
  const NodeDef &alpha_node = match.inputs[1].inputs[0].node;
  const NodeDef &input_node = match.inputs[0].inputs[0].node;

  NodeDef converted_node = leakyrelu_node;
  converted_node.set_op("LeakyReLU");
  converted_node.mutable_input()->Clear();
  AddNodeInput(input_node.name(), &converted_node);
  CopyNodeAttr(alpha_node, "value", "alpha", &converted_node);

  if (quantize_ignored) {
    SetNodeAttr("quantize_ignored", true, &converted_node);
  } else {
    CHECK(alpha_node.attr().count("wpos"))
        << "[DEPLOY ERROR] Cannot find quantize info for alpha: "
        << alpha_node.name();
    CopyNodeAttr(alpha_node, "wpos", "wpos", &converted_node);
    // Don't need to copy opos, because it's already there
  }
  new_nodes.push_back(converted_node);
  return new_nodes;
}

std::vector<NodeDef> ConvertConvfcBiasLeakyRelu(
    const NodeMatch &match, std::map<string, string> &inputs_to_rename,
    std::unordered_set<string> &nodes_to_ignore, const bool &quantize_ignored) {
  std::vector<NodeDef> new_nodes;

  const NodeDef &leakyrelu_node = match.node;
  const NodeDef &alpha_node = match.inputs[0].inputs[0].node;
  const NodeDef &biasadd_node = match.inputs[1].node;
  const NodeDef &bias_node = match.inputs[1].inputs[1].node;
  const NodeDef &convfc_node = match.inputs[1].inputs[0].node;
  const NodeDef &weight_node = match.inputs[1].inputs[0].inputs[1].node;
  const NodeDef &input_node = match.inputs[1].inputs[0].inputs[0].node;
  CHECK(weight_node.op() == "Const")
      << "[DEPLOY ERROR] Expect const node for weight, but " << weight_node.op()
      << "(" << weight_node.name() << ") found.";
  CHECK(bias_node.op() == "Const")
      << "[DEPLOY ERROR] Expect const node for bias, but " << bias_node.op()
      << "(" << bias_node.name() << ") found.";

  DLOG_INFO(1) << "Convert convfc + bias + leakyrelu: " << leakyrelu_node.name()
               << "(" << leakyrelu_node.op() << ") <-- " << biasadd_node.name()
               << "(" << biasadd_node.op() << ") <-- " << convfc_node.name()
               << "(" << convfc_node.op() << ")";

  NodeDef converted_convfc_node = convfc_node;
  converted_convfc_node.mutable_input()->Clear();
  AddNodeInput(input_node.name(), &converted_convfc_node);
  CopyNodeAttr(weight_node, "value", "weights", &converted_convfc_node);
  CopyNodeAttr(bias_node, "value", "bias", &converted_convfc_node);

  if (quantize_ignored) {
    SetNodeAttr("quantize_ignored", true, &converted_convfc_node);
  } else {
    CHECK(weight_node.attr().count("wpos"))
        << "[DEPLOY ERROR] Cannot find quantize info for weights: "
        << weight_node.name();
    CopyNodeAttr(weight_node, "wpos", "wpos", &converted_convfc_node);

    CHECK(bias_node.attr().count("wpos"))
        << "[DEPLOY ERROR] Cannot find quantize info for bias: "
        << bias_node.name();
    CopyNodeAttr(bias_node, "wpos", "bpos", &converted_convfc_node);

    CHECK(leakyrelu_node.attr().count("opos"))
        << "[DEPLOY ERROR] Cannot find quantize info for leakyrelu: "
        << bias_node.name();
    CopyNodeAttr(leakyrelu_node, "opos", "opos", &converted_convfc_node);
  }

  NodeDef converted_leakyrelu_node = leakyrelu_node;
  converted_leakyrelu_node.set_op("LeakyReLU");
  converted_leakyrelu_node.mutable_input()->Clear();
  AddNodeInput(converted_convfc_node.name(), &converted_leakyrelu_node);
  CopyNodeAttr(alpha_node, "value", "alpha", &converted_leakyrelu_node);

  if (quantize_ignored) {
    SetNodeAttr("quantize_ignored", true, &converted_leakyrelu_node);
  } else {
    CHECK(alpha_node.attr().count("wpos"))
        << "[DEPLOY ERROR] Cannot find quantize info for alpha: "
        << alpha_node.name();
    CopyNodeAttr(alpha_node, "wpos", "wpos", &converted_leakyrelu_node);
    // Don't need to copy opos, because it's already there
  }

  new_nodes.push_back(converted_convfc_node);
  new_nodes.push_back(converted_leakyrelu_node);
  return new_nodes;
}

std::vector<NodeDef> ConvertConvfcLeakyRelu(
    const NodeMatch &match, std::map<string, string> &inputs_to_rename,
    std::unordered_set<string> &nodes_to_ignore, const bool &quantize_ignored) {
  std::vector<NodeDef> new_nodes;

  const NodeDef &leakyrelu_node = match.node;
  const NodeDef &alpha_node = match.inputs[0].inputs[0].node;
  const NodeDef &convfc_node = match.inputs[1].node;
  const NodeDef &weight_node = match.inputs[1].inputs[1].node;
  const NodeDef &input_node = match.inputs[1].inputs[0].node;
  CHECK(weight_node.op() == "Const")
      << "[DEPLOY ERROR] Expect const node for weight, but " << weight_node.op()
      << "(" << weight_node.name() << ") found.";

  DLOG_INFO(1) << "Convert convfc + leakyrelu: " << leakyrelu_node.name() << "("
               << leakyrelu_node.op() << ") <-- " << convfc_node.name() << "("
               << convfc_node.op() << ")";

  NodeDef converted_convfc_node = convfc_node;
  converted_convfc_node.mutable_input()->Clear();
  AddNodeInput(input_node.name(), &converted_convfc_node);
  CopyNodeAttr(weight_node, "value", "weights", &converted_convfc_node);

  if (quantize_ignored) {
    SetNodeAttr("quantize_ignored", true, &converted_convfc_node);
  } else {
    CHECK(weight_node.attr().count("wpos"))
        << "[DEPLOY ERROR] Cannot find quantize info for weights: "
        << weight_node.name();
    CopyNodeAttr(weight_node, "wpos", "wpos", &converted_convfc_node);

    CHECK(leakyrelu_node.attr().count("opos"))
        << "[DEPLOY ERROR] Cannot find quantize info for leakyrelu: "
        << leakyrelu_node.name();
    CopyNodeAttr(leakyrelu_node, "opos", "opos", &converted_convfc_node);
  }

  NodeDef converted_leakyrelu_node = leakyrelu_node;
  converted_leakyrelu_node.set_op("LeakyReLU");
  converted_leakyrelu_node.mutable_input()->Clear();
  AddNodeInput(converted_convfc_node.name(), &converted_leakyrelu_node);
  CopyNodeAttr(alpha_node, "value", "alpha", &converted_leakyrelu_node);

  if (quantize_ignored) {
    SetNodeAttr("quantize_ignored", true, &converted_leakyrelu_node);
  } else {
    CHECK(alpha_node.attr().count("wpos"))
        << "[DEPLOY ERROR] Cannot find quantize info for alpha: "
        << alpha_node.name();
    CopyNodeAttr(alpha_node, "wpos", "wpos", &converted_leakyrelu_node);
    // Don't need to copy opos, because it's already there
  }

  new_nodes.push_back(converted_convfc_node);
  new_nodes.push_back(converted_leakyrelu_node);
  return new_nodes;
}

std::vector<NodeDef> ConvertLeakyRelu(
    const NodeMatch &match, std::map<string, string> &inputs_to_rename,
    std::unordered_set<string> &nodes_to_ignore, const bool &quantize_ignored) {
  std::vector<NodeDef> new_nodes;
  const NodeDef &leakyrelu_node = match.node;
  const NodeDef &alpha_node = match.inputs[0].inputs[0].node;
  const NodeDef &input_node = match.inputs[1].node;

  NodeDef converted_node = leakyrelu_node;
  converted_node.set_op("LeakyReLU");
  converted_node.mutable_input()->Clear();
  AddNodeInput(input_node.name(), &converted_node);
  CopyNodeAttr(alpha_node, "value", "alpha", &converted_node);

  if (quantize_ignored) {
    SetNodeAttr("quantize_ignored", true, &converted_node);
  } else {
    CHECK(alpha_node.attr().count("wpos"))
        << "[DEPLOY ERROR] Cannot find quantize info for alpha: "
        << alpha_node.name();
    CopyNodeAttr(alpha_node, "wpos", "wpos", &converted_node);
    // Don't need to copy opos, because it's already there
  }
  new_nodes.push_back(converted_node);
  return new_nodes;
}

std::vector<NodeDef> ConvertUpsampling(
    const NodeMatch &match, std::map<string, string> &inputs_to_rename,
    std::unordered_set<string> &nodes_to_ignore, const bool &quantize_ignored) {
  std::vector<NodeDef> new_nodes;
  const NodeDef &resize_node = match.node;
  const NodeDef &input_node = match.inputs[0].node;
  const NodeDef &resize_value_node = match.inputs[1].inputs[1].node;

  DLOG_INFO(1) << "Convert upsampling: " << resize_node.name() << " <-- "
               << input_node.name();

  NodeDef converted_node = resize_node;
  converted_node.set_op("DeephiResize");
  converted_node.mutable_input()->Clear();
  AddNodeInput(input_node.name(), &converted_node);

  Tensor resize_value_tensor;
  if (!resize_value_tensor.FromProto(
          resize_value_node.attr().at("value").tensor())) {
    LOG(FATAL) << "Decoding tensor failed for node: "
               << resize_value_node.name();
  }
  const int *resize_value_tensor_value = resize_value_tensor.flat<int>().data();
  SetNodeAttr("scale_h", resize_value_tensor_value[0], &converted_node);
  SetNodeAttr("scale_w", resize_value_tensor_value[1], &converted_node);
  string resize_type;
  if (resize_node.op() == "ResizeBilinear") {
    resize_type = "Bilinear";
  } else if (resize_node.op() == "ResizeNearestNeighbor") {
    resize_type = "Nearest";
  }
  SetNodeAttr("resize_type", resize_type, &converted_node);

  if (quantize_ignored) {
    SetNodeAttr("quantize_ignored", true, &converted_node);
  }
  // Don't need to copy opos, because it's already there

  new_nodes.push_back(converted_node);
  return new_nodes;
}

std::vector<NodeDef> ConvertResize(const NodeMatch &match,
                                   std::map<string, string> &inputs_to_rename,
                                   std::unordered_set<string> &nodes_to_ignore,
                                   const bool &quantize_ignored) {
  std::vector<NodeDef> new_nodes;
  const NodeDef &resize_node = match.node;
  const NodeDef &input_node = match.inputs[0].node;
  const NodeDef &resize_value_node = match.inputs[1].node;

  DLOG_INFO(1) << "Convert resize: " << resize_node.name() << "("
               << resize_node.op() << ") <-- " << input_node.name();

  NodeDef converted_node = resize_node;
  converted_node.set_op("DeephiResize");
  converted_node.mutable_input()->Clear();
  AddNodeInput(input_node.name(), &converted_node);

  Tensor resize_value_tensor;
  if (!resize_value_tensor.FromProto(
          resize_value_node.attr().at("value").tensor())) {
    LOG(FATAL) << "Decoding tensor failed for node: "
               << resize_value_node.name();
  }
  const int *resize_value_tensor_value = resize_value_tensor.flat<int>().data();
  SetNodeAttr("size_h", resize_value_tensor_value[0], &converted_node);
  SetNodeAttr("size_w", resize_value_tensor_value[1], &converted_node);
  string resize_type;
  if (resize_node.op() == "ResizeBilinear") {
    resize_type = "Bilinear";
  } else if (resize_node.op() == "ResizeNearestNeighbor") {
    resize_type = "Nearest";
  }
  SetNodeAttr("resize_type", resize_type, &converted_node);

  if (quantize_ignored) {
    SetNodeAttr("quantize_ignored", true, &converted_node);
  }
  // Don't need to copy opos, because it's already there

  new_nodes.push_back(converted_node);
  return new_nodes;
}

std::vector<NodeDef> ConvertTPUNearestNeighborUpsampling(
    const NodeMatch &match, std::map<string, string> &inputs_to_rename,
    std::unordered_set<string> &nodes_to_ignore, const bool &quantize_ignored) {
  std::vector<NodeDef> new_nodes;
  const NodeDef &output_reshape_node = match.node;
  const NodeDef &input_node = match.inputs[0].inputs[0].inputs[0].node;
  const NodeDef &resize_value_node = match.inputs[0].inputs[1].node;

  DLOG_INFO(1) << "Convert tpu nearest neighbor upsampling: "
               << output_reshape_node.name() << " <-- " << input_node.name();

  NodeDef converted_node = output_reshape_node;
  converted_node.set_op("DeephiResize");
  converted_node.mutable_input()->Clear();
  AddNodeInput(input_node.name(), &converted_node);

  Tensor resize_value_tensor;
  if (!resize_value_tensor.FromProto(
          resize_value_node.attr().at("value").tensor())) {
    LOG(FATAL) << "Decoding tensor failed for node: "
               << resize_value_node.name();
  }
  SetNodeAttr("scale_h", resize_value_tensor.dim_size(2), &converted_node);
  SetNodeAttr("scale_w", resize_value_tensor.dim_size(4), &converted_node);
  string resize_type = "Nearest";
  SetNodeAttr("resize_type", resize_type, &converted_node);

  if (quantize_ignored) {
    SetNodeAttr("quantize_ignored", true, &converted_node);
  }
  // Don't need to copy opos, because it's already there

  new_nodes.push_back(converted_node);
  return new_nodes;
}

std::vector<NodeDef> ConvertBatchNormRelu(
    const NodeMatch &match, std::map<string, string> &inputs_to_rename,
    std::unordered_set<string> &nodes_to_ignore, const bool &quantize_ignored) {
  std::vector<NodeDef> new_nodes;
  const NodeDef &relu_node = match.node;
  const NodeDef &add_node = match.inputs[0].node;
  const NodeDef &offset_node = match.inputs[0].inputs[1].node;
  const NodeDef &mul_node = match.inputs[0].inputs[0].node;
  const NodeDef &scale_node = match.inputs[0].inputs[0].inputs[1].node;
  const NodeDef &input_node = match.inputs[0].inputs[0].inputs[0].node;

  DLOG_INFO(1) << "Convert batchnorm + relu: " << relu_node.name() << " <-- "
               << add_node.name();

  NodeDef converted_node = add_node;
  converted_node.set_op("Scale");
  converted_node.mutable_input()->Clear();
  AddNodeInput(input_node.name(), &converted_node);
  SetNodeAttr("T", DT_FLOAT, &converted_node);
  CopyNodeAttr(scale_node, "value", "weights", &converted_node);
  CopyNodeAttr(offset_node, "value", "bias", &converted_node);

  NodeDef new_relu_node = relu_node;
  new_relu_node.set_input(0, converted_node.name());

  if (quantize_ignored) {
    SetNodeAttr("quantize_ignored", true, &converted_node);
  } else {
    CHECK(scale_node.attr().count("wpos"))
        << "[DEPLOY ERROR] Cannnot find quantize info for weights: "
        << scale_node.name();
    CopyNodeAttr(scale_node, "wpos", "wpos", &converted_node);

    CHECK(offset_node.attr().count("wpos"))
        << "[DEPLOY ERROR] Cannnot find quantize info for bias: "
        << offset_node.name();
    CopyNodeAttr(offset_node, "wpos", "bpos", &converted_node);

    CHECK(relu_node.attr().count("opos"))
        << "[DEPLOY ERROR] Cannnot find quantize info for relu: "
        << relu_node.name();
    CopyNodeAttr(relu_node, "opos", "opos", &converted_node);
  }

  DLOG_WARNING
      << "Batchnorm Node (" << add_node.name() << " + " << mul_node.name()
      << ") is not folded. It will be converted to a Scale node ("
      << converted_node.name()
      << ") to deploy on DPU. This may cause accuracy decrease and error "
         "for DPU compiler.";

  new_nodes.push_back(converted_node);
  new_nodes.push_back(new_relu_node);
  return new_nodes;
}

std::vector<NodeDef> ConvertBatchNorm(
    const NodeMatch &match, std::map<string, string> &inputs_to_rename,
    std::unordered_set<string> &nodes_to_ignore, const bool &quantize_ignored) {
  std::vector<NodeDef> new_nodes;
  const NodeDef &add_node = match.node;
  const NodeDef &offset_node = match.inputs[1].node;
  const NodeDef &mul_node = match.inputs[0].node;
  const NodeDef &scale_node = match.inputs[0].inputs[1].node;
  const NodeDef &input_node = match.inputs[0].inputs[0].node;

  DLOG_INFO(1) << "Convert batchnorm: " << add_node.name() << " <-- "
               << mul_node.name();

  NodeDef converted_node = add_node;
  converted_node.set_op("Scale");
  converted_node.mutable_input()->Clear();
  AddNodeInput(input_node.name(), &converted_node);
  SetNodeAttr("T", DT_FLOAT, &converted_node);
  CopyNodeAttr(scale_node, "value", "weights", &converted_node);
  CopyNodeAttr(offset_node, "value", "bias", &converted_node);

  if (quantize_ignored) {
    SetNodeAttr("quantize_ignored", true, &converted_node);
  } else {
    CHECK(scale_node.attr().count("wpos"))
        << "[DEPLOY ERROR] Cannnot find quantize info for weights: "
        << scale_node.name();
    CopyNodeAttr(scale_node, "wpos", "wpos", &converted_node);

    CHECK(offset_node.attr().count("wpos"))
        << "[DEPLOY ERROR] Cannnot find quantize info for bias: "
        << offset_node.name();
    CopyNodeAttr(offset_node, "wpos", "bpos", &converted_node);

    CHECK(add_node.attr().count("opos"))
        << "[DEPLOY ERROR] Cannnot find quantize info for add: "
        << add_node.name();
    CopyNodeAttr(add_node, "opos", "opos", &converted_node);
  }

  DLOG_WARNING
      << "Batchnorm Node (" << add_node.name() << " + " << mul_node.name()
      << ") is not folded. It will be converted to a Scale node ("
      << converted_node.name()
      << ") to deploy on DPU. This may cause accuracy decrease and error "
         "for DPU compiler.";

  new_nodes.push_back(converted_node);
  return new_nodes;
}

std::vector<NodeDef> ConvertAtrousConvBiasRelu(
    const NodeMatch &match, std::map<string, string> &inputs_to_rename,
    std::unordered_set<string> &nodes_to_ignore, const bool &quantize_ignored) {
  std::vector<NodeDef> new_nodes;
  const NodeDef &relu_node = match.node;
  const NodeDef &bias_node = match.inputs[0].inputs[1].node;
  const NodeDef &atrous_conv_node = match.inputs[0].inputs[0].inputs[0].node;
  const NodeDef &weights_node =
      match.inputs[0].inputs[0].inputs[0].inputs[1].node;
  const NodeDef &space_to_batch_block_shape_node =
      match.inputs[0].inputs[0].inputs[0].inputs[0].inputs[1].node;
  const NodeDef &space_to_batch_paddings_node =
      match.inputs[0].inputs[0].inputs[0].inputs[0].inputs[2].node;
  const NodeDef &batch_to_space_crops_node =
      match.inputs[0].inputs[0].inputs[2].node;
  const NodeDef &input_node =
      match.inputs[0].inputs[0].inputs[0].inputs[0].inputs[0].node;

  DLOG_INFO(1) << "Convert atrous_conv_bias_relu: " << relu_node.name()
               << " <-- " << atrous_conv_node.name();

  NodeDef converted_node = atrous_conv_node;
  converted_node.mutable_input()->Clear();
  AddNodeInput(input_node.name(), &converted_node);
  CopyNodeAttr(weights_node, "value", "weights", &converted_node);
  CopyNodeAttr(bias_node, "value", "bias", &converted_node);

  // Set dilations
  Tensor block_shape =
      GetNodeTensorAttr(space_to_batch_block_shape_node, "value");
  const int32 block_height = block_shape.flat<int32>()(0);
  const int32 block_width = block_shape.flat<int32>()(1);
  std::vector<int> dilations;
  auto dilations_list = atrous_conv_node.attr().at("dilations").list();
  for (auto i = 0; i < dilations_list.i_size(); i++) {
    dilations.push_back(dilations_list.i(i));
  }
  dilations[1] = block_height;
  dilations[2] = block_width;
  SetNodeAttr("dilations", dilations, &converted_node);

  // Set paddings
  Tensor paddings = GetNodeTensorAttr(space_to_batch_paddings_node, "value");
  SetNodeTensorAttr<int>("space_to_batch_paddings", paddings, &converted_node);

  // Set crops
  Tensor crops = GetNodeTensorAttr(batch_to_space_crops_node, "value");
  SetNodeTensorAttr<int>("batch_to_space_crops", crops, &converted_node);

  NodeDef new_relu_node = relu_node;
  new_relu_node.set_input(0, converted_node.name());

  if (quantize_ignored) {
    SetNodeAttr("quantize_ignored", true, &converted_node);
    SetNodeAttr("quantize_ignored", true, &new_relu_node);
  } else {
    CHECK(weights_node.attr().count("wpos"))
        << "[DEPLOY ERROR] Cannnot find quantize info for weights: "
        << weights_node.name();
    CopyNodeAttr(weights_node, "wpos", "wpos", &converted_node);

    CHECK(bias_node.attr().count("wpos"))
        << "[DEPLOY ERROR] Cannnot find quantize info for bias: "
        << bias_node.name();
    CopyNodeAttr(bias_node, "wpos", "bpos", &converted_node);

    CHECK(relu_node.attr().count("opos"))
        << "[DEPLOY ERROR] Cannnot find quantize info for relu: "
        << relu_node.name();
    CopyNodeAttr(relu_node, "opos", "opos", &converted_node);
  }

  new_nodes.push_back(converted_node);
  new_nodes.push_back(new_relu_node);
  return new_nodes;
}

std::vector<NodeDef> ConvertAtrousConvBias(
    const NodeMatch &match, std::map<string, string> &inputs_to_rename,
    std::unordered_set<string> &nodes_to_ignore, const bool &quantize_ignored) {
  std::vector<NodeDef> new_nodes;
  const NodeDef &biasadd_node = match.node;
  const NodeDef &bias_node = match.inputs[1].node;
  const NodeDef &atrous_conv_node = match.inputs[0].inputs[0].node;
  const NodeDef &weights_node = match.inputs[0].inputs[0].inputs[1].node;
  const NodeDef &space_to_batch_block_shape_node =
      match.inputs[0].inputs[0].inputs[0].inputs[1].node;
  const NodeDef &space_to_batch_paddings_node =
      match.inputs[0].inputs[0].inputs[0].inputs[2].node;
  const NodeDef &batch_to_space_crops_node = match.inputs[0].inputs[2].node;
  const NodeDef &input_node =
      match.inputs[0].inputs[0].inputs[0].inputs[0].node;

  DLOG_INFO(1) << "Convert atrous_conv_bias: " << biasadd_node.name() << " <-- "
               << atrous_conv_node.name();

  NodeDef converted_node = atrous_conv_node;
  converted_node.mutable_input()->Clear();
  AddNodeInput(input_node.name(), &converted_node);
  CopyNodeAttr(weights_node, "value", "weights", &converted_node);
  CopyNodeAttr(bias_node, "value", "bias", &converted_node);

  // Set dilations
  Tensor block_shape =
      GetNodeTensorAttr(space_to_batch_block_shape_node, "value");
  const int32 block_height = block_shape.flat<int32>()(0);
  const int32 block_width = block_shape.flat<int32>()(1);
  std::vector<int> dilations;
  auto dilations_list = atrous_conv_node.attr().at("dilations").list();
  for (auto i = 0; i < dilations_list.i_size(); i++) {
    dilations.push_back(dilations_list.i(i));
  }
  dilations[1] = block_height;
  dilations[2] = block_width;
  SetNodeAttr("dilations", dilations, &converted_node);

  // Set paddings
  Tensor paddings = GetNodeTensorAttr(space_to_batch_paddings_node, "value");
  SetNodeTensorAttr<int>("space_to_batch_paddings", paddings, &converted_node);

  // Set crops
  Tensor crops = GetNodeTensorAttr(batch_to_space_crops_node, "value");
  SetNodeTensorAttr<int>("batch_to_space_crops", crops, &converted_node);

  if (quantize_ignored) {
    SetNodeAttr("quantize_ignored", true, &converted_node);
  } else {
    CHECK(weights_node.attr().count("wpos"))
        << "[DEPLOY ERROR] Cannnot find quantize info for weights: "
        << weights_node.name();
    CopyNodeAttr(weights_node, "wpos", "wpos", &converted_node);

    CHECK(bias_node.attr().count("wpos"))
        << "[DEPLOY ERROR] Cannnot find quantize info for bias: "
        << bias_node.name();
    CopyNodeAttr(bias_node, "wpos", "bpos", &converted_node);

    CHECK(biasadd_node.attr().count("opos"))
        << "[DEPLOY ERROR] Cannnot find quantize info for relu: "
        << biasadd_node.name();
    CopyNodeAttr(biasadd_node, "opos", "opos", &converted_node);
  }

  new_nodes.push_back(converted_node);
  return new_nodes;
}

std::vector<NodeDef> ConvertAtrousConvRelu(
    const NodeMatch &match, std::map<string, string> &inputs_to_rename,
    std::unordered_set<string> &nodes_to_ignore, const bool &quantize_ignored) {
  std::vector<NodeDef> new_nodes;
  const NodeDef &relu_node = match.node;
  const NodeDef &atrous_conv_node = match.inputs[0].inputs[0].node;
  const NodeDef &weights_node = match.inputs[0].inputs[0].inputs[1].node;
  const NodeDef &space_to_batch_block_shape_node =
      match.inputs[0].inputs[0].inputs[0].inputs[1].node;
  const NodeDef &space_to_batch_paddings_node =
      match.inputs[0].inputs[0].inputs[0].inputs[2].node;
  const NodeDef &batch_to_space_crops_node = match.inputs[0].inputs[2].node;
  const NodeDef &input_node =
      match.inputs[0].inputs[0].inputs[0].inputs[0].node;

  DLOG_INFO(1) << "Convert atrous_conv_relu: " << relu_node.name() << " <-- "
               << atrous_conv_node.name();

  NodeDef converted_node = atrous_conv_node;
  converted_node.mutable_input()->Clear();
  AddNodeInput(input_node.name(), &converted_node);
  CopyNodeAttr(weights_node, "value", "weights", &converted_node);

  // Set dilations
  Tensor block_shape =
      GetNodeTensorAttr(space_to_batch_block_shape_node, "value");
  const int32 block_height = block_shape.flat<int32>()(0);
  const int32 block_width = block_shape.flat<int32>()(1);
  std::vector<int> dilations;
  auto dilations_list = atrous_conv_node.attr().at("dilations").list();
  for (auto i = 0; i < dilations_list.i_size(); i++) {
    dilations.push_back(dilations_list.i(i));
  }
  dilations[1] = block_height;
  dilations[2] = block_width;
  SetNodeAttr("dilations", dilations, &converted_node);

  // Set paddings
  Tensor paddings = GetNodeTensorAttr(space_to_batch_paddings_node, "value");
  SetNodeTensorAttr<int>("space_to_batch_paddings", paddings, &converted_node);

  // Set crops
  Tensor crops = GetNodeTensorAttr(batch_to_space_crops_node, "value");
  SetNodeTensorAttr<int>("batch_to_space_crops", crops, &converted_node);

  NodeDef new_relu_node = relu_node;
  new_relu_node.set_input(0, converted_node.name());

  if (quantize_ignored) {
    SetNodeAttr("quantize_ignored", true, &converted_node);
    SetNodeAttr("quantize_ignored", true, &new_relu_node);
  } else {
    CHECK(weights_node.attr().count("wpos"))
        << "[DEPLOY ERROR] Cannnot find quantize info for weights: "
        << weights_node.name();
    CopyNodeAttr(weights_node, "wpos", "wpos", &converted_node);

    CHECK(relu_node.attr().count("opos"))
        << "[DEPLOY ERROR] Cannnot find quantize info for relu: "
        << relu_node.name();
    CopyNodeAttr(relu_node, "opos", "opos", &converted_node);
  }

  new_nodes.push_back(converted_node);
  new_nodes.push_back(new_relu_node);
  return new_nodes;
}

std::vector<NodeDef> ConvertAtrousConv(
    const NodeMatch &match, std::map<string, string> &inputs_to_rename,
    std::unordered_set<string> &nodes_to_ignore, const bool &quantize_ignored) {
  std::vector<NodeDef> new_nodes;
  const NodeDef &batch_to_space_node = match.node;
  const NodeDef &atrous_conv_node = match.inputs[0].node;
  const NodeDef &weights_node = match.inputs[0].inputs[1].node;
  const NodeDef &space_to_batch_block_shape_node =
      match.inputs[0].inputs[0].inputs[1].node;
  const NodeDef &space_to_batch_paddings_node =
      match.inputs[0].inputs[0].inputs[2].node;
  const NodeDef &batch_to_space_crops_node = match.inputs[2].node;
  const NodeDef &input_node = match.inputs[0].inputs[0].inputs[0].node;

  DLOG_INFO(1) << "Convert atrous_conv: " << atrous_conv_node.name();

  NodeDef converted_node = atrous_conv_node;
  converted_node.mutable_input()->Clear();
  AddNodeInput(input_node.name(), &converted_node);
  CopyNodeAttr(weights_node, "value", "weights", &converted_node);

  // Set dilations
  Tensor block_shape =
      GetNodeTensorAttr(space_to_batch_block_shape_node, "value");
  const int32 block_height = block_shape.flat<int32>()(0);
  const int32 block_width = block_shape.flat<int32>()(1);
  std::vector<int> dilations;
  auto dilations_list = atrous_conv_node.attr().at("dilations").list();
  for (auto i = 0; i < dilations_list.i_size(); i++) {
    dilations.push_back(dilations_list.i(i));
  }
  dilations[1] = block_height;
  dilations[2] = block_width;
  SetNodeAttr("dilations", dilations, &converted_node);

  // Set paddings
  Tensor paddings = GetNodeTensorAttr(space_to_batch_paddings_node, "value");
  SetNodeTensorAttr<int>("space_to_batch_paddings", paddings, &converted_node);

  // Set crops
  Tensor crops = GetNodeTensorAttr(batch_to_space_crops_node, "value");
  SetNodeTensorAttr<int>("batch_to_space_crops", crops, &converted_node);

  if (quantize_ignored) {
    SetNodeAttr("quantize_ignored", true, &converted_node);
  } else {
    CHECK(weights_node.attr().count("wpos"))
        << "[DEPLOY ERROR] Cannnot find quantize info for weights: "
        << weights_node.name();
    CopyNodeAttr(weights_node, "wpos", "wpos", &converted_node);

    CHECK(batch_to_space_node.attr().count("opos"))
        << "[DEPLOY ERROR] Cannnot find quantize info for batch_to_space: "
        << batch_to_space_node.name();
    CopyNodeAttr(batch_to_space_node, "opos", "opos", &converted_node);
  }

  inputs_to_rename[batch_to_space_node.name()] = converted_node.name();
  new_nodes.push_back(converted_node);
  return new_nodes;
}

std::vector<NodeDef> ConvertArrayRelu(
    const NodeMatch &match, std::map<string, string> &inputs_to_rename,
    std::unordered_set<string> &nodes_to_ignore, const bool &quantize_ignored) {
  std::vector<NodeDef> new_nodes;
  const NodeDef &relu_node = match.node;
  const NodeDef &array_node = match.inputs[0].node;

  NodeDef converted_node = array_node;
  if (array_node.op() == "AddV2") {
    converted_node.set_op("Add");
  }

  NodeDef new_relu_node = relu_node;
  if (quantize_ignored) {
    SetNodeAttr("quantize_ignored", true, &converted_node);
    SetNodeAttr("quantize_ignored", true, &new_relu_node);
  } else {
    CHECK(relu_node.attr().count("opos"))
        << "[DEPLOY ERROR] Cannot find quantize info for relu: "
        << relu_node.name();
    CopyNodeAttr(relu_node, "opos", "opos", &converted_node);
  }

  new_nodes.push_back(converted_node);
  new_nodes.push_back(new_relu_node);
  return new_nodes;
}

std::vector<NodeDef> ConvertArray(const NodeMatch &match,
                                  std::map<string, string> &inputs_to_rename,
                                  std::unordered_set<string> &nodes_to_ignore,
                                  const bool &quantize_ignored) {
  std::vector<NodeDef> new_nodes;
  const NodeDef &array_node = match.node;

  NodeDef converted_node = array_node;
  if (array_node.op() == "AddV2") {
    converted_node.set_op("Add");
  }

  if (quantize_ignored) {
    SetNodeAttr("quantize_ignored", true, &converted_node);
  }
  // Don't need to copy opos, because it's already there

  new_nodes.push_back(converted_node);
  return new_nodes;
}

std::vector<NodeDef> ConvertAvgpoolMul(
    const NodeMatch &match, std::map<string, string> &inputs_to_rename,
    std::unordered_set<string> &nodes_to_ignore, const bool &quantize_ignored) {
  std::vector<NodeDef> new_nodes;
  const NodeDef &mul_node = match.node;
  const NodeDef &avgpool_node = match.inputs[0].node;

  NodeDef converted_node = avgpool_node;
  if (quantize_ignored) {
    SetNodeAttr("quantize_ignored", true, &converted_node);
  } else {
    CHECK(mul_node.attr().count("opos"))
        << "[DEPLOY ERROR] Cannot find quantize info for relu: "
        << mul_node.name();
    CopyNodeAttr(mul_node, "opos", "opos", &converted_node);
  }

  new_nodes.push_back(converted_node);
  inputs_to_rename[mul_node.name()] = avgpool_node.name();
  return new_nodes;
}

std::vector<NodeDef> ConvertClipByValue(
    const NodeMatch &match, std::map<string, string> &inputs_to_rename,
    std::unordered_set<string> &nodes_to_ignore, const bool &quantize_ignored) {
  std::vector<NodeDef> new_nodes;
  const NodeDef &maximum_node = match.node;
  const NodeDef &minimum_node = match.inputs[0].node;
  const NodeDef &min_value_node = match.inputs[1].node;
  const NodeDef &max_value_node = match.inputs[0].inputs[1].node;
  const NodeDef &input_node = match.inputs[0].inputs[0].node;

  NodeDef converted_node = maximum_node;
  converted_node.set_op("ClipByValue");
  converted_node.mutable_input()->Clear();
  AddNodeInput(input_node.name(), &converted_node);

  if (quantize_ignored) {
    SetNodeAttr("quantize_ignored", true, &converted_node);
  }
  // Don't need to copy opos, because it's already there

  Tensor min_value = GetNodeTensorAttr(min_value_node, "value");
  Tensor max_value = GetNodeTensorAttr(max_value_node, "value");
  SetNodeAttr("min_clip_value", min_value, &converted_node);
  SetNodeAttr("max_clip_value", max_value, &converted_node);

  new_nodes.push_back(converted_node);
  return new_nodes;
}

std::vector<NodeDef> ConvertOtherRelu(
    const NodeMatch &match, std::map<string, string> &inputs_to_rename,
    std::unordered_set<string> &nodes_to_ignore, const bool &quantize_ignored) {
  std::vector<NodeDef> new_nodes;
  const NodeDef &relu_node = match.node;
  const NodeDef &other_node = match.inputs[0].node;

  NodeDef converted_node = other_node;
  if (other_node.op() == "AddV2") {
    converted_node.set_op("Add");
  }

  NodeDef new_relu_node = relu_node;
  if (quantize_ignored) {
    SetNodeAttr("quantize_ignored", true, &converted_node);
    SetNodeAttr("quantize_ignored", true, &new_relu_node);
  } else {
    CHECK(relu_node.attr().count("opos"))
        << "[DEPLOY ERROR] Cannot find quantize info for relu: "
        << relu_node.name();
    CopyNodeAttr(relu_node, "opos", "opos", &converted_node);
  }

  new_nodes.push_back(converted_node);
  new_nodes.push_back(new_relu_node);
  return new_nodes;
}

std::vector<NodeDef> ConvertOther(const NodeMatch &match,
                                  std::map<string, string> &inputs_to_rename,
                                  std::unordered_set<string> &nodes_to_ignore,
                                  const bool &quantize_ignored) {
  std::vector<NodeDef> new_nodes;
  const NodeDef &other_node = match.node;

  NodeDef converted_node = other_node;
  if (other_node.op() == "AddV2") {
    converted_node.set_op("Add");
  }

  if (quantize_ignored) {
    SetNodeAttr("quantize_ignored", true, &converted_node);
  }
  // Don't need to copy opos, because it's already there

  new_nodes.push_back(converted_node);
  return new_nodes;
}

} // namespace

Status FoldFixNeuron(const GraphDef &input_graph_def,
                     GraphDef *output_graph_def, bool fold_only) {
  GraphDef current_graph_def, replaced_graph_def;
  std::map<string, string> inputs_to_rename;

  current_graph_def = input_graph_def;
  TF_RETURN_IF_ERROR(ReplaceMatchingOpTypes(
      current_graph_def,
      // clang-format off
      {"FixNeuron",
        {
          {"FixNeuron"},
        }
      },
      // clang-format on
      [&inputs_to_rename](const NodeMatch &match,
                          const std::set<string> &input_nodes,
                          const std::set<string> &output_nodes,
                          std::vector<NodeDef> *new_nodes) {
        const NodeDef &dup_fn_node = match.node;
        const NodeDef &fn_node = match.inputs[0].node;
        DLOG_INFO(1) << "Fold duplicate FixNeuron: " << dup_fn_node.name()
                     << "<--" << fn_node.name();
        int dup_quantize_pos = (int)dup_fn_node.attr().at("quantize_pos").i();
        int quantize_pos = (int)fn_node.attr().at("quantize_pos").i();

        if (dup_quantize_pos != quantize_pos) {
          DLOG_WARNING
              << "Found duplicate FixNeuron nodes with different quantize_pos: "
              << dup_fn_node.name() << "(" << dup_quantize_pos
              << ") != " << fn_node.name() << "(" << quantize_pos
              << "), use quantize pos: " << quantize_pos;
        }

        new_nodes->push_back(fn_node);
        inputs_to_rename[dup_fn_node.name()] = fn_node.name();
        return Status::OK();
      },
      {true}, &replaced_graph_def));
  if (fold_only) {
    TF_RETURN_IF_ERROR(RenameNodeInputs(replaced_graph_def, inputs_to_rename,
                                        std::unordered_set<string>(),
                                        output_graph_def));
    return Status::OK();
  }
  TF_RETURN_IF_ERROR(RenameNodeInputs(replaced_graph_def, inputs_to_rename,
                                      std::unordered_set<string>(),
                                      &current_graph_def));

  inputs_to_rename.clear();
  TF_RETURN_IF_ERROR(ReplaceMatchingOpTypes(
      current_graph_def,
      // clang-format off
      {"FixNeuron",
        {
          {"*"},
        }
      },
      // clang-format on
      [&inputs_to_rename](const NodeMatch &match,
                          const std::set<string> &input_nodes,
                          const std::set<string> &output_nodes,
                          std::vector<NodeDef> *new_nodes) {
        const NodeDef &fn_node = match.node;
        const NodeDef &old_node = match.inputs[0].node;
        DLOG_INFO(2) << "Try fold FixNeuron for: " << fn_node.name() << "<--"
                     << old_node.name();

        NodeDef folded_node;

        int bit_width = (int)fn_node.attr().at("bit_width").i();
        int quantize_pos = (int)fn_node.attr().at("quantize_pos").i();
        int mode = (int)fn_node.attr().at("mode").i();

        if (old_node.op() == "Const") {
          if (mode == QuantizeMode::WEIGHT || mode == QuantizeMode::DW_WEIGHT) {
            SetNodeAttr("wpos", std::vector<int>{bit_width, quantize_pos},
                        &folded_node);
          } else if (mode == QuantizeMode::ACTIVATION) {
            SetNodeAttr("opos", std::vector<int>{bit_width, quantize_pos},
                        &folded_node);
          } else {
            LOG(FATAL) << "Invalid mode(" << mode
                       << ") for node:" << old_node.name();
          }
          // Quantize Const
          const DataType old_dtype = old_node.attr().at("dtype").type();
          Tensor old_tensor;
          if (!old_tensor.FromProto(old_node.attr().at("value").tensor())) {
            LOG(FATAL) << "Decoding Tensor failed for node: "
                       << old_node.name();
          }
          if (old_dtype != DT_FLOAT) {
            new_nodes->push_back(old_node);
            return Status::OK();
          }
          auto flat_old = old_tensor.flat<float>();
          const int data_size = flat_old.size();

          Tensor quantized_tensor(DT_FLOAT, old_tensor.shape());
          auto flat_quantized = quantized_tensor.flat<float>();

          quantize_cpu(data_size, flat_old.data(), flat_quantized.data(),
                       bit_width, quantize_pos);
          folded_node.set_op("Const");
          folded_node.set_name(old_node.name());
          SetNodeAttr("dtype", DT_FLOAT, &folded_node);
          SetNodeTensorAttr<float>("value", quantized_tensor, &folded_node);
        } else {
          folded_node.CopyFrom(old_node);
          SetNodeAttr("opos", std::vector<int>{bit_width, quantize_pos},
                      &folded_node);
        }

        new_nodes->push_back(folded_node);
        inputs_to_rename[fn_node.name()] = old_node.name();
        return Status::OK();
      },
      {true}, &replaced_graph_def));
  TF_RETURN_IF_ERROR(RenameNodeInputs(replaced_graph_def, inputs_to_rename,
                                      std::unordered_set<string>(),
                                      output_graph_def));
  return Status::OK();
}

Status PolishActivationInfo(const GraphDef &input_graph_def,
                            GraphDef *output_graph_def) {
  GraphDef processed_graph_def;
  std::unordered_map<string, std::vector<int>> activation_info_map;
  std::unordered_set<string> unquantized_nodes;
  for (auto i = 0; i < input_graph_def.node_size(); ++i) {
    NodeDef current_node = input_graph_def.node(i);

    std::vector<int> current_input_pos;
    for (auto j = 0; j < current_node.input_size(); ++j) {
      string current_input = NodeNameFromInput(current_node.input(j));
      if (activation_info_map.count(current_input)) {
        int bit_width = activation_info_map.at(current_input)[0];
        int pos = activation_info_map.at(current_input)[1];
        current_input_pos.push_back(bit_width);
        current_input_pos.push_back(pos);
      } else {
        if (!unquantized_nodes.count(current_node.name())) {
          DLOG_WARNING << "Node " << current_node.name()
                       << "(Type: " << current_node.op()
                       << ") is not quantized and cannot be deployed to DPU,"
                       << "because it has unquantized input node: "
                       << current_input << ". Please deploy it on CPU.";
          unquantized_nodes.insert(current_node.name());
          continue;
        }
      }
    }
    if (current_input_pos.size()) {
      DLOG_INFO(2) << "Set ipos " << current_input_pos[1]
                   << " for node: " << current_node.name();
      SetNodeAttr("ipos", current_input_pos, &current_node);
    }

    if (current_node.attr().count("opos")) {
      if (!activation_info_map.count(current_node.name())) {
        DLOG_INFO(2) << "Collect opos for: " << current_node.name();
        GetNodeAttr(current_node, "opos",
                    &activation_info_map[current_node.name()]);
      } else {
        LOG(FATAL) << "Duplicated opos: " << current_node.name();
      }
    }

    *(processed_graph_def.mutable_node()->Add()) = current_node;
  }
  *output_graph_def = processed_graph_def;
  return Status::OK();
}

Status FoldOpParams(const GraphDef &input_graph_def,
                    GraphDef *output_graph_def) {
  std::set<string> optype_with_param(
      {"Mean", "Pad", "Concat", "ConcatV2", "ArgMax", "Reshape", "ExpandDims"});
  std::map<string, NodeDef> target_ops;
  std::set<string> param_ops;
  std::map<string, const NodeDef *> nodes_map;
  MapNamesToNodes(input_graph_def, &nodes_map);
  for (auto i = 0; i < input_graph_def.node_size(); ++i) {
    const NodeDef current_node = input_graph_def.node(i);
    if (optype_with_param.count(current_node.op())) {
      DLOG_INFO(1) << "FoldOpParams, target_op: " << current_node.name();

      int param_id = current_node.input_size() - 1;
      string param_name = NodeNameFromInput(current_node.input(param_id));
      param_ops.insert(param_name);
      DLOG_INFO(1) << "FoldOpParams, param_op: " << param_name;
      const NodeDef *param_node = nodes_map[param_name];
      CHECK(param_node->op() == "Const")
          << "[DEPLOY ERROR] Expect const node for node" << current_node.name()
          << "(" << current_node.op() << ")'s param, but " << param_node->op()
          << "(" << param_node->name() << ") found.";

      // Build folded node
      NodeDef folded_node = current_node;
      if (current_node.op() == "Mean") {
        CopyNodeAttr(*param_node, "value", "reduction_indices", &folded_node);
      } else if (current_node.op() == "Pad") {
        CopyNodeAttr(*param_node, "value", "paddings", &folded_node);
      } else if (current_node.op() == "Concat" ||
                 current_node.op() == "ConcatV2") {
        CopyNodeAttr(*param_node, "value", "axis", &folded_node);
      } else if (current_node.op() == "ArgMax") {
        CopyNodeAttr(*param_node, "value", "dimension", &folded_node);
      } else if (current_node.op() == "Reshape") {
        CopyNodeAttr(*param_node, "value", "shape", &folded_node);
      } else if (current_node.op() == "ExpandDims") {
        CopyNodeAttr(*param_node, "value", "dim", &folded_node);
      }

      folded_node.mutable_input()->Clear();
      for (auto j = 0; j < current_node.input_size() - 1; ++j) {
        AddNodeInput(current_node.input(j), &folded_node);
      }
      target_ops[current_node.name()] = folded_node;
    }
  }

  GraphDef folded_graph_def;
  for (auto i = 0; i < input_graph_def.node_size(); ++i) {
    NodeDef current_node = input_graph_def.node(i);
    if (target_ops.count(current_node.name())) {
      *(folded_graph_def.mutable_node()->Add()) =
          target_ops[current_node.name()];
    } else if (param_ops.count(current_node.name())) {
      continue;
    } else {
      *(folded_graph_def.mutable_node()->Add()) = current_node;
    }
  }

  *output_graph_def = folded_graph_def;
  return Status::OK();
}

Status ConvertPatterns(const GraphDef &input_graph_def,
                       GraphDef *output_graph_def, QuantizeConfig &config) {
  std::vector<std::tuple<int, NodeMatch>> matched_node_patterns;
  std::unordered_map<string, int> matched_nodes;
  std::unordered_set<string> unmatched_nodes;
  TF_RETURN_IF_ERROR(ParseGraph(input_graph_def, matched_node_patterns,
                                matched_nodes, config.ignore_nodes,
                                unmatched_nodes));

  typedef std::vector<NodeDef> (*ConvertFuncHandle)(
      const NodeMatch &, std::map<string, string> &,
      std::unordered_set<string> &, const bool &);

  std::map<string, ConvertFuncHandle> map_pattern_func({
      {"placeholder", &ConvertPlaceholder},
      {"atrous_conv_bias_relu", &ConvertAtrousConvBiasRelu},
      {"atrous_conv_bias", &ConvertAtrousConvBias},
      {"atrous_conv_relu", &ConvertAtrousConvRelu},
      {"atrous_conv", &ConvertAtrousConv},
      {"convfc_bias_id_relu", &ConvertConvfcBiasIdRelu},
      {"convfc_bias_relu", &ConvertConvfcBiasRelu},
      {"convfc_bias", &ConvertConvfcBias},
      {"convfc_relu", &ConvertConvfcRelu},
      {"convfc", &ConvertConvfc},
      {"conv2d_transpose_bias_relu", &ConvertConv2dTransposeBiasRelu},
      {"conv2d_transpose_bias", &ConvertConv2dTransposeBias},
      {"conv2d_transpose_relu", &ConvertConv2dTransposeRelu},
      {"conv2d_transpose", &ConvertConv2dTranspose},
      {"keras_conv2d_transpose_bias_relu",
       &ConvertConv2dTransposeBiasRelu}, // reuse function of Conv2dTranspose
                                         // for KerasConv2dTranspose
      {"keras_conv2d_transpose_bias", &ConvertConv2dTransposeBias},
      {"keras_conv2d_transpose_relu", &ConvertConv2dTransposeRelu},
      {"keras_conv2d_transpose", &ConvertConv2dTranspose},
      {"conv2d_backprop_input_bias_relu",
       &ConvertConv2dTransposeBiasRelu}, // reuse function of Conv2dTranspose
                                         // for Conv2d_backprop_input
      {"conv2d_backprop_input_bias", &ConvertConv2dTransposeBias},
      {"conv2d_backprop_input_relu", &ConvertConv2dTransposeRelu},
      {"conv2d_backprop_input", &ConvertConv2dTranspose},
      {"convfc_bias_fused_leakyrelu", &ConvertConvfcBiasFusedLeakyRelu},
      {"convfc_bias_keras_leakyrelu", &ConvertConvfcBiasKerasLeakyRelu},
      {"convfc_bias_leakyrelu", &ConvertConvfcBiasLeakyRelu},
      {"convfc_fused_leakyrelu", &ConvertConvfcFusedLeakyRelu},
      {"convfc_keras_leakyrelu", &ConvertConvfcKerasLeakyRelu},
      {"convfc_leakyrelu", &ConvertConvfcLeakyRelu},
      {"leakyrelu", &ConvertLeakyRelu},
      {"fused_leakyrelu", &ConvertFusedLeakyRelu},
      {"keras_leakyrelu", &ConvertKerasLeakyRelu},
      {"upsampling", &ConvertUpsampling},
      {"resize_bilinear", &ConvertResize},
      {"tpu_nearest_neighbor_upsampling", &ConvertTPUNearestNeighborUpsampling},
      {"batchnorm_relu", &ConvertBatchNormRelu},
      {"batchnorm", &ConvertBatchNorm},
      {"array_relu", &ConvertArrayRelu},
      {"array", &ConvertArray},
      {"avgpool_mul", &ConvertAvgpoolMul},
      {"clip_by_value", &ConvertClipByValue},
      {"other_relu", &ConvertOtherRelu},
      {"other", &ConvertOther},
  });

  GraphDef converted_graph_def;
  std::map<string, string> inputs_to_rename;
  std::unordered_set<string> nodes_to_ignore;
  std::unordered_map<string, int> processed_nodes;
  for (auto current_node : input_graph_def.node()) {
    if (processed_nodes.count(current_node.name())) {
      continue;
    }

    if (matched_nodes.count(current_node.name())) {
      int match_id = matched_nodes[current_node.name()];
      auto match = std::get<1>(matched_node_patterns[match_id]);
      auto pattern_id = std::get<0>(matched_node_patterns[match_id]);
      auto pattern_name = get_pattern_name_from_id(pattern_id);

      bool quantize_ignored = false;
      std::vector<const NodeDef *> input_nodes =
          get_input_nodes(match, pattern_name);
      std::set<string> input_node_names;
      for (int i = 0; i < input_nodes.size(); i++) {
        input_node_names.insert(input_nodes[i]->name());
      }
      if (CheckAnyIgnoredNodes(config.ignore_nodes, match, input_node_names)) {
        DLOG_WARNING << "Found ignored match: ";
        PrintNodeMatch(match, 2);
        quantize_ignored = true;
      }

      DLOG_INFO(1) << "Perform convert func, match_id: " << match_id
                   << ", pattern: " << pattern_name
                   << ", match_node: " << match.node.name();
      std::vector<NodeDef> new_nodes;
      new_nodes = (*map_pattern_func[pattern_name])(
          match, inputs_to_rename, nodes_to_ignore, quantize_ignored);
      for (auto node : new_nodes) {
        DLOG_INFO(1) << "Add quantized node: " << node.name() << "("
                     << node.op() << ")";
        NodeDef *new_node = converted_graph_def.mutable_node()->Add();
        *new_node = node;
      }
      RecordMatchedNodes(processed_nodes, match, match_id, input_node_names);
    } else {
      DLOG_INFO(1) << "Add unquantized node: " << current_node.name() << "("
                   << current_node.op() << ")";
      NodeDef *new_node = converted_graph_def.mutable_node()->Add();
      *new_node = current_node;
    }
  }
  TF_RETURN_IF_ERROR(RenameNodeInputs(converted_graph_def, inputs_to_rename,
                                      nodes_to_ignore, output_graph_def));
  return Status::OK();
}

Status DeployQuantizedGraph(const GraphDef &input_graph_def,
                            GraphDef *output_graph_def, QuantizeConfig config) {
  GraphDef current_graph_def, processed_graph_def;

  // Sort
  current_graph_def = input_graph_def;
  SortByExecutionOrder(current_graph_def, &processed_graph_def);

  // Convert patterns
  current_graph_def = processed_graph_def;
  ConvertPatterns(current_graph_def, &processed_graph_def, config);
  SaveGraphForDebugging(processed_graph_def, "convert_patterns.pb",
                        config.output_dir);

  // Fold op params
  current_graph_def = processed_graph_def;
  TF_RETURN_IF_ERROR(FoldOpParams(current_graph_def, &processed_graph_def));
  SaveGraphForDebugging(processed_graph_def, "fold_op_params.pb",
                        config.output_dir);

  // Sort
  current_graph_def = processed_graph_def;
  SortByExecutionOrder(current_graph_def, &processed_graph_def);

  // Polish activation info
  current_graph_def = processed_graph_def;
  PolishActivationInfo(current_graph_def, output_graph_def);

  return Status::OK();
}

Status DeployQuantizedGraphCommand(const GraphDef &input_graph_def,
                                   const TransformFuncContext &context,
                                   GraphDef *output_graph_def) {
  return DeployQuantizedGraph(input_graph_def, output_graph_def);
}

REGISTER_DECENT_Q_GRAPH_TRANSFORM("deploy_quantized_graph",
                                  DeployQuantizedGraphCommand);

} // namespace decent_q
} // namespace tensorflow
