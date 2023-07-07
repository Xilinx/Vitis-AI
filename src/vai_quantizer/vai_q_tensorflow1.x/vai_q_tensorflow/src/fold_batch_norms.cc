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

#include "fold_batch_norms.h"
#include "quantize_utils.h"
#include "tensorflow/core/lib/strings/str_util.h"

namespace tensorflow {
namespace decent_q {

// Ensures the tensor is the expected shape.
Status ErrorIfNotVector(const Tensor &input, const string &input_name,
                        int expected_width) {
  if ((input.shape().dims() != 1) ||
      (input.shape().dim_size(0) != expected_width)) {
    DLOG_INFO(1) << "The input dim: " << input.shape().dims()
                 << " dim size: " << input.shape().dim_size(0)
                 << " expected width: " << expected_width;
    return errors::InvalidArgument(
        input_name,
        " input to batch norm has bad shape: ", input.shape().DebugString());
  }
  return Status::OK();
}

Status GetScaleAndOffsetNodes(const NodeMatch &match,
                              std::map<string, string> &inputs_to_rename,
                              std::unordered_set<string> &nodes_to_ignore,
                              std::vector<NodeDef> *new_nodes) {
  // Find all the nodes we expect in the subgraph.
  const NodeDef &batch_norm_node = match.node;
  const NodeDef &input_node = match.inputs[0].node;
  // BatchNormWithGlobalNormalization and FusedBatchNorm ops only differ
  // by input order and attribute names.
  CHECK(batch_norm_node.op() == "BatchNormWithGlobalNormalization" ||
        batch_norm_node.op() == "FusedBatchNorm" ||
        batch_norm_node.op() == "FusedBatchNormV2" ||
        batch_norm_node.op() == "FusedBatchNormV3");
  const bool is_fused = batch_norm_node.op() == "FusedBatchNorm" ||
                        batch_norm_node.op() == "FusedBatchNormV2" ||
                        batch_norm_node.op() == "FusedBatchNormV3";
  const int mean_idx = is_fused ? 3 : 1;
  const int var_idx = is_fused ? 4 : 2;
  const int beta_idx = is_fused ? 2 : 3;
  const int gamma_idx = is_fused ? 1 : 4;
  const string epsilon_attr = is_fused ? "epsilon" : "variance_epsilon";
  // FusedBatchNorm always scales after normalization.
  const bool scale_after_normalization =
      is_fused || batch_norm_node.attr().at("scale_after_normalization").b();

  const NodeDef &mean_node = match.inputs[mean_idx].node;
  CHECK_EQ("Const", mean_node.op());
  const NodeDef &variance_node = match.inputs[var_idx].node;
  CHECK_EQ("Const", variance_node.op());
  const NodeDef &beta_node = match.inputs[beta_idx].node;
  CHECK_EQ("Const", beta_node.op());
  const NodeDef &gamma_node = match.inputs[gamma_idx].node;
  CHECK_EQ("Const", gamma_node.op());

  // We have a set of vectors that we want to combine into a vector of
  // scale values and offset values.
  Tensor mean = GetNodeTensorAttr(mean_node, "value");
  Tensor variance = GetNodeTensorAttr(variance_node, "value");
  Tensor beta = GetNodeTensorAttr(beta_node, "value");
  Tensor gamma = GetNodeTensorAttr(gamma_node, "value");
  const float variance_epsilon = batch_norm_node.attr().at(epsilon_attr).f();

  // Make sure all the inputs really are vectors with the same shape.
  const int64 num_cols = mean.shape().num_elements();
  TF_RETURN_IF_ERROR(ErrorIfNotVector(variance, "Variance", num_cols));
  TF_RETURN_IF_ERROR(ErrorIfNotVector(beta, "Beta", num_cols));
  TF_RETURN_IF_ERROR(ErrorIfNotVector(gamma, "gamma", num_cols));

  // Calulate the scale and offset
  TensorShape new_shape;
  if (batch_norm_node.attr().at("data_format").s() == "NHWC") {
    new_shape = mean.shape();
  } else if (batch_norm_node.attr().at("data_format").s() == "NCHW") {
    std::vector<int64> nums = {1l, mean.shape().num_elements(), 1l, 1l};
    TensorShapeUtils::MakeShape(nums.data(), (int64)nums.size(), &new_shape);
  } else {
    return errors::InvalidArgument(
        "Bad shape of the batch-normal's data_format: ",
        batch_norm_node.attr().at("data_format").s());
  }

  Tensor scale(DT_FLOAT, new_shape);
  auto scale_flatten = scale.flat<float>();
  Tensor offset(DT_FLOAT, new_shape);
  auto offset_flatten = offset.flat<float>();

  // Calculate the scale and offset values to apply.
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
        (-mean.flat<float>()(i) * scale_flatten(i)) + beta.flat<float>()(i);
  }

  // Construct the new nodes.
  new_nodes->push_back(input_node);

  NodeDef scale_node;
  scale_node.set_op("Const");
  scale_node.set_name(batch_norm_node.name() + "/scale");
  SetNodeAttr("dtype", DT_FLOAT, &scale_node);
  SetNodeTensorAttr<float>("value", scale, &scale_node);
  new_nodes->push_back(scale_node);

  NodeDef mul_node;
  mul_node.set_op("Mul");
  mul_node.set_name(batch_norm_node.name() + "/mul");
  SetNodeAttr("T", DT_FLOAT, &mul_node);
  AddNodeInput(input_node.name(), &mul_node);
  AddNodeInput(scale_node.name(), &mul_node);
  new_nodes->push_back(mul_node);

  NodeDef offset_node;
  offset_node.set_op("Const");
  offset_node.set_name(batch_norm_node.name() + "/offset");
  SetNodeAttr("dtype", DT_FLOAT, &offset_node);
  SetNodeTensorAttr<float>("value", offset, &offset_node);
  new_nodes->push_back(offset_node);

  NodeDef add_node;
  add_node.set_op("Add");
  add_node.set_name(batch_norm_node.name());
  SetNodeAttr("T", DT_FLOAT, &add_node);
  AddNodeInput(mul_node.name(), &add_node);
  AddNodeInput(offset_node.name(), &add_node);
  new_nodes->push_back(add_node);

  return Status::OK();
}

Status UpdateOldBatchNorms(const GraphDef &input_graph_def,
                           GraphDef *output_graph_def) {
  GraphDef replaced_graph_def;
  std::map<string, string> inputs_to_rename;
  std::unordered_set<string> nodes_to_ignore;
  TF_RETURN_IF_ERROR(ReplaceMatchingOpTypes(
      input_graph_def, // clang-format off
    {"BatchNormWithGlobalNormalization|FusedBatchNorm|FusedBatchNormV2|FusedBatchNormV3",    // batch_norm_node
      {
        {"*"},  // input_node
        {"Const"},  // mean_node
        {"Const"},  // variance_node
        {"Const"},  // beta_node
        {"Const"},  // gamma_node
      }
    }, // clang-format on
      [&inputs_to_rename, &nodes_to_ignore](
          const NodeMatch &match, const std::set<string> &input_nodes,
          const std::set<string> &output_nodes,
          std::vector<NodeDef> *new_nodes) {
        TF_RETURN_IF_ERROR(GetScaleAndOffsetNodes(match, inputs_to_rename,
                                                  nodes_to_ignore, new_nodes));
        return Status::OK();
      },
      {true}, &replaced_graph_def));
  TF_RETURN_IF_ERROR(RenameNodeInputs(replaced_graph_def, inputs_to_rename,
                                      nodes_to_ignore, output_graph_def));
  return Status::OK();
}

Status GetMergedConvWeights(const NodeDef &conv_node,
                            const NodeDef &weights_node,
                            const NodeDef &mul_values_node,
                            NodeDef *scaled_weights_node) {
  Tensor weights = GetNodeTensorAttr(weights_node, "value");
  Tensor mul_values = GetNodeTensorAttr(mul_values_node, "value");

  // Make sure all the inputs really are vectors, with as many entries
  // as there are columns in the weights.
  int weights_cols_index = 3;
  if (conv_node.op() == "Conv2D") {
    weights_cols_index = 3;
  } else if (conv_node.op() == "DepthwiseConv2dNative" ||
             conv_node.op() == "Conv2DBackpropInput") {
    weights_cols_index = 2;
  } else {
    weights_cols_index = 1;
  }
  const int64 weights_cols = weights.shape().dim_size(weights_cols_index);

  if (conv_node.op() != "MatMul") {
    std::string data_format = conv_node.attr().at("data_format").s();
    if (data_format == "NHWC") {
      if ((mul_values.shape().dims() != 1) ||
          (mul_values.shape().dim_size(0) != weights_cols)) {
        return errors::InvalidArgument(
            "Mul constant NHWC input to batch norm has bad shape: ",
            mul_values.shape().DebugString());
      }
    } else if (data_format == "NCHW") {
      if ((mul_values.shape().dims() == 1) ||
          (mul_values.shape().dim_size(1) != weights_cols)) {
        return errors::InvalidArgument(
            "Mul constant NCHW input to batch norm has bad shape: ",
            mul_values.shape().DebugString());
      }
    } else {
      return errors::InvalidArgument(
          "Mul constant input to batch norm has bad data format: ",
          data_format);
    }
  }

  // Multiply the original weights by the scale vector.
  Tensor scaled_weights(DT_FLOAT, weights.shape());
  if (conv_node.op() == "DepthwiseConv2dNative") {
    auto weights_flatten = weights.flat<float>();
    auto scaled_weights_flatten = scaled_weights.flat<float>();
    for (int64 i = 0; i < weights.NumElements(); ++i) {
      scaled_weights_flatten(i) =
          weights_flatten(i) * mul_values.flat<float>()(i % weights_cols);
    }
  } else if (conv_node.op() == "Conv2DBackpropInput") {
    auto weights_tensor = weights.tensor<float, 4>();
    auto scaled_weights_tensor = scaled_weights.tensor<float, 4>();
    const int64 weights_h = weights.shape().dim_size(0);
    const int64 weights_w = weights.shape().dim_size(1);
    const int64 weights_in = weights.shape().dim_size(3);
    // Conv2DBackpropInput weights index is [H, W, Out, In]
    // scaled_weights = weights[:,:,col,:] * mul_values[col]
    for (int64 col = 0; col < weights_cols; ++col) {
      for (int64 in_num = 0; in_num < weights_in; ++in_num) {
        for (int64 h = 0; h < weights_h; ++h) {
          for (int64 w = 0; w < weights_w; ++w) {
            scaled_weights_tensor(h, w, col, in_num) =
                weights_tensor(h, w, col, in_num) *
                mul_values.flat<float>()(col);
          }
        }
      }
    }
  } else {
    auto weights_matrix = weights.flat_inner_dims<float>();
    auto scaled_weights_matrix = scaled_weights.flat_inner_dims<float>();
    for (int64 row = 0; row < weights_matrix.dimension(0); ++row) {
      for (int64 col = 0; col < weights_cols; ++col) {
        scaled_weights_matrix(row, col) =
            weights_matrix(row, col) * mul_values.flat<float>()(col);
      }
    }
  }

  // Construct the new nodes.
  scaled_weights_node->set_op("Const");
  scaled_weights_node->set_name(weights_node.name());
  SetNodeAttr("dtype", DT_FLOAT, scaled_weights_node);
  SetNodeTensorAttr<float>("value", scaled_weights, scaled_weights_node);
  return Status::OK();
}

Status GetMergedConvBiases(const NodeDef &bias_node,
                           const NodeDef &mul_values_node,
                           const NodeDef &add_values_node,
                           NodeDef *scaled_bias_node) {
  Tensor bias = GetNodeTensorAttr(bias_node, "value");
  Tensor mul_values = GetNodeTensorAttr(mul_values_node, "value");
  Tensor add_values = GetNodeTensorAttr(add_values_node, "value");

  // Multiply the original biases by the scale vector and add the
  // add_values.
  Tensor scaled_bias(DT_FLOAT, bias.shape());
  auto bias_vector = bias.flat<float>();
  auto scaled_bias_vector = scaled_bias.flat<float>();
  for (int64 col = 0; col < bias.NumElements(); ++col) {
    scaled_bias_vector(col) = bias_vector(col) * mul_values.flat<float>()(col) +
                              add_values.flat<float>()(col);
  }

  scaled_bias_node->set_op("Const");
  scaled_bias_node->set_name(bias_node.name());
  SetNodeAttr("dtype", DT_FLOAT, scaled_bias_node);
  SetNodeTensorAttr<float>("value", scaled_bias, scaled_bias_node);
  return Status::OK();
}

Status FoldBatchNormsTraining(const GraphDef &input_graph_def,
                              GraphDef *output_graph_def) {
  GraphDef current_graph_def = input_graph_def;

  // Find moving_mean and moving_variance for all FusedBatchNorm
  std::map<string, string> bn_to_mm, bn_to_mv;
  std::vector<NodeMatch> matches;
  // conv + batchnorm
  // clang-format off
  OpTypePattern bn_moving_mean_var_pattern(
      {"Sub",
        {
          {"Identity|ReadVariableOp",
            {
              {"VariableV2|VarHandleOp"},
            }
          },
          {"FusedBatchNorm|FusedBatchNormV2|FusedBatchNormV3",
            {
              {"Conv2D|MatMul|DepthwiseConv2dNative|Identity"},  // conv_node
              {"*"},  // beta_node
              {"*"},  // gamma_node
              {"*"},  // mean_node
              {"*"},  // variance_node
            }
          },
        }
      });
  // clang-format on
  GraphMatcher matcher(current_graph_def);
  std::vector<NodeMatch> tmp_matches;
  bool allow_intersection = true;
  TF_RETURN_IF_ERROR(matcher.GetOpTypeMatches(
      bn_moving_mean_var_pattern, &tmp_matches, allow_intersection));

  for (const NodeMatch &match : tmp_matches) {
    matches.push_back(match);
  }

  tmp_matches.clear();
  // clang-format off
  OpTypePattern bias_bn_moving_mean_var_pattern(
      {"Sub",
        {
          {"Identity|ReadVariableOp",
            {
              {"VariableV2|VarHandleOp"},
            }
          },
          {"FusedBatchNorm|FusedBatchNormV2|FusedBatchNormV3",
            {
              {"BiasAdd|Add|AddV2",
                {
                  // {"Conv2D|MatMul|DepthwiseConv2dNative|Identity"},  // conv_node
                  {"Conv2D|MatMul|DepthwiseConv2dNative"},  // conv_node
                  {"*"}, // bias
                }
              },
              {"*"},  // beta_node
              {"*"},  // gamma_node
              {"*"},  // mean_node
              {"*"},  // variance_node
            }
          },
        }
      });
  // clang-format on
  TF_RETURN_IF_ERROR(matcher.GetOpTypeMatches(
      bias_bn_moving_mean_var_pattern, &tmp_matches, allow_intersection));

  for (const NodeMatch &match : tmp_matches) {
    matches.push_back(match);
  }

  for (const NodeMatch &match : matches) {
    const NodeDef &sub_node = match.node;
    const NodeDef &bn_node = match.inputs[1].node;
    const NodeDef &mean_var_node = match.inputs[0].node;
    if (sub_node.input(1) == bn_node.name() + ":1") {
      bn_to_mm[bn_node.name()] = mean_var_node.name();
    } else if (sub_node.input(1) == bn_node.name() + ":2") {
      bn_to_mv[bn_node.name()] = mean_var_node.name();
    } else {
      DLOG_WARNING << "Found unsupported fused node pattern"
                   << match.node.name() << " <-- "
                   << match.inputs[0].node.name() << " + "
                   << match.inputs[1].node.name();
    }
  }

  CHECK_EQ(bn_to_mm.size(), bn_to_mv.size());
  DLOG_INFO(1) << "Found conv [bias] batchnorm pattern number: "
               << bn_to_mm.size();
  for (auto i = 0; i < current_graph_def.node_size(); i++) {
    const NodeDef &cur_node = current_graph_def.node(i);
    if (cur_node.op() == "FusedBatchNorm" ||
        cur_node.op() == "FusedBatchNormV2" ||
        cur_node.op() == "FusedBatchNormV3") {
      if (cur_node.attr().at("is_training").b()) {
        if (!bn_to_mm.count(cur_node.name()) ||
            !bn_to_mv.count(cur_node.name())) {
          DLOG_WARNING
              << "Fail to find moving_mean and moving_variance for node "
              << cur_node.name();
        }
      } else {
        bn_to_mm[cur_node.name()] = cur_node.input(3);
        bn_to_mv[cur_node.name()] = cur_node.input(4);
      }
    }
  }

  std::map<string, string> inputs_to_rename;
  std::unordered_set<string> nodes_to_ignore;

  {
    // Fold Conv2D + bias + BatchNorm
    inputs_to_rename.clear();
    nodes_to_ignore.clear();
    GraphDef replaced_graph_def;
    bool allow_intersection = true;
    TF_RETURN_IF_ERROR(ReplaceMatchingOpTypes(
        current_graph_def, // clang-format off
        {"FusedBatchNorm|FusedBatchNormV2|FusedBatchNormV3",                // batchnorm
          {
            {"BiasAdd|Add|AddV2",
              {
                {"Conv2D|MatMul|DepthwiseConv2dNative"},  // conv_node
                {"*"}, // bias
              }
            },
            {"*"},  // beta_node
            {"*"},  // gamma_node
            {"*"},  // mean_node
            {"*"},  // variance_node
          }
        }, // clang-format on
        [&inputs_to_rename, &nodes_to_ignore, &bn_to_mm,
         &bn_to_mv](const NodeMatch &match, const std::set<string> &input_nodes,
                    const std::set<string> &output_nodes,
                    std::vector<NodeDef> *new_nodes) {
          // Find all the nodes we expect in the subgraph.
          const NodeDef &bn_node = match.node;
          const NodeDef &biasadd_node = match.inputs[0].node;
          const NodeDef &bias_node = match.inputs[0].inputs[1].node;
          const NodeDef &conv_node = match.inputs[0].inputs[0].node;
          const string input_node_name = conv_node.input()[0];
          const string weights_node_name = conv_node.input()[1];
          const NodeDef &gamma_node = match.inputs[1].node;
          const NodeDef &beta_node = match.inputs[2].node;
          const NodeDef &mean_node = match.inputs[3].node;
          const NodeDef &var_node = match.inputs[4].node;

          // CopyOriginalMatch(match, new_nodes);
          new_nodes->push_back(conv_node);
          new_nodes->push_back(biasadd_node);
          new_nodes->push_back(bias_node);
          new_nodes->push_back(bn_node);
          new_nodes->push_back(beta_node);
          new_nodes->push_back(gamma_node);
          new_nodes->push_back(mean_node);
          new_nodes->push_back(var_node);

          StringPiece current_scope_piece = conv_node.name();
          str_util::ConsumeSuffix(&current_scope_piece, "/Conv2D");
          string current_scope = string(current_scope_piece);
          string bn_fold_scope = current_scope + "/BatchNorm_Fold";

          // Construct the new nodes.
          NodeDef eps_node;
          eps_node.set_name(bn_fold_scope + "/eps");
          eps_node.set_op("Const");
          SetNodeAttr("dtype", DT_FLOAT, &eps_node);
          Tensor eps_tensor(DT_FLOAT, {1});
          eps_tensor.flat<float>()(0) = bn_node.attr().at("epsilon").f();
          SetNodeTensorAttr<float>("value", eps_tensor, &eps_node);
          new_nodes->push_back(eps_node);

          NodeDef add_node;
          add_node.set_name(bn_fold_scope + "/add");
          add_node.set_op("Add");
          SetNodeAttr("T", DT_FLOAT, &add_node);
          AddNodeInput(eps_node.name(), &add_node);
          // Use moving variance
          AddNodeInput(bn_to_mv[bn_node.name()], &add_node);
          new_nodes->push_back(add_node);

          NodeDef rsqrt_node;
          rsqrt_node.set_name(bn_fold_scope + "/rsqrt");
          rsqrt_node.set_op("Rsqrt");
          SetNodeAttr("T", DT_FLOAT, &rsqrt_node);
          AddNodeInput(add_node.name(), &rsqrt_node);
          new_nodes->push_back(rsqrt_node);

          NodeDef mul_node_0;
          mul_node_0.set_name(bn_fold_scope + "/mul_0");
          mul_node_0.set_op("Mul");
          SetNodeAttr("T", DT_FLOAT, &mul_node_0);
          AddNodeInput(rsqrt_node.name(), &mul_node_0);
          AddNodeInput(gamma_node.name(), &mul_node_0);
          new_nodes->push_back(mul_node_0);

          // Reshape scalar for depthwise conv
          NodeDef reshape_node;
          if (conv_node.op() == "DepthwiseConv2dNative") {
            NodeDef shape_node;
            shape_node.set_name(bn_fold_scope + "/shape");
            shape_node.set_op("Const");
            SetNodeAttr("dtype", DT_INT32, &shape_node);
            Tensor shape_tensor(DT_INT32, {2});
            shape_tensor.flat<int>()(0) = -1;
            shape_tensor.flat<int>()(1) = 1;
            SetNodeTensorAttr<float>("value", shape_tensor, &shape_node);
            new_nodes->push_back(shape_node);

            reshape_node.set_name(bn_fold_scope + "/reshape");
            reshape_node.set_op("Reshape");
            SetNodeAttr("T", DT_FLOAT, &reshape_node);
            SetNodeAttr("Tshape", DT_INT32, &reshape_node);
            AddNodeInput(mul_node_0.name(), &reshape_node);
            AddNodeInput(shape_node.name(), &reshape_node);
            new_nodes->push_back(reshape_node);
          }

          NodeDef mul_node_1;
          mul_node_1.set_name(bn_fold_scope + "/mul_1");
          mul_node_1.set_op("Mul");
          SetNodeAttr("T", DT_FLOAT, &mul_node_1);
          if (conv_node.op() == "DepthwiseConv2dNative") {
            AddNodeInput(reshape_node.name(), &mul_node_1);
          } else {
            AddNodeInput(mul_node_0.name(), &mul_node_1);
          }

          AddNodeInput(weights_node_name, &mul_node_1);
          new_nodes->push_back(mul_node_1);

          NodeDef fold_conv_node = conv_node;
          fold_conv_node.set_name(conv_node.name() + "_Fold");
          fold_conv_node.mutable_input()->Clear();
          AddNodeInput(input_node_name, &fold_conv_node);
          AddNodeInput(mul_node_1.name(), &fold_conv_node);
          new_nodes->push_back(fold_conv_node);
          inputs_to_rename[conv_node.name()] = fold_conv_node.name();
          nodes_to_ignore.insert(biasadd_node.name());

          NodeDef mul_node_2;
          mul_node_2.set_name(bn_fold_scope + "/mul_2");
          mul_node_2.set_op("Mul");
          SetNodeAttr("T", DT_FLOAT, &mul_node_2);
          AddNodeInput(mul_node_0.name(), &mul_node_2);
          // Use moving_mean
          AddNodeInput(bn_to_mm[bn_node.name()], &mul_node_2);
          new_nodes->push_back(mul_node_2);

          NodeDef sub_node;
          sub_node.set_name(bn_fold_scope + "/sub");
          sub_node.set_op("Sub");
          SetNodeAttr("T", DT_FLOAT, &sub_node);
          AddNodeInput(beta_node.name(), &sub_node);
          AddNodeInput(mul_node_2.name(), &sub_node);
          new_nodes->push_back(sub_node);

          NodeDef mul_node_3;
          mul_node_3.set_name(bn_fold_scope + "/mul_3");
          mul_node_3.set_op("Mul");
          SetNodeAttr("T", DT_FLOAT, &mul_node_3);
          AddNodeInput(mul_node_0.name(), &mul_node_3);
          AddNodeInput(bias_node.name(), &mul_node_3);
          new_nodes->push_back(mul_node_3);

          NodeDef add_node_1;
          add_node_1.set_name(bn_fold_scope + "/add_1");
          add_node_1.set_op("AddV2");
          SetNodeAttr("T", DT_FLOAT, &add_node_1);
          AddNodeInput(sub_node.name(), &add_node_1);
          AddNodeInput(mul_node_3.name(), &add_node_1);
          new_nodes->push_back(add_node_1);

          NodeDef fold_biasadd_node;
          fold_biasadd_node.set_name(current_scope + "/biasadd_fold");
          fold_biasadd_node.set_op("BiasAdd");
          SetNodeAttr("T", DT_FLOAT, &fold_biasadd_node);
          AddNodeInput(fold_conv_node.name(), &fold_biasadd_node);
          AddNodeInput(add_node_1.name(), &fold_biasadd_node);
          new_nodes->push_back(fold_biasadd_node);
          inputs_to_rename[biasadd_node.name()] = fold_biasadd_node.name();
          nodes_to_ignore.insert(bn_node.name());
          inputs_to_rename[bn_node.name()] = fold_biasadd_node.name();

          return Status::OK();
        },
        {true}, &replaced_graph_def, allow_intersection));
    TF_RETURN_IF_ERROR(RenameNodeInputs(replaced_graph_def, inputs_to_rename,
                                        nodes_to_ignore, &current_graph_def));
  }

  {
    // remove dupicate node introduecd by allow allow_intersection in fold
    // conv + bias + BN with shared weights
    GraphDef processed_graph_def;
    TF_RETURN_IF_ERROR(
        RemoveDuplicateNode(current_graph_def, &processed_graph_def));
    current_graph_def = processed_graph_def;
  }

  {
    // Fold Conv2D + BatchNorm
    inputs_to_rename.clear();
    nodes_to_ignore.clear();
    GraphDef replaced_graph_def;
    TF_RETURN_IF_ERROR(ReplaceMatchingOpTypes(
        current_graph_def, // clang-format off
        {"FusedBatchNorm|FusedBatchNormV2|FusedBatchNormV3",                // batchnorm
          {
            {"Conv2D|MatMul|DepthwiseConv2dNative"},  // conv_node
            {"*"},  // beta_node
            {"*"},  // gamma_node
            {"*"},  // mean_node
            {"*"},  // variance_node
          }
        }, // clang-format on
        [&inputs_to_rename, &nodes_to_ignore, &bn_to_mm,
         &bn_to_mv](const NodeMatch &match, const std::set<string> &input_nodes,
                    const std::set<string> &output_nodes,
                    std::vector<NodeDef> *new_nodes) {
          // Find all the nodes we expect in the subgraph.
          const NodeDef &bn_node = match.node;
          const NodeDef &conv_node = match.inputs[0].node;
          const string input_node_name = conv_node.input()[0];
          const string weights_node_name = conv_node.input()[1];
          const NodeDef &gamma_node = match.inputs[1].node;
          const NodeDef &beta_node = match.inputs[2].node;
          const NodeDef &mean_node = match.inputs[3].node;
          const NodeDef &var_node = match.inputs[4].node;

          // CopyOriginalMatch(match, new_nodes);
          new_nodes->push_back(conv_node);
          new_nodes->push_back(bn_node);
          new_nodes->push_back(beta_node);
          new_nodes->push_back(gamma_node);
          new_nodes->push_back(mean_node);
          new_nodes->push_back(var_node);

          StringPiece current_scope_piece = conv_node.name();
          str_util::ConsumeSuffix(&current_scope_piece, "/Conv2D");
          string current_scope = string(current_scope_piece);
          string bn_fold_scope = current_scope + "/BatchNorm_Fold";

          // Construct the new nodes.
          NodeDef eps_node;
          eps_node.set_name(bn_fold_scope + "/eps");
          eps_node.set_op("Const");
          SetNodeAttr("dtype", DT_FLOAT, &eps_node);
          Tensor eps_tensor(DT_FLOAT, {1});
          eps_tensor.flat<float>()(0) = bn_node.attr().at("epsilon").f();
          SetNodeTensorAttr<float>("value", eps_tensor, &eps_node);
          new_nodes->push_back(eps_node);

          NodeDef add_node;
          add_node.set_name(bn_fold_scope + "/add");
          add_node.set_op("Add");
          SetNodeAttr("T", DT_FLOAT, &add_node);
          AddNodeInput(eps_node.name(), &add_node);
          // Use moving variance
          AddNodeInput(bn_to_mv[bn_node.name()], &add_node);
          new_nodes->push_back(add_node);

          NodeDef rsqrt_node;
          rsqrt_node.set_name(bn_fold_scope + "/rsqrt");
          rsqrt_node.set_op("Rsqrt");
          SetNodeAttr("T", DT_FLOAT, &rsqrt_node);
          AddNodeInput(add_node.name(), &rsqrt_node);
          new_nodes->push_back(rsqrt_node);

          NodeDef mul_node_0;
          mul_node_0.set_name(bn_fold_scope + "/mul_0");
          mul_node_0.set_op("Mul");
          SetNodeAttr("T", DT_FLOAT, &mul_node_0);
          AddNodeInput(rsqrt_node.name(), &mul_node_0);
          AddNodeInput(gamma_node.name(), &mul_node_0);
          new_nodes->push_back(mul_node_0);

          // Reshape scalar for depthwise conv
          NodeDef reshape_node;
          if (conv_node.op() == "DepthwiseConv2dNative") {
            NodeDef shape_node;
            shape_node.set_name(bn_fold_scope + "/shape");
            shape_node.set_op("Const");
            SetNodeAttr("dtype", DT_INT32, &shape_node);
            Tensor shape_tensor(DT_INT32, {2});
            shape_tensor.flat<int>()(0) = -1;
            shape_tensor.flat<int>()(1) = 1;
            SetNodeTensorAttr<float>("value", shape_tensor, &shape_node);
            new_nodes->push_back(shape_node);

            reshape_node.set_name(bn_fold_scope + "/reshape");
            reshape_node.set_op("Reshape");
            SetNodeAttr("T", DT_FLOAT, &reshape_node);
            SetNodeAttr("Tshape", DT_INT32, &reshape_node);
            AddNodeInput(mul_node_0.name(), &reshape_node);
            AddNodeInput(shape_node.name(), &reshape_node);
            new_nodes->push_back(reshape_node);
          }

          NodeDef mul_node_1;
          mul_node_1.set_name(bn_fold_scope + "/mul_1");
          mul_node_1.set_op("Mul");
          SetNodeAttr("T", DT_FLOAT, &mul_node_1);
          if (conv_node.op() == "DepthwiseConv2dNative") {
            AddNodeInput(reshape_node.name(), &mul_node_1);
          } else {
            AddNodeInput(mul_node_0.name(), &mul_node_1);
          }

          AddNodeInput(weights_node_name, &mul_node_1);
          new_nodes->push_back(mul_node_1);

          NodeDef fold_conv_node = conv_node;
          fold_conv_node.set_name(conv_node.name() + "_Fold");
          fold_conv_node.mutable_input()->Clear();
          AddNodeInput(input_node_name, &fold_conv_node);
          AddNodeInput(mul_node_1.name(), &fold_conv_node);
          new_nodes->push_back(fold_conv_node);
          inputs_to_rename[conv_node.name()] = fold_conv_node.name();
          nodes_to_ignore.insert(bn_node.name());

          NodeDef mul_node_2;
          mul_node_2.set_name(bn_fold_scope + "/mul_2");
          mul_node_2.set_op("Mul");
          SetNodeAttr("T", DT_FLOAT, &mul_node_2);
          AddNodeInput(mul_node_0.name(), &mul_node_2);
          // Use moving_mean
          AddNodeInput(bn_to_mm[bn_node.name()], &mul_node_2);
          new_nodes->push_back(mul_node_2);

          NodeDef sub_node;
          sub_node.set_name(bn_fold_scope + "/sub");
          sub_node.set_op("Sub");
          SetNodeAttr("T", DT_FLOAT, &sub_node);
          AddNodeInput(beta_node.name(), &sub_node);
          AddNodeInput(mul_node_2.name(), &sub_node);
          new_nodes->push_back(sub_node);

          NodeDef biasadd_node;
          biasadd_node.set_name(current_scope + "/biasadd");
          biasadd_node.set_op("BiasAdd");
          SetNodeAttr("T", DT_FLOAT, &biasadd_node);
          AddNodeInput(fold_conv_node.name(), &biasadd_node);
          AddNodeInput(sub_node.name(), &biasadd_node);
          new_nodes->push_back(biasadd_node);
          inputs_to_rename[bn_node.name()] = biasadd_node.name();

          return Status::OK();
        },
        {true}, &replaced_graph_def));
    TF_RETURN_IF_ERROR(RenameNodeInputs(replaced_graph_def, inputs_to_rename,
                                        nodes_to_ignore, &current_graph_def));
  }

  {
    // Fold Conv2D + Identity + BatchNorm from ssdlite mobilenetv1 fpn
    // predictor
    inputs_to_rename.clear();
    nodes_to_ignore.clear();
    GraphDef replaced_graph_def;

    TF_RETURN_IF_ERROR(ReplaceMatchingOpTypes(
        current_graph_def, // clang-format off
      {"FusedBatchNorm|FusedBatchNormV2|FusedBatchNormV3",                // batchnorm
        {
          {"Identity",
            {
              {"Conv2D|MatMul|DepthwiseConv2dNative"},  // conv_node
            }
          },
          {"*"},  // beta_node
          {"*"},  // gamma_node
          {"*"},  // mean_node
          {"*"},  // variance_node
        }
      }, // clang-format on
        [&inputs_to_rename, &nodes_to_ignore, &bn_to_mm,
         &bn_to_mv](const NodeMatch &match, const std::set<string> &input_nodes,
                    const std::set<string> &output_nodes,
                    std::vector<NodeDef> *new_nodes) {
          // Find all the nodes we expect in the subgraph.
          const NodeDef &bn_node = match.node;
          const NodeDef &front_identity_node = match.inputs[0].node;
          const NodeDef &conv_node = match.inputs[0].inputs[0].node;
          const string input_node_name = conv_node.input()[0];
          const string weights_node_name = conv_node.input()[1];
          const NodeDef &gamma_node = match.inputs[1].node;
          const NodeDef &beta_node = match.inputs[2].node;
          const NodeDef &mean_node = match.inputs[3].node;
          const NodeDef &var_node = match.inputs[4].node;

          // CopyOriginalMatch(match, new_nodes);
          new_nodes->push_back(front_identity_node);
          new_nodes->push_back(conv_node);
          new_nodes->push_back(bn_node);
          new_nodes->push_back(beta_node);
          new_nodes->push_back(gamma_node);
          new_nodes->push_back(mean_node);
          new_nodes->push_back(var_node);

          StringPiece current_scope_piece = conv_node.name();
          str_util::ConsumeSuffix(&current_scope_piece, "/Conv2D");
          string current_scope = string(current_scope_piece);
          string bn_fold_scope = current_scope + "/BatchNorm_Fold";

          // Construct the new nodes.
          NodeDef eps_node;
          eps_node.set_name(bn_fold_scope + "/eps");
          eps_node.set_op("Const");
          SetNodeAttr("dtype", DT_FLOAT, &eps_node);
          Tensor eps_tensor(DT_FLOAT, {1});
          eps_tensor.flat<float>()(0) = bn_node.attr().at("epsilon").f();
          SetNodeTensorAttr<float>("value", eps_tensor, &eps_node);
          new_nodes->push_back(eps_node);

          NodeDef add_node;
          add_node.set_name(bn_fold_scope + "/add");
          add_node.set_op("Add");
          SetNodeAttr("T", DT_FLOAT, &add_node);
          AddNodeInput(eps_node.name(), &add_node);
          // Use moving variance
          AddNodeInput(bn_to_mv[bn_node.name()], &add_node);
          new_nodes->push_back(add_node);

          NodeDef rsqrt_node;
          rsqrt_node.set_name(bn_fold_scope + "/rsqrt");
          rsqrt_node.set_op("Rsqrt");
          SetNodeAttr("T", DT_FLOAT, &rsqrt_node);
          AddNodeInput(add_node.name(), &rsqrt_node);
          new_nodes->push_back(rsqrt_node);

          NodeDef mul_node_0;
          mul_node_0.set_name(bn_fold_scope + "/mul_0");
          mul_node_0.set_op("Mul");
          SetNodeAttr("T", DT_FLOAT, &mul_node_0);
          AddNodeInput(rsqrt_node.name(), &mul_node_0);
          AddNodeInput(gamma_node.name(), &mul_node_0);
          new_nodes->push_back(mul_node_0);

          // Reshape scalar for depthwise conv
          NodeDef reshape_node;
          if (conv_node.op() == "DepthwiseConv2dNative") {
            NodeDef shape_node;
            shape_node.set_name(bn_fold_scope + "/shape");
            shape_node.set_op("Const");
            SetNodeAttr("dtype", DT_INT32, &shape_node);
            Tensor shape_tensor(DT_INT32, {2});
            shape_tensor.flat<int>()(0) = -1;
            shape_tensor.flat<int>()(1) = 1;
            SetNodeTensorAttr<float>("value", shape_tensor, &shape_node);
            new_nodes->push_back(shape_node);

            reshape_node.set_name(bn_fold_scope + "/reshape");
            reshape_node.set_op("Reshape");
            SetNodeAttr("T", DT_FLOAT, &reshape_node);
            SetNodeAttr("Tshape", DT_INT32, &reshape_node);
            AddNodeInput(mul_node_0.name(), &reshape_node);
            AddNodeInput(shape_node.name(), &reshape_node);
            new_nodes->push_back(reshape_node);
          }

          NodeDef mul_node_1;
          mul_node_1.set_name(bn_fold_scope + "/mul_1");
          mul_node_1.set_op("Mul");
          SetNodeAttr("T", DT_FLOAT, &mul_node_1);
          if (conv_node.op() == "DepthwiseConv2dNative") {
            AddNodeInput(reshape_node.name(), &mul_node_1);
          } else {
            AddNodeInput(mul_node_0.name(), &mul_node_1);
          }

          AddNodeInput(weights_node_name, &mul_node_1);
          new_nodes->push_back(mul_node_1);

          NodeDef fold_conv_node = conv_node;
          fold_conv_node.set_name(conv_node.name() + "_Fold");
          fold_conv_node.mutable_input()->Clear();
          AddNodeInput(input_node_name, &fold_conv_node);
          AddNodeInput(mul_node_1.name(), &fold_conv_node);
          new_nodes->push_back(fold_conv_node);
          inputs_to_rename[front_identity_node.name()] = fold_conv_node.name();
          nodes_to_ignore.insert(bn_node.name());

          NodeDef mul_node_2;
          mul_node_2.set_name(bn_fold_scope + "/mul_2");
          mul_node_2.set_op("Mul");
          SetNodeAttr("T", DT_FLOAT, &mul_node_2);
          AddNodeInput(mul_node_0.name(), &mul_node_2);
          // Use moving_mean
          AddNodeInput(bn_to_mm[bn_node.name()], &mul_node_2);
          new_nodes->push_back(mul_node_2);

          NodeDef sub_node;
          sub_node.set_name(bn_fold_scope + "/sub");
          sub_node.set_op("Sub");
          SetNodeAttr("T", DT_FLOAT, &sub_node);
          AddNodeInput(beta_node.name(), &sub_node);
          AddNodeInput(mul_node_2.name(), &sub_node);
          new_nodes->push_back(sub_node);

          NodeDef biasadd_node;
          biasadd_node.set_name(current_scope + "/biasadd");
          biasadd_node.set_op("BiasAdd");
          SetNodeAttr("T", DT_FLOAT, &biasadd_node);
          AddNodeInput(fold_conv_node.name(), &biasadd_node);
          AddNodeInput(sub_node.name(), &biasadd_node);
          new_nodes->push_back(biasadd_node);
          inputs_to_rename[bn_node.name()] = biasadd_node.name();

          return Status::OK();
        },
        {true}, &replaced_graph_def));
    TF_RETURN_IF_ERROR(RenameNodeInputs(replaced_graph_def, inputs_to_rename,
                                        nodes_to_ignore, &current_graph_def));
  }

  *output_graph_def = current_graph_def;
  return Status::OK();
}

Status FoldConvMulInference(const GraphDef &input_graph_def,
                            GraphDef *output_graph_def) {
  GraphDef current_graph_def = input_graph_def;

  std::map<string, string> inputs_to_rename;
  std::unordered_set<string> nodes_to_ignore;
  bool did_graph_change;

  // Fold Conv2D + BiasAdd + Mul
  do {
    did_graph_change = false;
    inputs_to_rename.clear();
    nodes_to_ignore.clear();
    GraphDef replaced_graph_def;
    TF_RETURN_IF_ERROR(ReplaceMatchingOpTypes(
        current_graph_def, // clang-format off
        {"Mul",                // mul_node
          {
            {"BiasAdd|Add|AddV2",  // biasadd_node
              {
                {"Conv2D|MatMul|DepthwiseConv2dNative",  // conv_node
                  {
                    {"*"},         // input_node
                    {"Const"},     // weights_node
                  }
                },
                {"Const"},         // bias_node
              }
            },
            {"Const"},  // mul_values_node
          }
        }, // clang-format on
        [&did_graph_change, &inputs_to_rename, &nodes_to_ignore](
            const NodeMatch &match, const std::set<string> &input_nodes,
            const std::set<string> &output_nodes,
            std::vector<NodeDef> *new_nodes) {
          // Find all the nodes we expect in the subgraph.
          const NodeDef &mul_node = match.node;
          const NodeDef &biasadd_node = match.inputs[0].node;
          const NodeDef &mul_values_node = match.inputs[1].node;
          const NodeDef &conv_node = match.inputs[0].inputs[0].node;
          const NodeDef &bias_node = match.inputs[0].inputs[1].node;
          const NodeDef &input_node = match.inputs[0].inputs[0].inputs[0].node;
          const NodeDef &weights_node =
              match.inputs[0].inputs[0].inputs[1].node;

          // Check that nodes that we use are not used somewhere else.
          for (const auto &node : {conv_node, weights_node, mul_values_node,
                                   biasadd_node, bias_node}) {
            if (output_nodes.count(node.name())) {
              // Return original nodes.
              CopyOriginalMatch(match, new_nodes);
              return Status::OK();
            }
          }

          Tensor bias = GetNodeTensorAttr(bias_node, "value");
          Tensor mul_values = GetNodeTensorAttr(mul_values_node, "value");
          Tensor weights = GetNodeTensorAttr(weights_node, "value");
          NodeDef scaled_weights_node;

          // Multiply the original weights by the scale vector.
          Tensor scaled_weights(DT_FLOAT, weights.shape());
          if (conv_node.op() == "DepthwiseConv2dNative") {
            const int64 weights_cols = weights.shape().dim_size(2);
            auto weights_flatten = weights.flat<float>();
            auto scaled_weights_flatten = scaled_weights.flat<float>();
            // If the mul-value is only one element, here must fetch the
            // first number.
            if (mul_values.NumElements() == 1) {
              for (int64 i = 0; i < weights.NumElements(); ++i) {
                scaled_weights_flatten(i) =
                    weights_flatten(i) * mul_values.flat<float>()(0);
              }
            } else {
              for (int64 i = 0; i < weights.NumElements(); ++i) {
                scaled_weights_flatten(i) =
                    weights_flatten(i) *
                    mul_values.flat<float>()(i % weights_cols);
              }
            }
          } else {
            const int64 weights_cols = weights.shape().dim_size(3);
            auto weights_matrix = weights.flat_inner_dims<float>();
            auto scaled_weights_matrix =
                scaled_weights.flat_inner_dims<float>();
            if (mul_values.NumElements() == 1) {
              for (int64 row = 0; row < weights_matrix.dimension(0); ++row) {
                for (int64 col = 0; col < weights_cols; ++col) {
                  scaled_weights_matrix(row, col) =
                      weights_matrix(row, col) * mul_values.flat<float>()(0);
                }
              }
            } else {
              for (int64 row = 0; row < weights_matrix.dimension(0); ++row) {
                for (int64 col = 0; col < weights_cols; ++col) {
                  scaled_weights_matrix(row, col) =
                      weights_matrix(row, col) * mul_values.flat<float>()(col);
                }
              }
            }
          }

          // Construct the new nodes.
          scaled_weights_node.set_op("Const");
          scaled_weights_node.set_name(weights_node.name());
          SetNodeAttr("dtype", DT_FLOAT, &scaled_weights_node);
          SetNodeTensorAttr<float>("value", scaled_weights,
                                   &scaled_weights_node);

          new_nodes->push_back(scaled_weights_node);
          new_nodes->push_back(input_node);
          new_nodes->push_back(conv_node);

          // Multiply the original biases by the scale vector and add the
          // add_values.
          Tensor scaled_bias(DT_FLOAT, bias.shape());
          auto bias_vector = bias.flat<float>();
          auto scaled_bias_vector = scaled_bias.flat<float>();
          if (mul_values.NumElements() == 1) {
            for (int64 col = 0; col < bias.NumElements(); ++col) {
              scaled_bias_vector(col) =
                  bias_vector(col) * mul_values.flat<float>()(0);
            }
          } else {
            for (int64 col = 0; col < bias.NumElements(); ++col) {
              scaled_bias_vector(col) =
                  bias_vector(col) * mul_values.flat<float>()(col);
            }
          }
          NodeDef scaled_bias_node;
          scaled_bias_node.set_op("Const");
          scaled_bias_node.set_name(bias_node.name());
          SetNodeAttr("dtype", DT_FLOAT, &scaled_bias_node);
          SetNodeTensorAttr<float>("value", scaled_bias, &scaled_bias_node);

          new_nodes->push_back(scaled_bias_node);

          inputs_to_rename[mul_node.name()] = biasadd_node.name();
          new_nodes->push_back(biasadd_node);

          did_graph_change = true;
          return Status::OK();
        },
        {true}, &replaced_graph_def));
    TF_RETURN_IF_ERROR(RenameNodeInputs(replaced_graph_def, inputs_to_rename,
                                        nodes_to_ignore, &current_graph_def));
  } while (did_graph_change);

  do {
    did_graph_change = false;
    inputs_to_rename.clear();
    nodes_to_ignore.clear();
    GraphDef replaced_graph_def;
    TF_RETURN_IF_ERROR(ReplaceMatchingOpTypes(
        current_graph_def, // clang-format off
        {"Mul",                // mul_node
          {
            {"Const"},  // mul_values_node
            {"BiasAdd|Add|AddV2",  // biasadd_node
              {
                {"Conv2D|MatMul|DepthwiseConv2dNative",  // conv_node
                  {
                    {"*"},         // input_node
                    {"Const"},     // weights_node
                  }
                },
                {"Const"},         // bias_node
              }
            },
          }
        }, // clang-format on
        [&did_graph_change, &inputs_to_rename, &nodes_to_ignore](
            const NodeMatch &match, const std::set<string> &input_nodes,
            const std::set<string> &output_nodes,
            std::vector<NodeDef> *new_nodes) {
          // Find all the nodes we expect in the subgraph.
          const NodeDef &mul_node = match.node;
          const NodeDef &biasadd_node = match.inputs[1].node;
          const NodeDef &mul_values_node = match.inputs[0].node;
          const NodeDef &conv_node = match.inputs[1].inputs[0].node;
          const NodeDef &bias_node = match.inputs[1].inputs[1].node;
          const NodeDef &input_node = match.inputs[1].inputs[0].inputs[0].node;
          const NodeDef &weights_node =
              match.inputs[1].inputs[0].inputs[1].node;

          // Check that nodes that we use are not used somewhere else.
          for (const auto &node : {conv_node, weights_node, mul_values_node,
                                   biasadd_node, bias_node}) {
            if (output_nodes.count(node.name())) {
              // Return original nodes.
              CopyOriginalMatch(match, new_nodes);
              return Status::OK();
            }
          }

          Tensor bias = GetNodeTensorAttr(bias_node, "value");
          Tensor mul_values = GetNodeTensorAttr(mul_values_node, "value");
          Tensor weights = GetNodeTensorAttr(weights_node, "value");
          NodeDef scaled_weights_node;

          // Multiply the original weights by the scale vector.
          Tensor scaled_weights(DT_FLOAT, weights.shape());
          if (conv_node.op() == "DepthwiseConv2dNative") {
            const int64 weights_cols = weights.shape().dim_size(2);
            auto weights_flatten = weights.flat<float>();
            auto scaled_weights_flatten = scaled_weights.flat<float>();
            // If the mul-value is only one element, here must fetch the
            // first number.
            if (mul_values.NumElements() == 1) {
              for (int64 i = 0; i < weights.NumElements(); ++i) {
                scaled_weights_flatten(i) =
                    weights_flatten(i) * mul_values.flat<float>()(0);
              }
            } else {
              for (int64 i = 0; i < weights.NumElements(); ++i) {
                scaled_weights_flatten(i) =
                    weights_flatten(i) *
                    mul_values.flat<float>()(i % weights_cols);
              }
            }
          } else {
            const int64 weights_cols = weights.shape().dim_size(3);
            auto weights_matrix = weights.flat_inner_dims<float>();
            auto scaled_weights_matrix =
                scaled_weights.flat_inner_dims<float>();
            if (mul_values.NumElements() == 1) {
              for (int64 row = 0; row < weights_matrix.dimension(0); ++row) {
                for (int64 col = 0; col < weights_cols; ++col) {
                  scaled_weights_matrix(row, col) =
                      weights_matrix(row, col) * mul_values.flat<float>()(0);
                }
              }
            } else {
              for (int64 row = 0; row < weights_matrix.dimension(0); ++row) {
                for (int64 col = 0; col < weights_cols; ++col) {
                  scaled_weights_matrix(row, col) =
                      weights_matrix(row, col) * mul_values.flat<float>()(col);
                }
              }
            }
          }

          // Construct the new nodes.
          scaled_weights_node.set_op("Const");
          scaled_weights_node.set_name(weights_node.name());
          SetNodeAttr("dtype", DT_FLOAT, &scaled_weights_node);
          SetNodeTensorAttr<float>("value", scaled_weights,
                                   &scaled_weights_node);

          new_nodes->push_back(scaled_weights_node);
          new_nodes->push_back(input_node);
          new_nodes->push_back(conv_node);

          // Multiply the original biases by the scale vector and add the
          // add_values.
          Tensor scaled_bias(DT_FLOAT, bias.shape());
          auto bias_vector = bias.flat<float>();
          auto scaled_bias_vector = scaled_bias.flat<float>();
          if (mul_values.NumElements() == 1) {
            for (int64 col = 0; col < bias.NumElements(); ++col) {
              scaled_bias_vector(col) =
                  bias_vector(col) * mul_values.flat<float>()(0);
            }
          } else {
            for (int64 col = 0; col < bias.NumElements(); ++col) {
              scaled_bias_vector(col) =
                  bias_vector(col) * mul_values.flat<float>()(col);
            }
          }
          NodeDef scaled_bias_node;
          scaled_bias_node.set_op("Const");
          scaled_bias_node.set_name(bias_node.name());
          SetNodeAttr("dtype", DT_FLOAT, &scaled_bias_node);
          SetNodeTensorAttr<float>("value", scaled_bias, &scaled_bias_node);

          new_nodes->push_back(scaled_bias_node);

          inputs_to_rename[mul_node.name()] = biasadd_node.name();
          new_nodes->push_back(biasadd_node);

          did_graph_change = true;
          return Status::OK();
        },
        {true}, &replaced_graph_def));
    TF_RETURN_IF_ERROR(RenameNodeInputs(replaced_graph_def, inputs_to_rename,
                                        nodes_to_ignore, &current_graph_def));
  } while (did_graph_change);

  *output_graph_def = current_graph_def;
  return Status::OK();
}

Status FoldBatchNormsInference(const GraphDef &input_graph_def,
                               GraphDef *output_graph_def) {
  GraphDef current_graph_def = input_graph_def;

  std::map<string, string> inputs_to_rename;
  std::unordered_set<string> nodes_to_ignore;
  bool did_graph_change;

  // Fold Conv2D + BiasAdd + BatchNorm
  do {
    did_graph_change = false;
    inputs_to_rename.clear();
    nodes_to_ignore.clear();
    GraphDef replaced_graph_def;
    // SaveGraphForDebugging(current_graph_def, "current_fold_bn.pb",
    //"quantize_results");
    bool allow_intersection = true;
    TF_RETURN_IF_ERROR(ReplaceMatchingOpTypes(
        current_graph_def, // clang-format off
        {"BiasAdd|Add|AddV2",  // add_node
          {
            {"Mul",                // mul_node
              {
                {"BiasAdd|Add|AddV2",  // biasadd_node
                  {
                    {"Conv2D|MatMul|DepthwiseConv2dNative|Conv2DBackpropInput",  // conv_node
                      {
                        {"*"},         // input_node
                        {"Const"},     // weights_node
                      }
                    },
                    {"Const"},         // bias_node
                  }
                },
                {"Const"},  // mul_values_node
              }
            },
            {"Const"},  // add_values_node
          }
        }, // clang-format on
        [&did_graph_change, &inputs_to_rename, &nodes_to_ignore](
            const NodeMatch &match, const std::set<string> &input_nodes,
            const std::set<string> &output_nodes,
            std::vector<NodeDef> *new_nodes) {
          // Find all the nodes we expect in the subgraph.
          const NodeDef &add_node = match.node;
          const NodeDef &add_values_node = match.inputs[1].node;
          const NodeDef &biasadd_node = match.inputs[0].inputs[0].node;
          const NodeDef &mul_node = match.inputs[0].node;
          const NodeDef &mul_values_node = match.inputs[0].inputs[1].node;
          const NodeDef &conv_node = match.inputs[0].inputs[0].inputs[0].node;
          const NodeDef &bias_node = match.inputs[0].inputs[0].inputs[1].node;
          const NodeDef &input_node =
              match.inputs[0].inputs[0].inputs[0].inputs[0].node;
          const NodeDef &weights_node =
              match.inputs[0].inputs[0].inputs[0].inputs[1].node;

          // Check that nodes that we use are not used somewhere else.
          for (const auto &node : {conv_node, weights_node, mul_node,
                                   mul_values_node, biasadd_node, bias_node}) {
            if (output_nodes.count(node.name())) {
              // Return original nodes.
              CopyOriginalMatch(match, new_nodes);
              DLOG_INFO(1) << "Skip fold convfc + bias + bn_add + bn_mul: "
                           << conv_node.name() << " + " << biasadd_node.name()
                           << " + " << add_values_node.name() << " + "
                           << mul_values_node.name();
              return Status::OK();
            }
          }

          Tensor bias = GetNodeTensorAttr(bias_node, "value");
          Tensor mul_values = GetNodeTensorAttr(mul_values_node, "value");
          Tensor add_values = GetNodeTensorAttr(add_values_node, "value");

          DLOG_INFO(1) << "Fold convfc + bias + bn_add + bn_mul: "
                       << conv_node.name() << " + " << biasadd_node.name()
                       << " + " << add_values_node.name() << " + "
                       << mul_values_node.name();

          // Multiply the original biases by the scale vector and add the
          // add_values.
          Tensor scaled_bias(DT_FLOAT, bias.shape());
          auto bias_vector = bias.flat<float>();
          auto scaled_bias_vector = scaled_bias.flat<float>();
          for (int64 col = 0; col < bias.NumElements(); ++col) {
            scaled_bias_vector(col) =
                bias_vector(col) * mul_values.flat<float>()(col) +
                add_values.flat<float>()(col);
          }

          // Construct the new nodes.
          NodeDef scaled_weights_node;
          TF_RETURN_IF_ERROR(GetMergedConvWeights(
              conv_node, weights_node, mul_values_node, &scaled_weights_node));
          new_nodes->push_back(scaled_weights_node);

          new_nodes->push_back(input_node);
          new_nodes->push_back(conv_node);

          NodeDef scaled_bias_node;
          TF_RETURN_IF_ERROR(GetMergedConvBiases(
              bias_node, mul_values_node, add_values_node, &scaled_bias_node));
          new_nodes->push_back(scaled_bias_node);

          inputs_to_rename[add_node.name()] = biasadd_node.name();
          new_nodes->push_back(biasadd_node);

          did_graph_change = true;
          return Status::OK();
        },
        {true}, &replaced_graph_def, allow_intersection));
    TF_RETURN_IF_ERROR(RenameNodeInputs(replaced_graph_def, inputs_to_rename,
                                        nodes_to_ignore, &current_graph_def));
  } while (did_graph_change);

  {
    // remove dupicate node introduecd by allow allow_intersection in fold
    // conv + bias + BN, duplicated input node
    GraphDef processed_graph_def;
    TF_RETURN_IF_ERROR(
        RemoveDuplicateNode(current_graph_def, &processed_graph_def));
    current_graph_def = processed_graph_def;
  }

  // Fold Conv2D + BatchNorm
  do {
    did_graph_change = false;
    inputs_to_rename.clear();
    nodes_to_ignore.clear();
    GraphDef replaced_graph_def;
    TF_RETURN_IF_ERROR(
        ReplaceMatchingOpTypes(
            current_graph_def, // clang-format off
        {"BiasAdd|Add|AddV2",
          {
            {"Mul",                // mul_node
              {
                {"Conv2D|MatMul|DepthwiseConv2dNative|Conv2DBackpropInput",  // conv_node
                  {
                    {"*"},         // input_node
                    {"Const"},     // weights_node
                  }
                },
                {"Const"},         // mul_values_node
              }
            },
            {"Const"},  // add_values_node
          }
        }, // clang-format on
            [&did_graph_change, &inputs_to_rename, &nodes_to_ignore](
                const NodeMatch &match, const std::set<string> &input_nodes,
                const std::set<string> &output_nodes,
                std::vector<NodeDef> *new_nodes) {
              // Find all the nodes we expect in the subgraph.
              const NodeDef &add_node = match.node;
              // const NodeDef& mul_node = match.inputs[0].node;
              const NodeDef &add_values_node = match.inputs[1].node;
              const NodeDef &conv_node = match.inputs[0].inputs[0].node;
              const NodeDef &input_node =
                  match.inputs[0].inputs[0].inputs[0].node;
              const NodeDef &weights_node =
                  match.inputs[0].inputs[0].inputs[1].node;
              const NodeDef &mul_values_node = match.inputs[0].inputs[1].node;

              DLOG_INFO(1) << "Fold convfc + bn_add + bn_mul: "
                           << conv_node.name() << " + "
                           << add_values_node.name() << " + "
                           << mul_values_node.name();

              // Check that nodes that we use are not used somewhere else.
              for (const auto &node :
                   {conv_node, weights_node, mul_values_node}) {
                if (output_nodes.count(node.name())) {
                  // Return original nodes.
                  CopyOriginalMatch(match, new_nodes);
                  return Status::OK();
                }
              }

              // Construct the new nodes.
              NodeDef scaled_weights_node;
              TF_RETURN_IF_ERROR(GetMergedConvWeights(conv_node, weights_node,
                                                      mul_values_node,
                                                      &scaled_weights_node));
              new_nodes->push_back(scaled_weights_node);

              new_nodes->push_back(input_node);
              new_nodes->push_back(conv_node);

              StringPiece scope_name = weights_node.name();
              str_util::ConsumeSuffix(&scope_name, "/weights");
              str_util::ConsumeSuffix(&scope_name, "/depthwise_weights");

              NodeDef new_add_values_node = add_values_node;
              new_add_values_node.set_name((string)scope_name + "/biases");
              new_nodes->push_back(new_add_values_node);

              NodeDef new_add_node = add_node;
              new_add_node.set_op(add_node.op());
              new_add_node.mutable_input()->Clear();
              AddNodeInput(conv_node.name(), &new_add_node);
              AddNodeInput(new_add_values_node.name(), &new_add_node);
              new_nodes->push_back(new_add_node);

              did_graph_change = true;
              return Status::OK();
            },
            {true}, &replaced_graph_def));
    TF_RETURN_IF_ERROR(RenameNodeInputs(replaced_graph_def, inputs_to_rename,
                                        nodes_to_ignore, &current_graph_def));
  } while (did_graph_change);

  // Fold Conv2DBackpropInput + BiasAdd + BatchNorm
  do {
    did_graph_change = false;
    inputs_to_rename.clear();
    nodes_to_ignore.clear();
    GraphDef replaced_graph_def;
    TF_RETURN_IF_ERROR(ReplaceMatchingOpTypes(
        current_graph_def, // clang-format off
        {"BiasAdd|Add|AddV2",  // add_node
          {
            {"Mul",                // mul_node
              {
                {"BiasAdd|Add|AddV2",  // biasadd_node
                  {
                    {"Conv2DBackpropInput",  // conv_node
                      {
                        {"Pack"},         // pack_node
                        {"Const"},     // weights_node
                        {"*"},         // input_node
                      }
                    },
                    {"Const"},         // bias_node
                  }
                },
                {"Const"},  // mul_values_node
              }
            },
            {"Const"},  // add_values_node
          }
        }, // clang-format on
        [&did_graph_change, &inputs_to_rename, &nodes_to_ignore](
            const NodeMatch &match, const std::set<string> &input_nodes,
            const std::set<string> &output_nodes,
            std::vector<NodeDef> *new_nodes) {
          // Find all the nodes we expect in the subgraph.
          const NodeDef &add_node = match.node;
          // const NodeDef& mul_node = match.inputs[0].node;
          const NodeDef &add_values_node = match.inputs[1].node;
          const NodeDef &biasadd_node = match.inputs[0].inputs[0].node;
          const NodeDef &mul_values_node = match.inputs[0].inputs[1].node;
          const NodeDef &conv_node = match.inputs[0].inputs[0].inputs[0].node;
          const NodeDef &bias_node = match.inputs[0].inputs[0].inputs[1].node;
          const NodeDef &pack_node =
              match.inputs[0].inputs[0].inputs[0].inputs[0].node;
          const NodeDef &weights_node =
              match.inputs[0].inputs[0].inputs[0].inputs[1].node;
          const NodeDef &input_node =
              match.inputs[0].inputs[0].inputs[0].inputs[2].node;

          DLOG_INFO(1) << "Fold conv2d_transpose + bias + bn_add + bn_mul: "
                       << conv_node.name() << " + " << biasadd_node.name()
                       << " + " << add_values_node.name() << " + "
                       << mul_values_node.name();

          // Check that nodes that we use are not used somewhere else.
          for (const auto &node : {conv_node, weights_node, mul_values_node,
                                   biasadd_node, bias_node}) {
            if (output_nodes.count(node.name())) {
              // Return original nodes.
              CopyOriginalMatch(match, new_nodes);
              return Status::OK();
            }
          }

          Tensor bias = GetNodeTensorAttr(bias_node, "value");
          Tensor mul_values = GetNodeTensorAttr(mul_values_node, "value");
          Tensor add_values = GetNodeTensorAttr(add_values_node, "value");

          // Multiply the original biases by the scale vector and add the
          // add_values.
          Tensor scaled_bias(DT_FLOAT, bias.shape());
          auto bias_vector = bias.flat<float>();
          auto scaled_bias_vector = scaled_bias.flat<float>();
          for (int64 col = 0; col < bias.NumElements(); ++col) {
            scaled_bias_vector(col) =
                bias_vector(col) * mul_values.flat<float>()(col) +
                add_values.flat<float>()(col);
          }

          // Construct the new nodes.
          NodeDef scaled_weights_node;
          TF_RETURN_IF_ERROR(GetMergedConvWeights(
              conv_node, weights_node, mul_values_node, &scaled_weights_node));
          new_nodes->push_back(scaled_weights_node);

          new_nodes->push_back(input_node);
          new_nodes->push_back(pack_node);
          new_nodes->push_back(conv_node);

          NodeDef scaled_bias_node;
          TF_RETURN_IF_ERROR(GetMergedConvBiases(
              bias_node, mul_values_node, add_values_node, &scaled_bias_node));
          new_nodes->push_back(scaled_bias_node);

          inputs_to_rename[add_node.name()] = biasadd_node.name();
          new_nodes->push_back(biasadd_node);

          did_graph_change = true;
          return Status::OK();
        },
        {true}, &replaced_graph_def));
    TF_RETURN_IF_ERROR(RenameNodeInputs(replaced_graph_def, inputs_to_rename,
                                        nodes_to_ignore, &current_graph_def));
  } while (did_graph_change);

  // Fold Conv2DBackpropInput + BatchNorm
  do {
    did_graph_change = false;
    inputs_to_rename.clear();
    nodes_to_ignore.clear();
    GraphDef replaced_graph_def;
    TF_RETURN_IF_ERROR(ReplaceMatchingOpTypes(
        current_graph_def, // clang-format off
        {"BiasAdd|Add|AddV2",
          {
            {"Mul",                // mul_node
              {
                {"Conv2DBackpropInput",  // conv_node
                  {
                    {"Pack"},         // pack_node
                    {"Const"},     // weights_node
                    {"*"},         // input_node
                  }
                },
                {"Const"},         // mul_values_node
              }
            },
            {"Const"},  // add_values_node
          }
        }, // clang-format on
        [&did_graph_change, &inputs_to_rename, &nodes_to_ignore](
            const NodeMatch &match, const std::set<string> &input_nodes,
            const std::set<string> &output_nodes,
            std::vector<NodeDef> *new_nodes) {
          // Find all the nodes we expect in the subgraph.
          const NodeDef &add_node = match.node;
          // const NodeDef& mul_node = match.inputs[0].node;
          const NodeDef &add_values_node = match.inputs[1].node;
          const NodeDef &conv_node = match.inputs[0].inputs[0].node;
          const NodeDef &pack_node = match.inputs[0].inputs[0].inputs[0].node;
          const NodeDef &weights_node =
              match.inputs[0].inputs[0].inputs[1].node;
          const NodeDef &input_node = match.inputs[0].inputs[0].inputs[2].node;
          const NodeDef &mul_values_node = match.inputs[0].inputs[1].node;

          // Check that nodes that we use are not used somewhere else.
          for (const auto &node : {conv_node, weights_node, mul_values_node}) {
            if (output_nodes.count(node.name())) {
              // Return original nodes.
              CopyOriginalMatch(match, new_nodes);
              return Status::OK();
            }
          }

          DLOG_INFO(1) << "Fold conv2d_transpose + bn_add + bn_mul: "
                       << conv_node.name() << " + " << add_values_node.name()
                       << " + " << mul_values_node.name();

          // Construct the new nodes.
          NodeDef scaled_weights_node;
          TF_RETURN_IF_ERROR(GetMergedConvWeights(
              conv_node, weights_node, mul_values_node, &scaled_weights_node));
          new_nodes->push_back(scaled_weights_node);

          new_nodes->push_back(pack_node);
          new_nodes->push_back(input_node);
          new_nodes->push_back(conv_node);

          StringPiece scope_name = weights_node.name();
          str_util::ConsumeSuffix(&scope_name, "/weights");
          str_util::ConsumeSuffix(&scope_name, "/depthwise_weights");

          NodeDef new_add_values_node = add_values_node;
          new_add_values_node.set_name((string)scope_name + "/biases");
          new_nodes->push_back(new_add_values_node);

          NodeDef new_add_node = add_node;
          new_add_node.set_op(add_node.op());
          new_add_node.mutable_input()->Clear();
          AddNodeInput(conv_node.name(), &new_add_node);
          AddNodeInput(new_add_values_node.name(), &new_add_node);
          new_nodes->push_back(new_add_node);

          did_graph_change = true;
          return Status::OK();
        },
        {true}, &replaced_graph_def));
    TF_RETURN_IF_ERROR(RenameNodeInputs(replaced_graph_def, inputs_to_rename,
                                        nodes_to_ignore, &current_graph_def));
  } while (did_graph_change);

  // Fold MatMul + Reshape + BatchNorm
  do {
    did_graph_change = false;
    inputs_to_rename.clear();
    nodes_to_ignore.clear();
    GraphDef replaced_graph_def;
    TF_RETURN_IF_ERROR(ReplaceMatchingOpTypes(
        current_graph_def, // clang-format off
        {"BiasAdd|Add|AddV2",
          {
            {"Mul",                // mul_node
              {
                {"Reshape",       // reshape_node
                  {
                    {"MatMul",  // matmul_node
                      {
                        {"*"},         // input_node
                        {"Const"},     // weights_node
                      }
                    },
                    {"*"},         // shape_node
                  }
                },
                {"Const"},         // mul_values_node
              }
            },
            {"Const"},  // add_values_node
          }
        }, // clang-format on
        [&did_graph_change, &inputs_to_rename, &nodes_to_ignore](
            const NodeMatch &match, const std::set<string> &input_nodes,
            const std::set<string> &output_nodes,
            std::vector<NodeDef> *new_nodes) {
          // Find all the nodes we expect in the subgraph.
          const NodeDef &add_node = match.node;
          // const NodeDef& mul_node = match.inputs[0].node;
          const NodeDef &add_values_node = match.inputs[1].node;
          const NodeDef &reshape_node = match.inputs[0].inputs[0].node;
          const NodeDef &shape_node = match.inputs[0].inputs[0].inputs[1].node;

          const NodeDef &matmul_node = match.inputs[0].inputs[0].inputs[0].node;
          const NodeDef &input_node =
              match.inputs[0].inputs[0].inputs[0].inputs[0].node;
          const NodeDef &weights_node =
              match.inputs[0].inputs[0].inputs[0].inputs[1].node;
          const NodeDef &mul_values_node = match.inputs[0].inputs[1].node;

          // Check that nodes that we use are not used somewhere else.
          for (const auto &node : {weights_node, mul_values_node}) {
            if (output_nodes.count(node.name())) {
              // Return original nodes.
              CopyOriginalMatch(match, new_nodes);
              return Status::OK();
            }
          }

          // Construct the new nodes.
          NodeDef scaled_weights_node;
          TF_RETURN_IF_ERROR(GetMergedConvWeights(matmul_node, weights_node,
                                                  mul_values_node,
                                                  &scaled_weights_node));
          new_nodes->push_back(scaled_weights_node);

          new_nodes->push_back(input_node);
          new_nodes->push_back(matmul_node);

          StringPiece scope_name = weights_node.name();
          str_util::ConsumeSuffix(&scope_name, "/weights");
          str_util::ConsumeSuffix(&scope_name, "/depthwise_weights");

          NodeDef new_add_values_node = add_values_node;
          new_add_values_node.set_name((string)scope_name + "/biases");
          new_nodes->push_back(new_add_values_node);

          NodeDef new_add_node = add_node;
          new_add_node.set_op(add_node.op());
          new_add_node.mutable_input()->Clear();
          AddNodeInput(matmul_node.name(), &new_add_node);
          AddNodeInput(new_add_values_node.name(), &new_add_node);
          new_nodes->push_back(new_add_node);

          did_graph_change = true;
          return Status::OK();
        },
        {true}, &replaced_graph_def));
    TF_RETURN_IF_ERROR(RenameNodeInputs(replaced_graph_def, inputs_to_rename,
                                        nodes_to_ignore, &current_graph_def));
  } while (did_graph_change);

  // Fold Atrous_Conv2D + BiasAdd + BatchNorm
  do {
    did_graph_change = false;
    inputs_to_rename.clear();
    nodes_to_ignore.clear();
    GraphDef replaced_graph_def;
    TF_RETURN_IF_ERROR(ReplaceMatchingOpTypes(
        current_graph_def, // clang-format off
        {"BiasAdd|Add|AddV2", // add_node
          {
            {"Mul",                // mul_node
              {
                {"BiasAdd|Add|AddV2",    // bias_add_node
                  {
                    {"BatchToSpaceND", // batch_to_space node
                      {
                        {"Conv2D|DepthwiseConv2dNative", // conv node
                          {
                            {"SpaceToBatchND", // space_to_batch node
                              {
                                {"*"},  // input node
                                {"*"},  // block shape node
                                {"*"},  // paddings node
                              }
                            },
                            {"Const"}, // weights node
                          }
                        },
                        {"*"}, // block shape node
                        {"*"}, // crops node
                      }
                    },
                    {"Const"}, // biases node
                  }
                },
                {"Const"},  // mul_values_node
              }
            },
            {"Const"},  // add_values_node
          }
        }, // clang-format on
        [&did_graph_change, &inputs_to_rename, &nodes_to_ignore](
            const NodeMatch &match, const std::set<string> &input_nodes,
            const std::set<string> &output_nodes,
            std::vector<NodeDef> *new_nodes) {
          // Find all the nodes we expect in the subgraph.
          const NodeDef &add_node = match.node;
          // const NodeDef& mul_node = match.inputs[0].node;
          const NodeDef &mul_values_node = match.inputs[0].inputs[1].node;
          const NodeDef &add_values_node = match.inputs[1].node;
          const NodeDef &biasadd_node = match.inputs[0].inputs[0].node;
          const NodeDef &bias_node = match.inputs[0].inputs[0].inputs[1].node;
          const NodeDef &batch_to_space_node =
              match.inputs[0].inputs[0].inputs[0].node;
          const NodeDef &batch_to_space_block_shape_node =
              match.inputs[0].inputs[0].inputs[0].inputs[1].node;
          const NodeDef &batch_to_space_crops_node =
              match.inputs[0].inputs[0].inputs[0].inputs[2].node;
          const NodeDef &conv_node =
              match.inputs[0].inputs[0].inputs[0].inputs[0].node;
          const NodeDef &weights_node =
              match.inputs[0].inputs[0].inputs[0].inputs[0].inputs[1].node;
          const NodeDef &space_to_batch_node =
              match.inputs[0].inputs[0].inputs[0].inputs[0].inputs[0].node;
          const NodeDef &space_to_batch_block_shape_node = match.inputs[0]
                                                               .inputs[0]
                                                               .inputs[0]
                                                               .inputs[0]
                                                               .inputs[0]
                                                               .inputs[1]
                                                               .node;
          const NodeDef &space_to_batch_paddings_node = match.inputs[0]
                                                            .inputs[0]
                                                            .inputs[0]
                                                            .inputs[0]
                                                            .inputs[0]
                                                            .inputs[2]
                                                            .node;
          const NodeDef &input_node = match.inputs[0]
                                          .inputs[0]
                                          .inputs[0]
                                          .inputs[0]
                                          .inputs[0]
                                          .inputs[0]
                                          .node;

          // Check that nodes that we use are not used somewhere else.
          for (const auto &node : {conv_node, weights_node, mul_values_node}) {
            if (output_nodes.count(node.name())) {
              // Return original nodes.
              CopyOriginalMatch(match, new_nodes);
              return Status::OK();
            }
          }

          // Construct the new nodes.
          new_nodes->push_back(input_node);
          new_nodes->push_back(space_to_batch_paddings_node);
          new_nodes->push_back(space_to_batch_block_shape_node);
          new_nodes->push_back(space_to_batch_node);

          NodeDef scaled_weights_node;
          TF_RETURN_IF_ERROR(GetMergedConvWeights(
              conv_node, weights_node, mul_values_node, &scaled_weights_node));
          new_nodes->push_back(scaled_weights_node);

          new_nodes->push_back(conv_node);
          new_nodes->push_back(batch_to_space_block_shape_node);
          new_nodes->push_back(batch_to_space_crops_node);
          new_nodes->push_back(batch_to_space_node);

          NodeDef scaled_bias_node;
          TF_RETURN_IF_ERROR(GetMergedConvBiases(
              bias_node, mul_values_node, add_values_node, &scaled_bias_node));
          new_nodes->push_back(scaled_bias_node);

          inputs_to_rename[add_node.name()] = biasadd_node.name();
          new_nodes->push_back(biasadd_node);

          did_graph_change = true;
          return Status::OK();
        },
        {true}, &replaced_graph_def));
    TF_RETURN_IF_ERROR(RenameNodeInputs(replaced_graph_def, inputs_to_rename,
                                        nodes_to_ignore, &current_graph_def));
  } while (did_graph_change);

  // Fold Atrous_Conv2D + BatchNorm
  do {
    did_graph_change = false;
    inputs_to_rename.clear();
    nodes_to_ignore.clear();
    GraphDef replaced_graph_def;
    TF_RETURN_IF_ERROR(ReplaceMatchingOpTypes(
        current_graph_def, // clang-format off
        {"BiasAdd|Add|AddV2", // add_node
          {
            {"Mul", // mul_node
              {
                {"BatchToSpaceND", // batch_to_space node
                  {
                    {"Conv2D|DepthwiseConv2dNative", // conv node
                      {
                        {"SpaceToBatchND", // space_to_batch node
                          {
                            {"*"},  // input node
                            {"*"},  // block shape node
                            {"*"},  // paddings node
                          }
                        },
                        {"Const"}, // weights node
                      }
                    },
                    {"*"}, // block shape node
                    {"*"}, // crops node
                  }
                },
                {"Const"},  // mul_values_node
              }
            },
            {"Const"},  // add_values_node
          }
        }, // clang-format on
        [&did_graph_change, &inputs_to_rename, &nodes_to_ignore](
            const NodeMatch &match, const std::set<string> &input_nodes,
            const std::set<string> &output_nodes,
            std::vector<NodeDef> *new_nodes) {
          // Find all the nodes we expect in the subgraph.
          const NodeDef &add_node = match.node;
          // const NodeDef& mul_node = match.inputs[0].node;
          const NodeDef &mul_values_node = match.inputs[0].inputs[1].node;
          const NodeDef &add_values_node = match.inputs[1].node;

          const NodeDef &batch_to_space_node = match.inputs[0].inputs[0].node;
          const NodeDef &batch_to_space_block_shape_node =
              match.inputs[0].inputs[0].inputs[1].node;
          const NodeDef &batch_to_space_crops_node =
              match.inputs[0].inputs[0].inputs[2].node;
          const NodeDef &conv_node = match.inputs[0].inputs[0].inputs[0].node;
          const NodeDef &weights_node =
              match.inputs[0].inputs[0].inputs[0].inputs[1].node;
          const NodeDef &space_to_batch_node =
              match.inputs[0].inputs[0].inputs[0].inputs[0].node;
          const NodeDef &space_to_batch_block_shape_node =
              match.inputs[0].inputs[0].inputs[0].inputs[0].inputs[1].node;
          const NodeDef &space_to_batch_paddings_node =
              match.inputs[0].inputs[0].inputs[0].inputs[0].inputs[2].node;
          const NodeDef &input_node =
              match.inputs[0].inputs[0].inputs[0].inputs[0].inputs[0].node;

          // Check that nodes that we use are not used somewhere else.
          for (const auto &node : {conv_node, weights_node, mul_values_node}) {
            if (output_nodes.count(node.name())) {
              // Return original nodes.
              CopyOriginalMatch(match, new_nodes);
              return Status::OK();
            }
          }

          // Construct the new nodes.
          new_nodes->push_back(input_node);
          new_nodes->push_back(space_to_batch_paddings_node);
          new_nodes->push_back(space_to_batch_block_shape_node);
          new_nodes->push_back(space_to_batch_node);

          NodeDef scaled_weights_node;
          TF_RETURN_IF_ERROR(GetMergedConvWeights(
              conv_node, weights_node, mul_values_node, &scaled_weights_node));
          new_nodes->push_back(scaled_weights_node);

          new_nodes->push_back(conv_node);
          new_nodes->push_back(batch_to_space_block_shape_node);
          new_nodes->push_back(batch_to_space_crops_node);
          new_nodes->push_back(batch_to_space_node);
          new_nodes->push_back(add_values_node);

          NodeDef new_add_node = add_node;
          new_add_node.set_op("BiasAdd");
          new_add_node.mutable_input()->Clear();
          AddNodeInput(batch_to_space_node.name(), &new_add_node);
          AddNodeInput(add_values_node.name(), &new_add_node);
          new_nodes->push_back(new_add_node);

          did_graph_change = true;
          return Status::OK();
        },
        {true}, &replaced_graph_def));
    TF_RETURN_IF_ERROR(RenameNodeInputs(replaced_graph_def, inputs_to_rename,
                                        nodes_to_ignore, &current_graph_def));
  } while (did_graph_change);

  *output_graph_def = current_graph_def;
  return Status::OK();
}

Status FoldBatchNorms(const GraphDef &input_graph_def,
                      GraphDef *output_graph_def, bool is_training) {
  if (is_training) {
    return FoldBatchNormsTraining(input_graph_def, output_graph_def);
  } else {
    return FoldBatchNormsInference(input_graph_def, output_graph_def);
  }
}

// Command Wrapper
Status FoldBatchNormsCommand(const GraphDef &input_graph_def,
                             const TransformFuncContext &context,
                             GraphDef *output_graph_def) {
  return (FoldBatchNorms(input_graph_def, output_graph_def));
}

REGISTER_DECENT_Q_GRAPH_TRANSFORM("fold_batch_norms", FoldBatchNormsCommand);

} // namespace decent_q
} // namespace tensorflow
