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

#include "known_patterns.h"

namespace tensorflow {
namespace decent_q {

// clang-format off

// input pattern
const OpTypePattern input_pattern({"*"});

// Weight pattern
const OpTypePattern weight_pattern({"Const|Identity|ReadVariableOp|Mul"});

// Weight pattern
const OpTypePattern bias_pattern({"Const|Identity|ReadVariableOp|Sub|Add|AddV2"});

// Placeholder
const OpTypePattern placeholder_pattern({"Placeholder"});
DEFINE_GET_INPUT_NODES(Placeholder) {
  return std::vector<const NodeDef*> ();
}
DEFINE_GET_WEIGHTS_NODES(Placeholder) {
  return std::vector<const NodeDef*> ();
}
const PlaceholderPattern placeholder_pattern_wrapper(placeholder_pattern, "placeholder");

// AtrousConv
const OpTypePattern atrous_conv_pattern(
    {"BatchToSpaceND",
      {
        {"Conv2D|DepthwiseConv2dNative", // conv node
          {
            {"SpaceToBatchND",
              {
                input_pattern,  // input node
                {"*"},  // block shape node
                {"*"},  // paddings node
              }
            },
            weight_pattern, // weights node
          }
        },
        {"*"}, // block shape node
        {"*"}, // crops node
      }
    });
DEFINE_GET_INPUT_NODES(AtrousConv) {
  std::vector<const NodeDef*> input_nodes;
  input_nodes.push_back(&(match.inputs[0].inputs[0].inputs[0].node));
  return input_nodes;
}
DEFINE_GET_WEIGHTS_NODES(AtrousConv) {
  std::vector<const NodeDef*> weights_nodes;
  weights_nodes.push_back(&(match.inputs[0].inputs[1].node));
  return weights_nodes;
}
const AtrousConvPattern atrous_conv_pattern_wrapper(atrous_conv_pattern, "atrous_conv");

// AtrousConv + bias
const OpTypePattern atrous_conv_bias_pattern(
    {"BiasAdd|Add|AddV2",
      {
        atrous_conv_pattern,
        bias_pattern, // bias node
      }
    });
DEFINE_GET_INPUT_NODES(AtrousConvBias) {
  std::vector<const NodeDef*> input_nodes;
  input_nodes.push_back(&(match.inputs[0].inputs[0].inputs[0].inputs[0].node));
  return input_nodes;
}
DEFINE_GET_WEIGHTS_NODES(AtrousConvBias) {
  std::vector<const NodeDef*> weights_nodes;
  weights_nodes.push_back(&(match.inputs[0].inputs[0].inputs[1].node));
  weights_nodes.push_back(&(match.inputs[1].node));
  return weights_nodes;
}
const AtrousConvBiasPattern atrous_conv_bias_pattern_wrapper(atrous_conv_bias_pattern, "atrous_conv_bias");

// AtrousConv + relu
const OpTypePattern atrous_conv_relu_pattern(
    {"Relu|Relu6",
      {
        atrous_conv_pattern,
      }
    });
DEFINE_GET_INPUT_NODES(AtrousConvRelu) {
  std::vector<const NodeDef*> input_nodes;
  input_nodes.push_back(&(match.inputs[0].inputs[0].inputs[0].inputs[0].node));
  return input_nodes;
}
DEFINE_GET_WEIGHTS_NODES(AtrousConvRelu) {
  std::vector<const NodeDef*> weights_nodes;
  weights_nodes.push_back(&(match.inputs[0].inputs[0].inputs[1].node));
  return weights_nodes;
}
const AtrousConvReluPattern atrous_conv_relu_pattern_wrapper(atrous_conv_relu_pattern, "atrous_conv_relu");

// AtrousConv
const OpTypePattern atrous_conv_bias_relu_pattern(
    {"Relu|Relu6",
      {
        atrous_conv_bias_pattern,
      }
    });
DEFINE_GET_INPUT_NODES(AtrousConvBiasRelu) {
  std::vector<const NodeDef*> input_nodes;
  input_nodes.push_back(&(match.inputs[0].inputs[0].inputs[0].inputs[0].inputs[0].node));
  return input_nodes;
}
DEFINE_GET_WEIGHTS_NODES(AtrousConvBiasRelu) {
    std::vector<const NodeDef*> weights_nodes;
    weights_nodes.push_back(&(match.inputs[0].inputs[0].inputs[0].inputs[1].node));
    weights_nodes.push_back(&(match.inputs[0].inputs[1].node));
    return weights_nodes;
}
const AtrousConvBiasReluPattern atrous_conv_bias_relu_pattern_wrapper(atrous_conv_bias_relu_pattern, "atrous_conv_bias_relu");

// ConvFc
const OpTypePattern convfc_pattern(
    {"Conv2D|DepthwiseConv2d|DepthwiseConv2dNative|MatMul|Conv3D",
      {
        input_pattern, // input node
        weight_pattern, // weight node
      }
    });
DEFINE_GET_INPUT_NODES(Convfc) {
  std::vector<const NodeDef*> input_nodes;
  input_nodes.push_back(&(match.inputs[0].node));
  return input_nodes;
}
DEFINE_GET_WEIGHTS_NODES(Convfc) {
  std::vector<const NodeDef*> weights_nodes;
  weights_nodes.push_back(&(match.inputs[1].node));
  return weights_nodes;
}
const ConvfcPattern convfc_pattern_wrapper(convfc_pattern, "convfc");

// ConvFc + relu
const OpTypePattern convfc_relu_pattern(
    {"Relu|Relu6",
      {
        convfc_pattern,
      }
    });
DEFINE_GET_INPUT_NODES(ConvfcRelu) {
  std::vector<const NodeDef*> input_nodes;
  input_nodes.push_back(&(match.inputs[0].inputs[0].node));
  return input_nodes;
}
DEFINE_GET_WEIGHTS_NODES(ConvfcRelu) {
  std::vector<const NodeDef*> weights_nodes;
  weights_nodes.push_back(&(match.inputs[0].inputs[1].node));
  return weights_nodes;
}
const ConvfcReluPattern convfc_relu_pattern_wrapper(convfc_relu_pattern, "convfc_relu");

// ConvFc + bias
const OpTypePattern convfc_bias_pattern(
    {"BiasAdd|Add|AddV2",
      {
        convfc_pattern,
        bias_pattern, // bias node
      }
    });
DEFINE_GET_INPUT_NODES(ConvfcBias) {
  std::vector<const NodeDef*> input_nodes;
  input_nodes.push_back(&(match.inputs[0].inputs[0].node));
  return input_nodes;
}
DEFINE_GET_WEIGHTS_NODES(ConvfcBias) {
  std::vector<const NodeDef*> weights_nodes;
  weights_nodes.push_back(&(match.inputs[0].inputs[1].node));
  weights_nodes.push_back(&(match.inputs[1].node));
  return weights_nodes;
}
const ConvfcBiasPattern convfc_bias_pattern_wrapper(convfc_bias_pattern, "convfc_bias");

// ConvFc + bias + relu
const OpTypePattern convfc_bias_relu_pattern(
    {"Relu|Relu6", // relu node
      {
        convfc_bias_pattern,
      }
    });
DEFINE_GET_INPUT_NODES(ConvfcBiasRelu) {
  std::vector<const NodeDef*> input_nodes;
  input_nodes.push_back(&(match.inputs[0].inputs[0].inputs[0].node));
  return input_nodes;
}
DEFINE_GET_WEIGHTS_NODES(ConvfcBiasRelu) {
  std::vector<const NodeDef*> weights_nodes;
  weights_nodes.push_back(&(match.inputs[0].inputs[0].inputs[1].node));
  weights_nodes.push_back(&(match.inputs[0].inputs[1].node));
  return weights_nodes;
}
const ConvfcBiasReluPattern convfc_bias_relu_pattern_wrapper(convfc_bias_relu_pattern, "convfc_bias_relu");

// ConvFc + bias + Identity + relu
const OpTypePattern convfc_bias_id_relu_pattern(
    {"Relu|Relu6", // relu node
      {
        {"Identity",
          {
            convfc_bias_pattern,
          }
        },
      }
    });
DEFINE_GET_INPUT_NODES(ConvfcBiasIdRelu) {
  std::vector<const NodeDef*> input_nodes;
  input_nodes.push_back(&(match.inputs[0].inputs[0].inputs[0].inputs[0].node));
  return input_nodes;
}
DEFINE_GET_WEIGHTS_NODES(ConvfcBiasIdRelu) {
  std::vector<const NodeDef*> weights_nodes;
  weights_nodes.push_back(&(match.inputs[0].inputs[0].inputs[0].inputs[1].node));
  weights_nodes.push_back(&(match.inputs[0].inputs[0].inputs[1].node));
  return weights_nodes;
}
const ConvfcBiasIdReluPattern convfc_bias_id_relu_pattern_wrapper(convfc_bias_id_relu_pattern, "convfc_bias_id_relu");

// Conv2d_transpose
const OpTypePattern conv2d_transpose_pattern(
    {"Conv2DBackpropInput", // conv2d_transpose node
      {
        {"Pack",
          {
            {"StridedSlice",
              {
                {"Shape|Const"},
                {"Const"},
                {"Const"},
                {"Const"},
              }
            },
            {"Mul",
              {
                {"StridedSlice"},
                {"Const"},
              }
            },
            {"Mul",
              {
                {"StridedSlice"},
                {"Const"},
              }
            },
            {"Const"},
          }
        },
        weight_pattern, // filter node
        input_pattern, // input node
      }
    });
DEFINE_GET_INPUT_NODES(Conv2dTranspose) {
  std::vector<const NodeDef*> input_nodes;
  input_nodes.push_back(&(match.inputs[2].node));
  return input_nodes;
}
DEFINE_GET_WEIGHTS_NODES(Conv2dTranspose) {
  std::vector<const NodeDef*> weights_nodes;
  weights_nodes.push_back(&(match.inputs[1].node));
  return weights_nodes;
}
const Conv2dTransposePattern conv2d_transpose_pattern_wrapper(conv2d_transpose_pattern, "conv2d_transpose");

// Conv2d_transpose + bias
const OpTypePattern conv2d_transpose_bias_pattern(
    {"BiasAdd|Add|AddV2", // bias_add node
      {
        conv2d_transpose_pattern,
        bias_pattern, // bias node
      }
    });
DEFINE_GET_INPUT_NODES(Conv2dTransposeBias) {
  std::vector<const NodeDef*> input_nodes;
  input_nodes.push_back(&(match.inputs[0].inputs[2].node));
  return input_nodes;
}
DEFINE_GET_WEIGHTS_NODES(Conv2dTransposeBias) {
  std::vector<const NodeDef*> weights_nodes;
  weights_nodes.push_back(&(match.inputs[0].inputs[1].node));
  weights_nodes.push_back(&(match.inputs[1].node));
  return weights_nodes;
}
const Conv2dTransposeBiasPattern conv2d_transpose_bias_pattern_wrapper(conv2d_transpose_bias_pattern, "conv2d_transpose_bias");

// Conv2d_transpose + relu
const OpTypePattern conv2d_transpose_relu_pattern(
    {"Relu|Relu6", // relu_node
      {
        conv2d_transpose_pattern,
      }
    });
DEFINE_GET_INPUT_NODES(Conv2dTransposeRelu) {
  std::vector<const NodeDef*> input_nodes;
  input_nodes.push_back(&(match.inputs[0].inputs[2].node));
  return input_nodes;
}
DEFINE_GET_WEIGHTS_NODES(Conv2dTransposeRelu) {
  std::vector<const NodeDef*> weights_nodes;
  weights_nodes.push_back(&(match.inputs[0].inputs[1].node));
  return weights_nodes;
}
const Conv2dTransposeReluPattern conv2d_transpose_relu_pattern_wrapper(conv2d_transpose_relu_pattern, "conv2d_transpose_relu");

// Conv2d_transpose + bias + relu
const OpTypePattern conv2d_transpose_bias_relu_pattern(
    {"Relu|Relu6", // relu node
      {
        conv2d_transpose_bias_pattern,
      }
    });
DEFINE_GET_INPUT_NODES(Conv2dTransposeBiasRelu) {
  std::vector<const NodeDef*> input_nodes;
  input_nodes.push_back(&(match.inputs[0].inputs[0].inputs[2].node));
  return input_nodes;
}
DEFINE_GET_WEIGHTS_NODES(Conv2dTransposeBiasRelu) {
  std::vector<const NodeDef*> weights_nodes;
  weights_nodes.push_back(&(match.inputs[0].inputs[0].inputs[1].node));
  weights_nodes.push_back(&(match.inputs[0].inputs[1].node));
  return weights_nodes;
}
const Conv2dTransposeBiasReluPattern conv2d_transpose_bias_relu_pattern_wrapper(conv2d_transpose_bias_relu_pattern, "conv2d_transpose_bias_relu");

// keras.Conv2DTranspose
const OpTypePattern keras_conv2d_transpose_pattern(
    {"Conv2DBackpropInput", // conv2d_transpose node
      {
        {"Pack",
          {
            {"StridedSlice",
              {
                {"Shape|Const"},
                {"Const"},
                {"Const"},
                {"Const"},
              }
            },
            {"Add|AddV2",
              {
                {"Mul",
                  {
                    {"StridedSlice"},
                    {"Const"},
                  }
                },
                {"Const"},
              }
            },
            {"Add|AddV2",
              {
                {"Mul",
                  {
                    {"StridedSlice"},
                    {"Const"},
                  }
                },
                {"Const"},
              }
            },
            {"Const"},
          }
        },
        weight_pattern, // filter node
        input_pattern, // input node
      }
    });
DEFINE_GET_INPUT_NODES(KerasConv2dTranspose) {
  std::vector<const NodeDef*> input_nodes;
  input_nodes.push_back(&(match.inputs[2].node));
  return input_nodes;
}
DEFINE_GET_WEIGHTS_NODES(KerasConv2dTranspose) {
  std::vector<const NodeDef*> weights_nodes;
  weights_nodes.push_back(&(match.inputs[1].node));
  return weights_nodes;
}
const KerasConv2dTransposePattern keras_conv2d_transpose_pattern_wrapper(keras_conv2d_transpose_pattern, "keras_conv2d_transpose");

// Keras.Conv2DTranspose + bias
const OpTypePattern keras_conv2d_transpose_bias_pattern(
    {"BiasAdd|Add|AddV2", // bias_add node
      {
        keras_conv2d_transpose_pattern,
        bias_pattern, // bias node
      }
    });
DEFINE_GET_INPUT_NODES(KerasConv2dTransposeBias) {
  std::vector<const NodeDef*> input_nodes;
  input_nodes.push_back(&(match.inputs[0].inputs[2].node));
  return input_nodes;
}
DEFINE_GET_WEIGHTS_NODES(KerasConv2dTransposeBias) {
  std::vector<const NodeDef*> weights_nodes;
  weights_nodes.push_back(&(match.inputs[0].inputs[1].node));
  weights_nodes.push_back(&(match.inputs[1].node));
  return weights_nodes;
}
const KerasConv2dTransposeBiasPattern keras_conv2d_transpose_bias_pattern_wrapper(keras_conv2d_transpose_bias_pattern, "keras_conv2d_transpose_bias");

// Keras.Conv2DTranspose + relu
const OpTypePattern keras_conv2d_transpose_relu_pattern(
    {"Relu|Relu6", // relu node
      {
        keras_conv2d_transpose_pattern,
      }
    });
DEFINE_GET_INPUT_NODES(KerasConv2dTransposeRelu) {
  std::vector<const NodeDef*> input_nodes;
  input_nodes.push_back(&(match.inputs[0].inputs[2].node));
  return input_nodes;
}
DEFINE_GET_WEIGHTS_NODES(KerasConv2dTransposeRelu) {
  std::vector<const NodeDef*> weights_nodes;
  weights_nodes.push_back(&(match.inputs[0].inputs[1].node));
  return weights_nodes;
}
const KerasConv2dTransposeReluPattern keras_conv2d_transpose_relu_pattern_wrapper(keras_conv2d_transpose_relu_pattern, "keras_conv2d_transpose_relu");

// Keras.Conv2DTranspose + bias + relu
const OpTypePattern keras_conv2d_transpose_bias_relu_pattern(
    {"Relu|Relu6", // relu node
      {
        keras_conv2d_transpose_bias_pattern,
      }
    });
DEFINE_GET_INPUT_NODES(KerasConv2dTransposeBiasRelu) {
  std::vector<const NodeDef*> input_nodes;
  input_nodes.push_back(&(match.inputs[0].inputs[0].inputs[2].node));
  return input_nodes;
}
DEFINE_GET_WEIGHTS_NODES(KerasConv2dTransposeBiasRelu) {
  std::vector<const NodeDef*> weights_nodes;
  weights_nodes.push_back(&(match.inputs[0].inputs[0].inputs[1].node));
  weights_nodes.push_back(&(match.inputs[0].inputs[1].node));
  return weights_nodes;
}
const KerasConv2dTransposeBiasReluPattern keras_conv2d_transpose_bias_relu_pattern_wrapper(keras_conv2d_transpose_bias_relu_pattern, "keras_conv2d_transpose_bias_relu");

// Conv2d_backprop_input
const OpTypePattern conv2d_backprop_input_pattern(
    {"Conv2DBackpropInput", // conv2d_backprop_input node
      {
        {"Const"}, // output_shape node
        weight_pattern, // filter node
        input_pattern, // input node
      }
    });
DEFINE_GET_INPUT_NODES(Conv2dBackpropInput) {
  std::vector<const NodeDef*> input_nodes;
  input_nodes.push_back(&(match.inputs[2].node));
  return input_nodes;
}
DEFINE_GET_WEIGHTS_NODES(Conv2dBackpropInput) {
  std::vector<const NodeDef*> weights_nodes;
  weights_nodes.push_back(&(match.inputs[1].node));
  return weights_nodes;
}
const Conv2dBackpropInputPattern conv2d_backprop_input_pattern_wrapper(conv2d_backprop_input_pattern, "conv2d_backprop_input");

// Conv2d_backprop_input + bias
const OpTypePattern conv2d_backprop_input_bias_pattern(
    {"BiasAdd|Add|AddV2", // bias_add node
      {
        conv2d_backprop_input_pattern,
        bias_pattern, // bias node
      }
    });
DEFINE_GET_INPUT_NODES(Conv2dBackpropInputBias) {
  std::vector<const NodeDef*> input_nodes;
  input_nodes.push_back(&(match.inputs[0].inputs[2].node));
  return input_nodes;
}
DEFINE_GET_WEIGHTS_NODES(Conv2dBackpropInputBias) {
  std::vector<const NodeDef*> weights_nodes;
  weights_nodes.push_back(&(match.inputs[0].inputs[1].node));
  weights_nodes.push_back(&(match.inputs[1].node));
  return weights_nodes;
}
const Conv2dBackpropInputBiasPattern conv2d_backprop_input_bias_pattern_wrapper(conv2d_backprop_input_bias_pattern, "conv2d_backprop_input_bias");

// Conv2d_backprop_input + relu
const OpTypePattern conv2d_backprop_input_relu_pattern(
    {"Relu|Relu6", // relu node
      {
        conv2d_backprop_input_pattern,
      }
    });
DEFINE_GET_INPUT_NODES(Conv2dBackpropInputRelu) {
  std::vector<const NodeDef*> input_nodes;
  input_nodes.push_back(&(match.inputs[0].inputs[2].node));
  return input_nodes;
}
DEFINE_GET_WEIGHTS_NODES(Conv2dBackpropInputRelu) {
  std::vector<const NodeDef*> weights_nodes;
  weights_nodes.push_back(&(match.inputs[0].inputs[1].node));
  return weights_nodes;
}
const Conv2dBackpropInputReluPattern conv2d_backprop_input_relu_pattern_wrapper(conv2d_backprop_input_relu_pattern, "conv2d_backprop_input_relu");

// Conv2d_backprop_input + bias + Relu
const OpTypePattern conv2d_backprop_input_bias_relu_pattern(
    {"Relu|Relu6", // relu node
      {
        conv2d_backprop_input_bias_pattern,
      }
    });
DEFINE_GET_INPUT_NODES(Conv2dBackpropInputBiasRelu) {
  std::vector<const NodeDef*> input_nodes;
  input_nodes.push_back(&(match.inputs[0].inputs[0].inputs[2].node));
  return input_nodes;
}
DEFINE_GET_WEIGHTS_NODES(Conv2dBackpropInputBiasRelu) {
  std::vector<const NodeDef*> weights_nodes;
  weights_nodes.push_back(&(match.inputs[0].inputs[0].inputs[1].node));
  weights_nodes.push_back(&(match.inputs[0].inputs[1].node));
  return weights_nodes;
}
const Conv2dBackpropInputBiasReluPattern conv2d_backprop_input_bias_relu_pattern_wrapper(conv2d_backprop_input_bias_relu_pattern, "conv2d_backprop_input_bias_relu");

// LeakyRelu
// - nn.leaky_relu
const OpTypePattern leakyrelu_pattern(
    {"Maximum",
      {
        {"Mul",
          {
            {"Const"}, // alpha node
            input_pattern, // input node
          }
        },
        input_pattern, // input node
      }
    });
DEFINE_GET_INPUT_NODES(Leakyrelu) {
  std::vector<const NodeDef*> input_nodes;
  input_nodes.push_back(&(match.inputs[1].node));
  return input_nodes;
}
DEFINE_GET_WEIGHTS_NODES(Leakyrelu) {
  return std::vector<const NodeDef*> ();
}
const LeakyreluPattern leakyrelu_pattern_wrapper(leakyrelu_pattern, "leakyrelu");

// - tf1.15.fused_leaky_relu
const OpTypePattern fused_leakyrelu_pattern(
    {"LeakyRelu",
      {
        input_pattern, // input node
      }
    });
DEFINE_GET_INPUT_NODES(FusedLeakyrelu) {
  std::vector<const NodeDef*> input_nodes;
  input_nodes.push_back(&(match.inputs[0].node));
  return input_nodes;
}
DEFINE_GET_WEIGHTS_NODES(FusedLeakyrelu) {
  return std::vector<const NodeDef*> ();
}
const FusedLeakyreluPattern fused_leakyrelu_pattern_wrapper(fused_leakyrelu_pattern, "fused_leakyrelu");

// - conv + tf1.15.fused_leaky_relu
const OpTypePattern convfc_fused_leakyrelu_pattern(
    {"LeakyRelu",
      {
        convfc_pattern, // input node
      }
    });
DEFINE_GET_INPUT_NODES(ConvfcFusedLeakyrelu) {
  std::vector<const NodeDef*> input_nodes;
  input_nodes.push_back(&(match.inputs[0].inputs[0].node));
  return input_nodes;
}
DEFINE_GET_WEIGHTS_NODES(ConvfcFusedLeakyrelu) {
  std::vector<const NodeDef*> weights_nodes;
  weights_nodes.push_back(&(match.inputs[0].inputs[1].node));
  return weights_nodes;
}
const ConvfcFusedLeakyreluPattern convfc_fused_leakyrelu_pattern_wrapper(convfc_fused_leakyrelu_pattern, "convfc_fused_leakyrelu");

// - conv + bias + tf1.15.fused_leaky_relu
const OpTypePattern convfc_bias_fused_leakyrelu_pattern(
    {"LeakyRelu",
      {
        convfc_bias_pattern, // input node
      }
    });
DEFINE_GET_INPUT_NODES(ConvfcBiasFusedLeakyrelu) {
  std::vector<const NodeDef*> input_nodes;
  input_nodes.push_back(&(match.inputs[0].inputs[0].inputs[0].node));
  return input_nodes;
}
DEFINE_GET_WEIGHTS_NODES(ConvfcBiasFusedLeakyrelu) {
  std::vector<const NodeDef*> weights_nodes;
  weights_nodes.push_back(&(match.inputs[0].inputs[0].inputs[1].node));
  weights_nodes.push_back(&(match.inputs[0].inputs[1].node));
  return weights_nodes;
}
const ConvfcBiasFusedLeakyreluPattern convfc_bias_fused_leakyrelu_pattern_wrapper(convfc_bias_fused_leakyrelu_pattern, "convfc_bias_fused_leakyrelu");

// - conv +  nn.leaky_relu
const OpTypePattern convfc_leakyrelu_pattern(
    {"Maximum",
      {
        {"Mul",
          {
            {"Const"}, // alpha node
            convfc_pattern, // input node
          }
        },
        convfc_pattern, // input node
      }
    });
DEFINE_GET_INPUT_NODES(ConvfcLeakyrelu) {
  std::vector<const NodeDef*> input_nodes;
  input_nodes.push_back(&(match.inputs[1].inputs[0].node));
  return input_nodes;
}
DEFINE_GET_WEIGHTS_NODES(ConvfcLeakyrelu) {
  std::vector<const NodeDef*> weights_nodes;
  weights_nodes.push_back(&(match.inputs[1].inputs[1].node));
  return weights_nodes;
}
const ConvfcLeakyreluPattern convfc_leakyrelu_pattern_wrapper(convfc_leakyrelu_pattern, "convfc_leakyrelu");

// - conv + bias + nn.leaky_relu
const OpTypePattern convfc_bias_leakyrelu_pattern(
    {"Maximum",
      {
        {"Mul",
          {
            {"Const"}, // alpha node;
            convfc_bias_pattern, // input node
          }
        },
        convfc_bias_pattern, // input node
      }
    });
DEFINE_GET_INPUT_NODES(ConvfcBiasLeakyrelu) {
  std::vector<const NodeDef*> input_nodes;
  input_nodes.push_back(&(match.inputs[1].inputs[0].inputs[0].node));
  return input_nodes;
}
DEFINE_GET_WEIGHTS_NODES(ConvfcBiasLeakyrelu) {
  std::vector<const NodeDef*> weights_nodes;
  weights_nodes.push_back(&(match.inputs[1].inputs[0].inputs[1].node));
  weights_nodes.push_back(&(match.inputs[1].inputs[1].node));
  return weights_nodes;
}
const ConvfcBiasLeakyreluPattern convfc_bias_leakyrelu_pattern_wrapper(convfc_bias_leakyrelu_pattern, "convfc_bias_leakyrelu");

// swish
const OpTypePattern swish_pattern(
    {"Mul",
      {
        input_pattern, // input node
        {"Sigmoid",
          {
            input_pattern, // input node
          }
        },
      }
    });
DEFINE_GET_INPUT_NODES(Swish) {
  std::vector<const NodeDef*> input_nodes;
  input_nodes.push_back(&(match.inputs[0].node));
  return input_nodes;
}
DEFINE_GET_WEIGHTS_NODES(Swish) {
  return std::vector<const NodeDef*> ();
}
const SwishPattern swish_pattern_wrapper(swish_pattern, "swish");

// Squeeze Excite gating function, hard-sigmoid
const OpTypePattern hard_sigmoid_pattern(
    {"Mul",
      {
        {"Mul|RealDiv",
          {
           {"Relu6",
             {
               {"Add|AddV2",
                 {
                   input_pattern, // input node
                   {"Const"},    // add 3
                 }
               },
             }
           },
           {"Const"}, // scale=1/6
          }
        },
        {"Const"}, // vitis dpu scale 6*2371/2^14
      }
    });
DEFINE_GET_INPUT_NODES(HardSigmoid) {
  std::vector<const NodeDef*> input_nodes;
  input_nodes.push_back(&(match.inputs[0].inputs[0].inputs[0].inputs[0].node));
  return input_nodes;
}
DEFINE_GET_WEIGHTS_NODES(HardSigmoid) {
  return std::vector<const NodeDef*> ();
}
const HardSigmoidPattern hard_sigmoid_pattern_wrapper(hard_sigmoid_pattern, "hard_sigmoid");

// hard swish replace sigmoid_with hard_sigmoid
const OpTypePattern hard_swish_pattern(
    {"Mul",
      {
        input_pattern, // input node
        hard_sigmoid_pattern,
      }
    });
DEFINE_GET_INPUT_NODES(HardSwish) {
  std::vector<const NodeDef*> input_nodes;
  input_nodes.push_back(&(match.inputs[0].node));
  return input_nodes;
}
DEFINE_GET_WEIGHTS_NODES(HardSwish) {
  return std::vector<const NodeDef*> ();
}
const HardSwishPattern hard_swish_pattern_wrapper(hard_swish_pattern, "hard_swish");

const OpTypePattern depth_to_space_pattern(
    {"DepthToSpace"});
DEFINE_GET_INPUT_NODES(DepthToSpace) {
  return std::vector<const NodeDef*> ();
}
DEFINE_GET_WEIGHTS_NODES(DepthToSpace) {
  return std::vector<const NodeDef*> ();
}
const OtherPattern depth_to_space_pattern_wrapper(depth_to_space_pattern, "depth_to_space");

// - tf.keras.LeakyRelu
const OpTypePattern keras_leakyrelu_pattern(
    {"Sub",
      {
        {"Relu",
          {
            input_pattern, // input node
          }
        },
        {"Mul",
          {
            {"Const"}, // alpha node
            {"Relu",
              {
                {"Neg"},
              }
            },
          }
        },
      }
    });
DEFINE_GET_INPUT_NODES(KerasLeakyrelu) {
  std::vector<const NodeDef*> input_nodes;
  input_nodes.push_back(&(match.inputs[0].inputs[0].node));
  return input_nodes;
}
DEFINE_GET_WEIGHTS_NODES(KerasLeakyrelu) {
  return std::vector<const NodeDef*> ();
}
const KerasLeakyreluPattern keras_leakyrelu_pattern_wrapper(keras_leakyrelu_pattern, "keras_leakyrelu");

// - conv + tf.keras.LeakyRelu
const OpTypePattern convfc_keras_leakyrelu_pattern(
    {"Sub",
      {
        {"Relu",
          {
            convfc_pattern,
          }
        },
        {"Mul",
          {
            {"Const"}, // alpha node
            {"Relu",
              {
                {"Neg"},
              }
            },
          }
        },
      }
    });
DEFINE_GET_INPUT_NODES(ConvfcKerasLeakyrelu) {
  std::vector<const NodeDef*> input_nodes;
  input_nodes.push_back(&(match.inputs[0].inputs[0].inputs[0].node));
  return input_nodes;
}
DEFINE_GET_WEIGHTS_NODES(ConvfcKerasLeakyrelu) {
  std::vector<const NodeDef*> weights_nodes;
  weights_nodes.push_back(&(match.inputs[0].inputs[0].inputs[1].node));
  return weights_nodes;
}
const ConvfcKerasLeakyreluPattern convfc_keras_leakyrelu_pattern_wrapper(convfc_keras_leakyrelu_pattern, "convfc_keras_leakyrelu");

// - conv + bias + tf.keras.LeakyRelu
const OpTypePattern convfc_bias_keras_leakyrelu_pattern(
    {"Sub",
      {
        {"Relu",
          {
            convfc_bias_pattern,
          }
        },
        {"Mul",
          {
            {"Const"}, // alpha node
            {"Relu",
              {
                {"Neg"},
              }
            },
          }
        },
      }
    });
DEFINE_GET_INPUT_NODES(ConvfcBiasKerasLeakyrelu) {
  std::vector<const NodeDef*> input_nodes;
  input_nodes.push_back(&(match.inputs[0].inputs[0].inputs[0].inputs[0].node));
  return input_nodes;
}
DEFINE_GET_WEIGHTS_NODES(ConvfcBiasKerasLeakyrelu) {
  std::vector<const NodeDef*> weights_nodes;
  weights_nodes.push_back(&(match.inputs[0].inputs[0].inputs[0].inputs[1].node));
  weights_nodes.push_back(&(match.inputs[0].inputs[0].inputs[1].node));
  return weights_nodes;
}
const ConvfcBiasKerasLeakyreluPattern convfc_bias_keras_leakyrelu_pattern_wrapper(convfc_bias_keras_leakyrelu_pattern, "convfc_bias_keras_leakyrelu");

// Upsampling
// - tf.keras.layers.Upsampling2D
const OpTypePattern upsampling_pattern(
    {"ResizeBilinear|ResizeNearestNeighbor", // resize node
      {
        input_pattern, // input node
        {"Mul", // mul node
          {
            {"StridedSlice",
              {
                {"Shape"},
                {"Const"},
                {"Const"},
                {"Const"},
              }
            },
            {"Const"}, // resize value node
          }
        },
      }
    });
DEFINE_GET_INPUT_NODES(Upsampling) {
  std::vector<const NodeDef*> input_nodes;
  input_nodes.push_back(&(match.inputs[0].node));
  return input_nodes;
}
DEFINE_GET_WEIGHTS_NODES(Upsampling) {
  return std::vector<const NodeDef*> ();
}
const UpsamplingPattern upsampling_pattern_wrapper(upsampling_pattern, "upsampling");

// - ResizeBilinear
const OpTypePattern resize_pattern(
    {"ResizeBilinear|ResizeNearestNeighbor",
      {
        input_pattern, // input node
        {"*"}, // size node
      }
    });
DEFINE_GET_INPUT_NODES(Resize) {
  std::vector<const NodeDef*> input_nodes;
  input_nodes.push_back(&(match.inputs[0].node));
  return input_nodes;
}
DEFINE_GET_WEIGHTS_NODES(Resize) {
  return std::vector<const NodeDef*> ();
}
const ResizePattern resize_pattern_wrapper(resize_pattern, "resize_bilinear");

// nearest_neighbor_upsampling implementation
// - this implementation only uses reshape and broadcasting to make it TPU compatible,
// from https://github.com/tensorflow/models/tree/master/research/object_detection/utils/ops.py
const OpTypePattern tpu_nearest_neighbor_upsampling_pattern(
    {"Reshape", // output_reshape node
      {
        {"Mul",
          {
            {"Reshape",  // input_reshape node
              {
                input_pattern,  // input node
                {"Const|Pack"},  // input_shape node
              }
            },
            {"Const"},  // resize_value node
          }
        },
        {"Const|Pack"},  // output_shape node
      }
    });
DEFINE_GET_INPUT_NODES(TpuNearestNeighborUpsampling) {
  std::vector<const NodeDef*> input_nodes;
  input_nodes.push_back(&(match.inputs[0].inputs[0].inputs[0].node));
  return input_nodes;
}
DEFINE_GET_WEIGHTS_NODES(TpuNearestNeighborUpsampling) {
  return std::vector<const NodeDef*> ();
}
const TpuNearestNeighborUpsamplingPattern tpu_nearest_neighbor_upsampling_pattern_wrapper(tpu_nearest_neighbor_upsampling_pattern, "tpu_nearest_neighbor_upsampling");

// BatchNorm
const OpTypePattern batchnorm_pattern(
    {"Add|AddV2", // add node
      {
        {"Mul", // mul node
          {
            input_pattern, // input node
            {"Const"}, // scale node
          }
        },
        {"Const"}, // offset node
      }
    });
DEFINE_GET_INPUT_NODES(Batchnorm) {
  std::vector<const NodeDef*> input_nodes;
  input_nodes.push_back(&(match.inputs[0].inputs[0].node));
  return input_nodes;
}
DEFINE_GET_WEIGHTS_NODES(Batchnorm) {
  return std::vector<const NodeDef*> ();
}
const BatchnormPattern batchnorm_pattern_wrapper(batchnorm_pattern, "batchnorm");

// BatchNorm + relu
const OpTypePattern batchnorm_relu_pattern(
    {"Relu|Relu6", // relu node
      {
        batchnorm_pattern,
      }
    });
DEFINE_GET_INPUT_NODES(BatchnormRelu) {
  std::vector<const NodeDef*> input_nodes;
  input_nodes.push_back(&(match.inputs[0].inputs[0].inputs[0].node));
  return input_nodes;
}
DEFINE_GET_WEIGHTS_NODES(BatchnormRelu) {
  return std::vector<const NodeDef*> ();
}
const BatchnormReluPattern batchnorm_relu_pattern_wrapper(batchnorm_relu_pattern, "batchnorm_relu");

// Array
const OpTypePattern array_pattern(
    // Mul for hard-swish 2-nd mul
    {"Add|AddV2|Mul",
      {
        {"*"}, // input node 1
        {"*"}, // input node 2
      }
    });
DEFINE_GET_INPUT_NODES(Array) {
  std::vector<const NodeDef*> input_nodes;
  input_nodes.push_back(&(match.inputs[0].node));
  input_nodes.push_back(&(match.inputs[1].node));
  return input_nodes;
}
DEFINE_GET_WEIGHTS_NODES(Array) {
  return std::vector<const NodeDef*> ();
}
const ArrayPattern array_pattern_wrapper(array_pattern, "array");

// Array + relu
const OpTypePattern array_relu_pattern(
    {"Relu|Relu6",
      {
        array_pattern,
      }
    });
DEFINE_GET_INPUT_NODES(ArrayRelu) {
  std::vector<const NodeDef*> input_nodes;
  input_nodes.push_back(&(match.inputs[0].inputs[0].node));
  input_nodes.push_back(&(match.inputs[0].inputs[1].node));
  return input_nodes;
}
DEFINE_GET_WEIGHTS_NODES(ArrayRelu) {
  return std::vector<const NodeDef*> ();
}
const ArrayReluPattern array_relu_pattern_wrapper(array_relu_pattern, "array_relu");

// mul_v1
const OpTypePattern mul_v1_pattern(
    {"Mul",
      {
        {"*"}, // input node 1
        {"Const"}, // constant scale
      }
    });
DEFINE_GET_INPUT_NODES(Mul_v1) {
  std::vector<const NodeDef*> input_nodes;
  input_nodes.push_back(&(match.inputs[0].node));
  return input_nodes;
}
DEFINE_GET_WEIGHTS_NODES(Mul_v1) {
  return std::vector<const NodeDef*> ();
}
const Mul_v1Pattern mul_v1_pattern_wrapper(mul_v1_pattern, "mul_v1");

// Mul_v2
const OpTypePattern mul_v2_pattern(
    {"Mul",
      {
        {"Const"}, // constant scale
        {"*"}, // input node 2
      }
    });
DEFINE_GET_INPUT_NODES(Mul_v2) {
  std::vector<const NodeDef*> input_nodes;
  input_nodes.push_back(&(match.inputs[1].node));
  return input_nodes;
}
DEFINE_GET_WEIGHTS_NODES(Mul_v2) {
  return std::vector<const NodeDef*> ();
}
const Mul_v2Pattern mul_v2_pattern_wrapper(mul_v2_pattern, "mul_v2");

// AvgPool + scale
const OpTypePattern avgpool_mul_pattern(
    {"Mul",
      {
        {"AvgPool"},
        {"Const"},
      }
    });
DEFINE_GET_INPUT_NODES(AvgpoolMul) {
  return std::vector<const NodeDef*> ();
}
DEFINE_GET_WEIGHTS_NODES(AvgpoolMul) {
  return std::vector<const NodeDef*> ();
}
const AvgpoolMulPattern avgpool_mul_pattern_wrapper(avgpool_mul_pattern, "avgpool_mul");

// tf.clip_by_value
const OpTypePattern clip_by_value_pattern(
    {"Maximum",
      {
        {"Minimum",
          {
            {"*"}, // input node
            {"*"}, // max value
          }
        },
        {"*"}, // min value
      }
    });
DEFINE_GET_INPUT_NODES(ClipByValue) {
  std::vector<const NodeDef*> input_nodes;
  input_nodes.push_back(&(match.inputs[0].inputs[0].node));
  return input_nodes;
}
DEFINE_GET_WEIGHTS_NODES(ClipByValue) {
  return std::vector<const NodeDef*> ();
}
const ClipByValuePattern clip_by_value_pattern_wrapper(clip_by_value_pattern, "clip_by_value");

// Other
const OpTypePattern other_pattern(
    {"Max|AvgPool|MaxPool|Mean|Pad|MirrorPad|Transpose|Concat|ConcatV2|Squeeze|Reshape|ExpandDims|Relu|Relu6|AddN"});
DEFINE_GET_INPUT_NODES(Other) {
  return std::vector<const NodeDef*> ();
}
DEFINE_GET_WEIGHTS_NODES(Other) {
  return std::vector<const NodeDef*> ();
}
const OtherPattern other_pattern_wrapper(other_pattern, "other");

// Other + relu
const OpTypePattern other_relu_pattern(
    {"Relu|Relu6",
      {
        other_pattern,
      }
    });
DEFINE_GET_INPUT_NODES(OtherRelu) {
  return std::vector<const NodeDef*> ();
}
DEFINE_GET_WEIGHTS_NODES(OtherRelu) {
  return std::vector<const NodeDef*> ();
}
const OtherReluPattern other_relu_pattern_wrapper(other_relu_pattern, "other_relu");


// Known patterns
// The quantization will perform in the order of this vector, the previously matched
// pattern will not be matched again. Watch out for the order
const std::vector<const OpTypePatternBase*> known_patterns ({
  &placeholder_pattern_wrapper,
  &atrous_conv_bias_relu_pattern_wrapper,
  &atrous_conv_bias_pattern_wrapper,
  &atrous_conv_relu_pattern_wrapper,
  &atrous_conv_pattern_wrapper,
  &convfc_bias_leakyrelu_pattern_wrapper,
  &convfc_bias_fused_leakyrelu_pattern_wrapper,
  &convfc_bias_keras_leakyrelu_pattern_wrapper,
  &convfc_leakyrelu_pattern_wrapper,
  &convfc_fused_leakyrelu_pattern_wrapper,
  &convfc_keras_leakyrelu_pattern_wrapper,
  &leakyrelu_pattern_wrapper,
  &fused_leakyrelu_pattern_wrapper,
  &keras_leakyrelu_pattern_wrapper,
  &convfc_bias_id_relu_pattern_wrapper,
  &convfc_bias_relu_pattern_wrapper,
  &convfc_bias_pattern_wrapper,
  &convfc_relu_pattern_wrapper,
  &convfc_pattern_wrapper,
  &conv2d_transpose_bias_relu_pattern_wrapper,
  &conv2d_transpose_bias_pattern_wrapper,
  &conv2d_transpose_relu_pattern_wrapper,
  &conv2d_transpose_pattern_wrapper,
  &keras_conv2d_transpose_bias_relu_pattern_wrapper,
  &keras_conv2d_transpose_bias_pattern_wrapper,
  &keras_conv2d_transpose_relu_pattern_wrapper,
  &keras_conv2d_transpose_pattern_wrapper,
  &conv2d_backprop_input_bias_relu_pattern_wrapper,
  &conv2d_backprop_input_bias_pattern_wrapper,
  &conv2d_backprop_input_relu_pattern_wrapper,
  &conv2d_backprop_input_pattern_wrapper,
  &depth_to_space_pattern_wrapper,
  &upsampling_pattern_wrapper,
  &resize_pattern_wrapper,
  &hard_swish_pattern_wrapper,
  &hard_sigmoid_pattern_wrapper,
  &swish_pattern_wrapper,
  &tpu_nearest_neighbor_upsampling_pattern_wrapper,
  &batchnorm_relu_pattern_wrapper,
  &batchnorm_pattern_wrapper,
  &avgpool_mul_pattern_wrapper,
  &array_relu_pattern_wrapper,
  &mul_v1_pattern_wrapper,
  &mul_v2_pattern_wrapper,
  &array_pattern_wrapper,
  &clip_by_value_pattern_wrapper,
  &other_relu_pattern_wrapper,
  &other_pattern_wrapper
  });

//// ignore patterns:
// FusedBatchNorm + ConvFc
const OpTypePattern convfc_fusedbn_pattern(
    {"FusedBatchNorm|FusedBatchNormV2|FusedBatchNormV3",                // batchnorm
      {
        {"Conv2D|MatMul|DepthwiseConv2dNative"},  // conv_node
        {"*"},  // beta_node
        {"*"},  // gamma_node
        {"*"},  // mean_node
        {"*"},  // variance_node
      }
    });

// FusedBatchNorm + identity + ConvFc
const OpTypePattern convfc_id_fusedbn_pattern(
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
    });

// Sub + Mul + AssignSub,  from ssd mobilenet v1 fpn batch norm, there is a control
// dependence to a identity op, so can not strip this during partition graph
const OpTypePattern assign_mul_sub_pattern(
    {"AssignSub",                // batchnorm
      {
        {"*"},  // moving mean
        {"Mul",
          {
            {"Sub"},
            {"*"}, // decay
          }
        },
      }
    });

const std::vector<std::tuple<const string, const OpTypePattern>> known_ignore_patterns({
  std::make_tuple("convfc_fusedbn", convfc_fusedbn_pattern),
  std::make_tuple("convfc_id_fusedbn", convfc_id_fusedbn_pattern),
  std::make_tuple("assgin_mul_sub", assign_mul_sub_pattern),
  });

const std::set<string> compute_patterns{
    convfc_pattern_wrapper.GetName(),
    convfc_relu_pattern_wrapper.GetName(),
    convfc_bias_pattern_wrapper.GetName(),
    convfc_bias_relu_pattern_wrapper.GetName(),
    convfc_bias_id_relu_pattern_wrapper.GetName(),
    atrous_conv_pattern_wrapper.GetName(),
    atrous_conv_relu_pattern_wrapper.GetName(),
    atrous_conv_bias_pattern_wrapper.GetName(),
    atrous_conv_bias_relu_pattern_wrapper.GetName(),
    convfc_leakyrelu_pattern_wrapper.GetName(),
    convfc_fused_leakyrelu_pattern_wrapper.GetName(),
    convfc_keras_leakyrelu_pattern_wrapper.GetName(),
    convfc_bias_leakyrelu_pattern_wrapper.GetName(),
    convfc_bias_fused_leakyrelu_pattern_wrapper.GetName(),
    convfc_bias_keras_leakyrelu_pattern_wrapper.GetName(),
    conv2d_transpose_pattern_wrapper.GetName(),
    conv2d_transpose_relu_pattern_wrapper.GetName(),
    conv2d_transpose_bias_pattern_wrapper.GetName(),
    conv2d_transpose_bias_relu_pattern_wrapper.GetName(),
    keras_conv2d_transpose_pattern_wrapper.GetName(),
    keras_conv2d_transpose_relu_pattern_wrapper.GetName(),
    keras_conv2d_transpose_bias_pattern_wrapper.GetName(),
    keras_conv2d_transpose_bias_relu_pattern_wrapper.GetName(),
    conv2d_backprop_input_pattern_wrapper.GetName(),
    conv2d_backprop_input_relu_pattern_wrapper.GetName(),
    conv2d_backprop_input_bias_pattern_wrapper.GetName(),
    conv2d_backprop_input_bias_relu_pattern_wrapper.GetName(),
};

const string get_pattern_name_from_id(const int pattern_id) {
  if (pattern_id < 0 || pattern_id > known_patterns.size()) {
    LOG(FATAL) << "Invalid pattern id: " << pattern_id;
  }
  return known_patterns[pattern_id]->GetName();
}

const string get_ignore_pattern_name_from_id(const int pattern_id) {
    if (pattern_id < 0 || pattern_id > known_ignore_patterns.size()) {
    LOG(FATAL) << "Invalid pattern id: " << pattern_id;
  }
  return std::get<0>(known_ignore_patterns[pattern_id]);
}
// clang-format on

std::vector<const NodeDef *> get_input_nodes(const NodeMatch &match,
                                             const string &pattern_name) {
  std::vector<const NodeDef *> input_nodes;
  for (auto it = known_patterns.begin(); it != known_patterns.end(); ++it) {
    if ((*it)->GetName() == pattern_name) {
      return (*it)->GetInputNodes(match);
    }
  }
  LOG(FATAL) << "Unknown pattern_name: " << pattern_name;
  return input_nodes;
}

std::vector<const NodeDef *> get_ignore_nodes(const NodeMatch &match,
                                              const string &pattern_name) {
  std::vector<const NodeDef *> ignore_nodes;
  if (pattern_name == "convfc_fusedbn") {
    ignore_nodes.push_back(&(match.inputs[0].node));
  } else if (pattern_name == "convfc_id_fusedbn") {
    ignore_nodes.push_back(&(match.inputs[0].inputs[0].node));
  } else if (pattern_name == "assgin_mul_sub") {
    ignore_nodes.push_back(&(match.inputs[1].node));
  } else {
    LOG(FATAL) << "Unknown pattern_name: " << pattern_name;
  }
  return ignore_nodes;
}

std::vector<const NodeDef *> get_weights_nodes(const NodeMatch &match,
                                               const string &pattern_name) {
  std::vector<const NodeDef *> weights_nodes;
  for (auto it = known_patterns.begin(); it != known_patterns.end(); ++it) {
    if ((*it)->GetName() == pattern_name) {
      return (*it)->GetWeightsNodes(match);
    }
  }
  LOG(FATAL) << "Unknown pattern_name: " << pattern_name;
  return weights_nodes;
}

} // namespace decent_q
} // namespace tensorflow
