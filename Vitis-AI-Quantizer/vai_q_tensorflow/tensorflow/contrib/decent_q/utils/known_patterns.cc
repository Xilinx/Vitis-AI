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

#include "tensorflow/contrib/decent_q/utils/known_patterns.h"

namespace tensorflow {
namespace decent_q {

// clang-format off

// input pattern
const OpTypePattern input_pattern({"*"});

// Weight pattern
const OpTypePattern weight_pattern({"Const|Identity|ReadVariableOp|Mul"});

// Weight pattern
const OpTypePattern bias_pattern({"Const|Identity|ReadVariableOp|Sub"});

// Placeholder
const OpTypePattern placeholder_pattern({"Placeholder"});

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

// AtrousConv + bias
const OpTypePattern atrous_conv_bias_pattern(
    {"BiasAdd|Add|AddV2",
      {
        atrous_conv_pattern,
        bias_pattern, // bias node
      }
    });

// AtrousConv + relu
const OpTypePattern atrous_conv_relu_pattern(
    {"Relu|Relu6",
      {
        atrous_conv_pattern,
      }
    });

// AtrousConv
const OpTypePattern atrous_conv_bias_relu_pattern(
    {"Relu|Relu6",
      {
        atrous_conv_bias_pattern,
      }
    });

// ConvFc
const OpTypePattern convfc_pattern(
    {"Conv2D|DepthwiseConv2d|DepthwiseConv2dNative|MatMul|Conv3D",
      {
        input_pattern, // input node
        weight_pattern, // weight node
      }
    });

// ConvFc + relu
const OpTypePattern convfc_relu_pattern(
    {"Relu|Relu6",
      {
        convfc_pattern,
      }
    });

// ConvFc + bias
const OpTypePattern convfc_bias_pattern(
    {"BiasAdd|Add|AddV2",
      {
        convfc_pattern,
        bias_pattern, // bias node
      }
    });

// ConvFc + bias + relu
const OpTypePattern convfc_bias_relu_pattern(
    {"Relu|Relu6", // relu node
      {
        convfc_bias_pattern,
      }
    });

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

// Conv2d_transpose + bias
const OpTypePattern conv2d_transpose_bias_pattern(
    {"BiasAdd|Add|AddV2", // bias_add node
      {
        conv2d_transpose_pattern,
        bias_pattern, // bias node
      }
    });

// Conv2d_transpose + relu
const OpTypePattern conv2d_transpose_relu_pattern(
    {"Relu|Relu6", // relu_node
      {
        conv2d_transpose_pattern,
      }
    });

// Conv2d_transpose + bias + relu
const OpTypePattern conv2d_transpose_bias_relu_pattern(
    {"Relu|Relu6", // relu node
      {
        conv2d_transpose_bias_pattern,
      }
    });

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

// Keras.Conv2DTranspose + bias
const OpTypePattern keras_conv2d_transpose_bias_pattern(
    {"BiasAdd|Add|AddV2", // bias_add node
      {
        keras_conv2d_transpose_pattern,
        bias_pattern, // bias node
      }
    });

// Keras.Conv2DTranspose + relu
const OpTypePattern keras_conv2d_transpose_relu_pattern(
    {"Relu|Relu6", // relu node
      {
        keras_conv2d_transpose_pattern,
      }
    });

// Keras.Conv2DTranspose + bias + relu
const OpTypePattern keras_conv2d_transpose_bias_relu_pattern(
    {"Relu|Relu6", // relu node
      {
        keras_conv2d_transpose_bias_pattern,
      }
    });

// Conv2d_backprop_input
const OpTypePattern conv2d_backprop_input_pattern(
    {"Conv2DBackpropInput", // conv2d_backprop_input node
      {
        {"Const"}, // output_shape node
        weight_pattern, // filter node
        input_pattern, // input node
      }
    });

// Conv2d_backprop_input + bias
const OpTypePattern conv2d_backprop_input_bias_pattern(
    {"BiasAdd|Add|AddV2", // bias_add node
      {
        conv2d_backprop_input_pattern,
        bias_pattern, // bias node
      }
    });

// Conv2d_backprop_input + relu
const OpTypePattern conv2d_backprop_input_relu_pattern(
    {"Relu|Relu6", // relu node
      {
        conv2d_backprop_input_pattern,
      }
    });

// Conv2d_backprop_input + bias + Relu
const OpTypePattern conv2d_backprop_input_bias_relu_pattern(
    {"Relu|Relu6", // relu node
      {
        conv2d_backprop_input_bias_pattern,
      }
    });

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

// - tf1.15.fused_leaky_relu
const OpTypePattern fused_leakyrelu_pattern(
    {"LeakyRelu",
      {
        input_pattern, // input node
      }
    });

// - conv + tf1.15.fused_leaky_relu
const OpTypePattern convfc_fused_leakyrelu_pattern(
    {"LeakyRelu",
      {
        convfc_pattern, // input node
      }
    });

// - conv + bias + tf1.15.fused_leaky_relu
const OpTypePattern convfc_bias_fused_leakyrelu_pattern(
    {"LeakyRelu",
      {
        convfc_bias_pattern, // input node
      }
    });

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

// - conv + bias + nn.leaky_relu
const OpTypePattern convfc_bias_leakyrelu_pattern(
    {"Maximum",
      {
        {"Mul",
          {
            {"Const"}, // alpha node
            convfc_bias_pattern, // input node
          }
        },
        convfc_bias_pattern, // input node
      }
    });

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

// - ResizeBilinear
const OpTypePattern resize_pattern(
    {"ResizeBilinear|ResizeNearestNeighbor",
      {
        input_pattern, // input node
        {"*"}, // size node
      }
    });

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
                {"Const"},  // input_shape node
              }
            },
            {"Const"},  // resize_value node
          }
        },
        {"Const"},  // output_shape node
      }
    });

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

// BatchNorm + relu
const OpTypePattern batchnorm_relu_pattern(
    {"Relu|Relu6", // relu node
      {
        batchnorm_pattern,
      }
    });

// Array
const OpTypePattern array_pattern(
    {"Add|AddV2",
      {
        {"*"}, // input node 1
        {"*"}, // input node 2
      }
    });

// Array + relu
const OpTypePattern array_relu_pattern(
    {"Relu|Relu6",
      {
        array_pattern,
      }
    });

// AvgPool + scale
const OpTypePattern avgpool_mul_pattern(
    {"Mul",
      {
        {"AvgPool",
          {
            {"*"},
          }
        },
        {"Const"},
      }
    });

// Other
const OpTypePattern other_pattern(
    {"AvgPool|MaxPool|Mean|Pad|Concat|ConcatV2|Squeeze|Reshape|ExpandDims|Relu|Relu6|AddN|AddV2"});

// Other + relu
const OpTypePattern other_relu_pattern(
    {"Relu|Relu6",
      {
        other_pattern,
      }
    });

// Known patterns
// The quantization will perform in the order of this vector, the previously matched
// pattern will not be matched again. Watch out for the order.
const std::vector<std::tuple<const string, const OpTypePattern>> known_patterns({
  std::make_tuple("placeholder", placeholder_pattern),
  std::make_tuple("atrous_conv_bias_relu", atrous_conv_bias_relu_pattern),
  std::make_tuple("atrous_conv_bias", atrous_conv_bias_pattern),
  std::make_tuple("atrous_conv_relu", atrous_conv_relu_pattern),
  std::make_tuple("atrous_conv", atrous_conv_pattern),
  std::make_tuple("convfc_bias_leakyrelu", convfc_bias_leakyrelu_pattern),
  std::make_tuple("convfc_bias_fused_leakyrelu", convfc_bias_fused_leakyrelu_pattern),
  std::make_tuple("convfc_bias_keras_leakyrelu", convfc_bias_keras_leakyrelu_pattern),
  std::make_tuple("convfc_leakyrelu", convfc_leakyrelu_pattern),
  std::make_tuple("convfc_fused_leakyrelu", convfc_fused_leakyrelu_pattern),
  std::make_tuple("convfc_keras_leakyrelu", convfc_keras_leakyrelu_pattern),
  std::make_tuple("leakyrelu", leakyrelu_pattern),
  std::make_tuple("fused_leakyrelu", fused_leakyrelu_pattern),
  std::make_tuple("keras_leakyrelu", keras_leakyrelu_pattern),
  std::make_tuple("convfc_bias_id_relu", convfc_bias_id_relu_pattern),
  std::make_tuple("convfc_bias_relu", convfc_bias_relu_pattern),
  std::make_tuple("convfc_bias", convfc_bias_pattern),
  std::make_tuple("convfc_relu", convfc_relu_pattern),
  std::make_tuple("convfc", convfc_pattern),
  std::make_tuple("conv2d_transpose_bias_relu", conv2d_transpose_bias_relu_pattern),
  std::make_tuple("conv2d_transpose_bias", conv2d_transpose_bias_pattern),
  std::make_tuple("conv2d_transpose_relu", conv2d_transpose_relu_pattern),
  std::make_tuple("conv2d_transpose", conv2d_transpose_pattern),
  std::make_tuple("keras_conv2d_transpose_bias_relu", keras_conv2d_transpose_bias_relu_pattern),
  std::make_tuple("keras_conv2d_transpose_bias", keras_conv2d_transpose_bias_pattern),
  std::make_tuple("keras_conv2d_transpose_relu", keras_conv2d_transpose_relu_pattern),
  std::make_tuple("keras_conv2d_transpose", keras_conv2d_transpose_pattern),
  std::make_tuple("conv2d_backprop_input_bias_relu", conv2d_backprop_input_bias_relu_pattern),
  std::make_tuple("conv2d_backprop_input_bias", conv2d_backprop_input_bias_pattern),
  std::make_tuple("conv2d_backprop_input_relu", conv2d_backprop_input_relu_pattern),
  std::make_tuple("conv2d_backprop_input", conv2d_backprop_input_pattern),
  std::make_tuple("upsampling", upsampling_pattern),
  std::make_tuple("resize_bilinear", resize_pattern),
  std::make_tuple("tpu_nearest_neighbor_upsampling", tpu_nearest_neighbor_upsampling_pattern),
  std::make_tuple("batchnorm_relu", batchnorm_relu_pattern),
  std::make_tuple("batchnorm", batchnorm_pattern),
  std::make_tuple("array_relu", array_relu_pattern),
  std::make_tuple("array", array_pattern),
  std::make_tuple("avgpool_mul", avgpool_mul_pattern),
  std::make_tuple("other_relu", other_relu_pattern),
  std::make_tuple("other", other_pattern),
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

const std::vector<std::tuple<const string, const OpTypePattern>> known_ignore_patterns({
  std::make_tuple("convfc_fusedbn", convfc_fusedbn_pattern),
  std::make_tuple("convfc_id_fusedbn", convfc_id_fusedbn_pattern),
  });

const std::set<string> compute_patterns{
    "convfc",
    "convfc_relu",
    "convfc_bias",
    "convfc_bias_relu",
    "convfc_bias_id_relu",
    "atrous_conv",
    "atrous_conv_relu",
    "atrous_conv_bias",
    "atrous_conv_bias_relu",
    "convfc_leakyrelu",
    "convfc_fused_leakyrelu",
    "convfc_keras_leakyrelu",
    "convfc_bias_leakyrelu",
    "convfc_bias_fused_leakyrelu",
    "convfc_bias_keras_leakyrelu",
    "conv2d_transpose",
    "conv2d_transpose_relu",
    "conv2d_transpose_bias",
    "conv2d_transpose_bias_relu",
    "keras_conv2d_transpose",
    "keras_conv2d_transpose_relu",
    "keras_conv2d_transpose_bias",
    "keras_conv2d_transpose_bias_relu",
    "conv2d_backprop_input",
    "conv2d_backprop_input_relu",
    "conv2d_backprop_input_bias",
    "conv2d_backprop_input_bias_relu"};

const string get_pattern_name_from_id(const int pattern_id) {
  if (pattern_id < 0 || pattern_id > known_patterns.size()) {
    LOG(FATAL) << "Invalid pattern id: " << pattern_id;
  }
  return std::get<0>(known_patterns[pattern_id]);
}

const string get_ignore_pattern_name_from_id(const int pattern_id) {
    if (pattern_id < 0 || pattern_id > known_ignore_patterns.size()) {
    LOG(FATAL) << "Invalid pattern id: " << pattern_id;
  }
  return std::get<0>(known_ignore_patterns[pattern_id]);
}
// clang-format on

std::vector<const NodeDef*> get_input_nodes(const NodeMatch& match,
                                            const string& pattern_name) {
  std::vector<const NodeDef*> input_nodes;
  if (pattern_name == "placeholder") {
    // No input
  } else if (pattern_name == "atrous_conv_bias_relu") {
    input_nodes.push_back(
        &(match.inputs[0].inputs[0].inputs[0].inputs[0].inputs[0].node));
  } else if (pattern_name == "atrous_conv_bias") {
    input_nodes.push_back(
        &(match.inputs[0].inputs[0].inputs[0].inputs[0].node));
  } else if (pattern_name == "atrous_conv_relu") {
    input_nodes.push_back(
        &(match.inputs[0].inputs[0].inputs[0].inputs[0].node));
  } else if (pattern_name == "atrous_conv") {
    input_nodes.push_back(&(match.inputs[0].inputs[0].inputs[0].node));
  } else if (pattern_name == "convfc_bias_id_relu") {
    input_nodes.push_back(
        &(match.inputs[0].inputs[0].inputs[0].inputs[0].node));
  } else if (pattern_name == "convfc_bias_relu") {
    input_nodes.push_back(&(match.inputs[0].inputs[0].inputs[0].node));
  } else if (pattern_name == "convfc_bias") {
    input_nodes.push_back(&(match.inputs[0].inputs[0].node));
  } else if (pattern_name == "convfc_relu") {
    input_nodes.push_back(&(match.inputs[0].inputs[0].node));
  } else if (pattern_name == "convfc") {
    input_nodes.push_back(&(match.inputs[0].node));
  } else if (pattern_name == "conv2d_transpose_bias_relu") {
    input_nodes.push_back(&(match.inputs[0].inputs[0].inputs[2].node));
  } else if (pattern_name == "conv2d_transpose_bias") {
    input_nodes.push_back(&(match.inputs[0].inputs[2].node));
  } else if (pattern_name == "conv2d_transpose_relu") {
    input_nodes.push_back(&(match.inputs[0].inputs[2].node));
  } else if (pattern_name == "conv2d_transpose") {
    input_nodes.push_back(&(match.inputs[2].node));
  } else if (pattern_name == "keras_conv2d_transpose_bias_relu") {
    input_nodes.push_back(&(match.inputs[0].inputs[0].inputs[2].node));
  } else if (pattern_name == "keras_conv2d_transpose_bias") {
    input_nodes.push_back(&(match.inputs[0].inputs[2].node));
  } else if (pattern_name == "keras_conv2d_transpose_relu") {
    input_nodes.push_back(&(match.inputs[0].inputs[2].node));
  } else if (pattern_name == "keras_conv2d_transpose") {
    input_nodes.push_back(&(match.inputs[2].node));
  } else if (pattern_name == "conv2d_backprop_input_bias_relu") {
    input_nodes.push_back(&(match.inputs[0].inputs[0].inputs[2].node));
  } else if (pattern_name == "conv2d_backprop_input_bias") {
    input_nodes.push_back(&(match.inputs[0].inputs[2].node));
  } else if (pattern_name == "conv2d_backprop_input_relu") {
    input_nodes.push_back(&(match.inputs[0].inputs[2].node));
  } else if (pattern_name == "conv2d_backprop_input") {
    input_nodes.push_back(&(match.inputs[2].node));
  } else if (pattern_name == "convfc_bias_fused_leakyrelu") {
    input_nodes.push_back(&(match.inputs[0].inputs[0].inputs[0].node));
  } else if (pattern_name == "convfc_bias_leakyrelu") {
    input_nodes.push_back(&(match.inputs[1].inputs[0].inputs[0].node));
  } else if (pattern_name == "convfc_bias_keras_leakyrelu") {
    input_nodes.push_back(
        &(match.inputs[0].inputs[0].inputs[0].inputs[0].node));
  } else if (pattern_name == "convfc_fused_leakyrelu") {
    input_nodes.push_back(&(match.inputs[0].inputs[0].node));
  } else if (pattern_name == "convfc_leakyrelu") {
    input_nodes.push_back(&(match.inputs[1].inputs[0].node));
  } else if (pattern_name == "convfc_keras_leakyrelu") {
    input_nodes.push_back(&(match.inputs[0].inputs[0].inputs[0].node));
  } else if (pattern_name == "leakyrelu") {
    input_nodes.push_back(&(match.inputs[1].node));
  } else if (pattern_name == "fused_leakyrelu") {
    input_nodes.push_back(&(match.inputs[0].node));
  } else if (pattern_name == "keras_leakyrelu") {
    input_nodes.push_back(&(match.inputs[0].inputs[0].node));
  } else if (pattern_name == "upsampling") {
    input_nodes.push_back(&(match.inputs[0].node));
  } else if (pattern_name == "resize_bilinear") {
    input_nodes.push_back(&(match.inputs[0].node));
  } else if (pattern_name == "tpu_nearest_neighbor_upsampling") {
    input_nodes.push_back(&(match.inputs[0].inputs[0].inputs[0].node));
  } else if (pattern_name == "batchnorm_relu") {
    input_nodes.push_back(&(match.inputs[0].inputs[0].inputs[0].node));
  } else if (pattern_name == "batchnorm") {
    input_nodes.push_back(&(match.inputs[0].inputs[0].node));
  } else if (pattern_name == "array_relu") {
    input_nodes.push_back(&(match.inputs[0].inputs[0].node));
    input_nodes.push_back(&(match.inputs[0].inputs[1].node));
  } else if (pattern_name == "array") {
    input_nodes.push_back(&(match.inputs[0].node));
    input_nodes.push_back(&(match.inputs[1].node));
  } else if (pattern_name == "avgpool_mul") {
    input_nodes.push_back(&(match.inputs[0].inputs[0].node));
  } else if (pattern_name == "other_relu") {
  } else if (pattern_name == "other") {
  } else {
    LOG(FATAL) << "Unknown pattern_name: " << pattern_name;
  }
  return input_nodes;
}

std::vector<const NodeDef*> get_ignore_nodes(const NodeMatch& match,
                                             const string& pattern_name) {
  std::vector<const NodeDef*> ignore_nodes;
  if (pattern_name == "convfc_fusedbn") {
    ignore_nodes.push_back(&(match.inputs[0].node));
  } else if (pattern_name == "convfc_id_fusedbn") {
    ignore_nodes.push_back(&(match.inputs[0].inputs[0].node));
  } else {
    LOG(FATAL) << "Unknown pattern_name: " << pattern_name;
  }
  return ignore_nodes;
}

}  // namespace decent_q
}  // namespace tensorflow
