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

#ifndef TENSORFLOW_CONTRIB_DECENT_Q_UTILS_FOLD_BATCHNORMS_H_
#define TENSORFLOW_CONTRIB_DECENT_Q_UTILS_FOLD_BATCHNORMS_H_

#include "transform_utils.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace decent_q {

// Get the merged weights of folded conv and batchnorm
Status GetMergedConvWeights(const NodeDef& conv_node,
                            const NodeDef& weights_node,
                            const NodeDef& mul_values_node,
                            NodeDef* scaled_weights_node);

// Get the merged biases of folded conv, biasadd and batchnorm
Status GetMergedConvBiases(const NodeDef& bias_node,
                           const NodeDef& mul_values_node,
                           const NodeDef& add_values_node,
                           NodeDef* scaled_bias_node);

// Finds monolithic batch norm ops (as used in early versions of TensorFlow) and
// converts them into premultiplied weight inputs to convolutions.
Status FoldOldBatchNorms(const GraphDef& input_graph_def,
                         GraphDef* output_graph_def);

// Finds monolithic batch norm ops (as used in early versions of TensorFlow) and
// converts them into premultiplied weight inputs to convolutions.
Status UpdateOldBatchNorms(const GraphDef& input_graph_def,
                           GraphDef* output_graph_def);

// Converts Conv2D or MatMul ops followed by column-wise Muls into equivalent
// ops with the Mul baked into the convolution weights, to save computation
// during inference.
Status FoldBatchNormsInference(const GraphDef& input_graph_def,
                               GraphDef* output_graph_def);

// Converts Conv2D and a mul ops into a Conv2D
Status FoldConvMulInference(const GraphDef& input_graph_def,
                            GraphDef* output_graph_def);

// Fold batchnorms for quantize training
Status FoldBatchNormsTraining(const GraphDef& input_graph_def,
                              GraphDef* output_graph_def);

Status FoldBatchNorms(const GraphDef& input_graph_def,
                      GraphDef* output_graph_def, bool is_training = false);

// Command Wrapper for Decent_Q Graph Transform
Status FoldBatchNormsCommand(const GraphDef& input_graph_def,
                             const TransformFuncContext& context,
                             GraphDef* output_graph_def);

}  // namespace decent_q
}  // namespace tensorflow

#endif  // TENSORFLOW_CONTRIB_DECENT_Q_UTILS_INSERT_FIX_NEURON_OPS_H_
