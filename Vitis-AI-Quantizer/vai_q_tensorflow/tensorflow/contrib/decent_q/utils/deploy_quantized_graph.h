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

#ifndef TENSORFLOW_CONTRIB_DECENT_Q_UTILS_DEPLOY_QUANTIZED_GRAPH_H_
#define TENSORFLOW_CONTRIB_DECENT_Q_UTILS_DEPLOY_QUANTIZED_GRAPH_H_

#include "tensorflow/contrib/decent_q/utils/graph_quantizer.h"
#include "tensorflow/contrib/decent_q/utils/transform_utils.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace decent_q {

// Fold FixNeuron into previous op
// - Const + FixNeuron -> Const(with wpos)
// - Non-Const + FixNeuron -> Non-Const(with opos)
Status FoldFixNeuron(const GraphDef& input_graph_def,
                     GraphDef* output_graph_def);

// Fold op parameter input into attributes
Status FoldOpParams(const GraphDef& input_graph_def,
                    GraphDef* output_graph_def);

// Convert Patterns to Deephi format
Status ConvertPatterns(const GraphDef& input_graph_def,
                       GraphDef* output_graph_def, QuantizeConfig& config);

// Fill up [ipos,opos] for the nodes not connected with FixNeuron op
Status PolishActivationInfo(const GraphDef& input_graph_def,
                            GraphDef* output_graph_def);

// Deploy Quantized Graph to folded version and add quantize info to nodes
//
// Steps:
//   1. Fold FixNeuronOps into previous node
//   2. Convert some patterns into deephi patterns
//   3. Polish activation quantize info for every node
Status DeployQuantizedGraph(const GraphDef& input_graph_def,
                            GraphDef* output_graph_def,
                            QuantizeConfig config = QuantizeConfig());

// Command Wrapper
Status DeployQuantizedGraphCommand(const GraphDef& input_graph_def,
                                   const TransformFuncContext& context,
                                   GraphDef* output_graph_def);

}  // namespace decent_q
}  // namespace tensorflow

#endif  // TENSORFLOW_CONTRIB_DECENT_Q_UTILS_DEPLOY_QUANTIZED_GRAPH_H_
