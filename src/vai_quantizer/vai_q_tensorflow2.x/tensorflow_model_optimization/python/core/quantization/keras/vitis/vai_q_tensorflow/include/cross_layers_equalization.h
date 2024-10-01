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

#ifndef TENSORFLOW_CONTRIB_DECENT_Q_UTILS_CROSS_LAYERS_EQUALIZETION_
#define TENSORFLOW_CONTRIB_DECENT_Q_UTILS_CROSS_LAYERS_EQUALIZETION_

#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace decent_q {

// Replace Relu6 With Relu to meet the constraints af(x) = f(ax)
Status ReplaceRelu6WithRelu(const GraphDef& input_graph_def,
                            GraphDef* output_graph_def);

// add channel wise mul for short cut
// Status ConvertShortCut(const GraphDef& input_graph_def,
                              //GraphDef& output_graph_def);

typedef std::pair<const NodeDef*, const NodeDef*> ConvBiasPair;

// Get a dict that map node to its output node name
void MapNamesToOutputNodesName(
    const GraphDef& input_graph_def,
    std::map<string, std::vector<const NodeDef*>>& output_node_map);

// parse conv pairs from graph
Status ParseConvPairs(
    const GraphDef& input_graph_def,
    std::vector<std::vector<ConvBiasPair>>& conv_group,
    const std::vector<string>& input_nodes,
    const std::vector<string>& output_nodes);

// do equalization,
Status EqualizeConvPair(const GraphDef& input_graph_def,
                      GraphDef& output_graph_def, const ConvBiasPair& head,
                      const ConvBiasPair& tail, const string& method = "max",
                      const float& threshold = 0.5);

// strip ':' for some node name
string GetRealName(const string& node_name);

}  // namespace decent_q
}  // namespace tensorflow
#endif  // TENSORFLOW_CONTRIB_DECENT_Q_UTILS_CROSS_LAYERS_EQUALIZETION_
