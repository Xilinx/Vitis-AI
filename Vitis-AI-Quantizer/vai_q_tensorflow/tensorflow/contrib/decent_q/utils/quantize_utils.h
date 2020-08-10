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

/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CONTRIB_DECENT_Q_UTILS_QUANTIZE_UTILS_H_
#define TENSORFLOW_CONTRIB_DECENT_Q_UTILS_QUANTIZE_UTILS_H_

#include <cmath>
#include <fstream>
#include <set>
#include <unordered_set>
#include <vector>

#include "tensorflow/contrib/decent_q/utils/graph_quantizer.h"
#include "tensorflow/contrib/decent_q/utils/known_patterns.h"
#include "tensorflow/contrib/decent_q/utils/transform_utils.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace decent_q {

// Decent debug logging
#define DLOG_INFO(x)                                                    \
  if (getenv("DECENT_DEBUG") && std::atoi(getenv("DECENT_DEBUG")) >= x) \
  LOG(INFO)

#define DLOG_WARNING LOG(WARNING) << "[DECENT_WARNING] "

// Quantize a float number with given step, and saturate truncation
template <typename T>
T quantize_kernel_cpu(const T x, const T step, const T lower_bound,
                      const T upper_bound);

// Quantize a float array with given pos and bit_width
template <typename T>
void quantize_cpu(const int n, const T* x, T* y, const int bit_width,
                  const int pos);

// Get node.op() by node name.
Status GetNodeTypeByName(const GraphDef& input_graph_def,
                         const string& node_name, string* node_type);

// Record NodaMatch to a vector matched_node_patterns<pattern_id, NodeMatch>
Status RecordMatchedPatterns(
    std::vector<std::tuple<int, NodeMatch>>& matched_node_patterns,
    const int& pattern_id, const NodeMatch& match);

// Record all nodes in NodeMatch to matched_nodes_map<node_name, match_id>
Status RecordMatchedNodes(
    std::unordered_map<string, int>& matched_nodes, const NodeMatch& match,
    const int& match_id,
    const std::set<string>& irrelevant_nodes = std::set<string>());

// Check if any nodes in a match is already recorded in matched_nodes.
// This is used to avoid repeated matching
bool CheckAnyMatchedNodes(
    const std::unordered_map<string, int>& matched_nodes,
    const NodeMatch& match,
    const std::set<string>& irrelevant_nodes = std::set<string>());

// Print matched_node_patterns
void PrintMatchedNodePatterns(
    const std::vector<std::tuple<int, NodeMatch>>& matched_node_patterns);

// Check if any node in the given NodeMatch should be ignored
bool CheckAnyIgnoredNodes(
    const std::set<string>& ignore_nodes, const NodeMatch& match,
    const std::set<string>& irrelevant_nodes = std::set<string>());

// check if specified node name in transformed graph
void CheckSpecifiedNodeNames(
    const GraphDef& input_graph_def,
    std::unordered_map<string, QuantizeConfig> ops_to_quantize,
    const std::map<string, int>& nodes_to_bit);

// Convert constants to variables
Status _ConvertConstantsToVariables(const GraphDef& input_graph_def,
                                    GraphDef* output_graph_def);

// Parse graph def to get all matched patterns
Status ParseGraph(
    const GraphDef& input_graph_def,
    std::vector<std::tuple<int, NodeMatch>>& matched_node_patterns,
    std::unordered_map<string, int>& matched_nodes,
    std::set<string>& ignore_nodes,
    std::unordered_set<string>& unmatched_nodes);

// Get data type of given node
Status GetNodeDataType(const NodeDef& node, DataType* data_type);

// Get data types of given node names
Status GetDataTypeOfNodes(
    const GraphDef& input_graph_def, const std::vector<std::string> node_names,
    std::unordered_map<string, DataType>* data_type_of_nodes);

// Get shapes of given node names and batch_size
Status GetShapeOfNodes(
    const GraphDef& input_graph_def, const std::vector<std::string> node_names,
    const int batch_size,
    std::unordered_map<string, std::vector<int>>* shape_of_nodes);

// Inference shape and make constant input for reshape ops
Status InferenceShape(const GraphDef& input_graph_def,
                      GraphDef* output_graph_def, const int batch_size);

// Convert mean to avgpool to unify the op of avgpool, and for convinient use in
// SimulateDPU
Status ConvertMeanToAvgpool(const GraphDef& input_graph_def,
                            GraphDef* output_graph_def);

// Convert the graph to simulate DPU behaviour, such as avgpooling and leakyrelu
Status SimulateDPU(const GraphDef& input_graph_def, GraphDef* output_graph_def);

// Insert Identity node after nodes and keep the node name unchanged
Status InsertIdForNodes(const GraphDef& input_graph_def,
                        GraphDef* output_graph_def,
                        const std::vector<string>& node_names);

// Save graph_def to output_dir/decent_debug for debugging
Status SaveGraphForDebugging(const GraphDef& graph_def, const string& out,
                             const string& output_dir);

// Save quantize_info to ./decent_debug for debugging
Status SaveQuantizeInfoForDebugging(const GraphDef& input_graph_def,
                                    const string& output_dir);

// Load quantize_info from file into a map
Status LoadQuantizeInfoFromFile(
    const GraphDef& input_graph_def,
    std::unordered_map<string, std::vector<int>>* quantize_info_map,
    const string& output_dir);

// Save node groups infomation to file
Status SaveNodeGroupsToFile(const std::set<NodeGroup>& node_groups,
                            const string& output_dir);

// Load node groups infomation from file
Status LoadNodeGroupsFromFile(std::set<NodeGroup>& node_groups,
                              const string& output_dir);

}  // namespace decent_q
}  // namespace tensorflow

#endif  // TENSORFLOW_CONTRIB_DECENT_Q_UTILS_QUANTIZE_UTILS_H_
