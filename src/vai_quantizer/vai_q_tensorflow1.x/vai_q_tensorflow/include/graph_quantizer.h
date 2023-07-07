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

#ifndef TENSORFLOW_CONTRIB_DECENT_Q_UTILS_GRAPH_QUANTIZER_H_
#define TENSORFLOW_CONTRIB_DECENT_Q_UTILS_GRAPH_QUANTIZER_H_

#include "transform_utils.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace decent_q {

enum QuantizePhase { CALIB, EVAL, TRAIN };
enum QuantizeMethod { NOOF, DIFFS, DW_DIFFS };
enum QuantizeMode { WEIGHT, ACTIVATION, DW_WEIGHT };

class QuantizeConfig {
 public:
  // The quantize behavior phase: 0:calibration, 1:evaluation, 2:train
  QuantizePhase phase;

  // The quantize method: 0:non-overflow, 1:Min-diffs, 2:Min-diffs with
  // normalize
  QuantizeMethod method;

  // The bit width for weights/biases
  int weight_bit;

  // The bit width for activation
  int activation_bit;

  // specified op quantize bit width map
  std::map<string, int> nodes_bit;

  // specified op quantize bit width map
  std::map<string, int> nodes_method;

  // The quantize mode: 0: weight, 1:activation, 2: depthwise quantize strategy
  QuantizeMode mode;

  // Vector of input node names
  std::vector<string> input_nodes;

  // Vector of quant output node names
  std::vector<string> output_nodes;

  // Vector of ignore node names
  std::set<string> ignore_nodes;

  // Vector of input_shapes
  std::vector<string> input_shapes;

  // Vector of quant_input_dtypes
  std::vector<string> quant_input_dtypes;

  // Calib iter
  int calib_iter;

  // Output directory
  string output_dir;

  // Align maxpool and avgpool
  int align_pool;

  // Align concat
  int align_concat;

  // Adjust shift bias
  int adjust_shift_bias;

  // Adjust shift cut
  int adjust_shift_cut;

  // Simulate dpu
  int simulate_dpu;

  // do cross layers equalization
  int do_cle;

  // Scale all avgpool
  int scale_all_avgpool;

  // replace relu6 with relu
  int replace_relu6;

  // replace sigmoid with hard-sigmoid
  int replace_sigmoid;

  // if only fold bn when create optimized graph
  int fold_bn_only;

  // replace softmax with dpu_softmax
  int replace_softmax;

  QuantizeConfig(const QuantizePhase& phase = QuantizePhase::CALIB,
                 const QuantizeMethod& method = QuantizeMethod::NOOF,
                 const int& weight_bit = 8, const int& activation_bit = 8,
                 const std::map<string, int> nodes_bit = {},
                 const std::map<string, int> nodes_method = {},
                 const QuantizeMode& mode = QuantizeMode::WEIGHT,
                 const std::vector<string> input_nodes = {},
                 const std::vector<string> output_nodes = {},
                 const std::set<string> ignore_nodes = {},
                 const std::vector<string> input_shapes = {},
                 const std::vector<string> quant_input_dtypes = {},
                 const int& calib_iter = 0, const string output_dir = "",
                 const int& align_concat = 0, const int& align_pool = 0,
                 const int& adjust_shift_bias = 0, const int& adjust_shift_cut = 0,
                 const int& simulate_dpu = 0, const int& do_cle = 0,
                 const int& scale_all_avgpool = 1, const int& replace_relu6 = 1,
                 const int& replace_sigmoid = 0,
                 const int& fold_bn_only = 0,
                 const int& replace_softmax = 0)
      : phase(phase),
        method(method),
        weight_bit(weight_bit),
        activation_bit(activation_bit),
        nodes_bit(nodes_bit),
        nodes_method(nodes_method),
        mode(mode),
        input_nodes(input_nodes),
        output_nodes(output_nodes),
        ignore_nodes(ignore_nodes),
        input_shapes(input_shapes),
        quant_input_dtypes(input_shapes),
        calib_iter(calib_iter),
        output_dir(output_dir),
        align_concat(align_concat),
        align_pool(align_pool),
        adjust_shift_bias(adjust_shift_bias),
        adjust_shift_cut(adjust_shift_cut),
        simulate_dpu(simulate_dpu),
        do_cle(do_cle),
        scale_all_avgpool(scale_all_avgpool),
        replace_relu6(replace_relu6),
        replace_sigmoid(replace_sigmoid),
        fold_bn_only(fold_bn_only),
        replace_softmax(replace_softmax) {}

  Status FromString(const string config_string);
};

typedef std::vector<string> NodeGroup;

class GraphQuantizer {
 public:
  GraphQuantizer() {}

  GraphQuantizer(const QuantizeConfig& config) : _config(config) {}

  // Check Graph
  Status CheckGraph(const GraphDef& input_graph_def, const string graph_path);

  // replace sigmoid op with hard_sigmoid
  Status ReplaceSigmoidWithHardSigmoid(const GraphDef& input_graph_def,
                                       GraphDef* output_graph_def);

  // Convert Constants To Variables
  Status ConvertConstantsToVariables(const GraphDef& input_graph_def,
                                     GraphDef& output_graph_def);

  // Partition graph into main graph and aux graph according to input_nodes and
  // output_nodes
  Status PartitionGraph(const GraphDef& input_graph_def,
                        GraphDef& main_graph_def, GraphDef& aux_graph_def,
                        std::map<string, NodeDef>& origin_input_nodes);

  // Merge main graph and aux graph
  Status MergeGraph(const GraphDef& main_graph_def,
                    const GraphDef& aux_graph_def,
                    const std::map<string, NodeDef> origin_input_nodes,
                    GraphDef& output_graph_def);

  // Create Optimized Graph
  Status CreateOptimizedGraph(const GraphDef& input_graph_def,
                              GraphDef& output_graph_def);

  // implement cross layers equlization
  Status CrossLayersEqualization(const GraphDef& input_graph_def,
                                  GraphDef& output_graph_def);

  // Create Quantize Calibration Graph
  Status CreateQuantizeCalibrationGraph(const GraphDef& input_graph_def,
                                        GraphDef& output_graph_def);

  // Create Quantize Training Graph
  Status CreateQuantizeTrainingGraph(const GraphDef& input_graph_def,
                                     GraphDef& output_graph_def);

  // Create Quantize Evaluation Graph
  Status CreateQuantizeEvaluationGraph(const GraphDef& input_graph_def,
                                       GraphDef& output_graph_def);

  // Convert folded batchnorms
  Status ConvertFoldedBatchnorms(const GraphDef& input_graph_def,
                                 GraphDef& output_graph_def);

  // Create Quantize Deploy Graph
  Status CreateQuantizeDeployGraph(const GraphDef& input_graph_def,
                                   GraphDef& output_graph_def);

 private:
  QuantizeConfig _config;

  // vector<pattern_id, match>
  std::vector<std::tuple<int, NodeMatch>> _matched_node_patterns;

  // map<node_name, match_id>
  std::unordered_map<string, int> _matched_nodes;

  // set<node_name>
  std::unordered_set<string> _unmatched_nodes;

  // map<node_name, config>
  std::unordered_map<string, QuantizeConfig> _ops_to_quantize;

  // set<compute_node_name, input_node_name, output_node_name, weight_node_name,
  // bias_node_name>
  std::set<NodeGroup> _node_groups;

  // map<node_name, node_fn_name>
  std::unordered_map<string, string> _node_to_fn;

  Status _InsertFixNeuronOps(const GraphDef& input_graph_def,
                             GraphDef& output_graph_def);

  Status _FreezeFixNeuronOps(const GraphDef& input_graph_def,
                             GraphDef& output_graph_def);

  Status _AdjustQuantizePos(const GraphDef& input_graph_def,
                            GraphDef& output_graph_def);

  Status _LocateOpsToQuantize(const GraphDef& input_graph_def);

  Status _ModifyFixNeuronConfig(const GraphDef& input_graph_def);

  Status _MatchQuantizedNodeName(const GraphDef& input_graph_def,
                                 const std::string& node_name,
                                 std::string& matched_node_name);
};

}  // namespace decent_q
}  // namespace tensorflow

#endif  // TENSORFLOW_CONTRIB_DECENT_Q_UTILS_GRAPH_QUANTIZER_H_
