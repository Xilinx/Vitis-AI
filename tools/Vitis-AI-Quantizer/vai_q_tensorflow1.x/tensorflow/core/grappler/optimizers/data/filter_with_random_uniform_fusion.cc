/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/grappler/optimizers/data/filter_with_random_uniform_fusion.h"

#include <iostream>

#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/grappler/clusters/cluster.h"
#include "tensorflow/core/grappler/graph_view.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.h"
#include "tensorflow/core/grappler/optimizers/data/graph_utils.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/platform/protobuf.h"

namespace tensorflow {
namespace grappler {

constexpr char kFusedOpName[] = "SamplingDataset";

NodeDef MakeFusedNode(const NodeDef& filter_node, float rate, int64 seed,
                      int64 seed2, MutableGraphView* graph) {
  NodeDef fused_node;
  graph_utils::SetUniqueGraphNodeName("fused_sampling", graph->graph(),
                                      &fused_node);
  fused_node.set_op(kFusedOpName);

  // Copy over inputs.
  for (int i = 0; i < filter_node.input_size(); ++i) {
    fused_node.add_input(filter_node.input(i));
  }

  // Required attrs.
  for (auto key : {"output_shapes", "output_types"}) {
    graph_utils::CopyAttribute(key, filter_node, &fused_node);
  }

  // Optional attrs.
  for (auto key : {"use_inter_op_parallelism", "sloppy"}) {
    if (gtl::FindOrNull(filter_node.attr(), key)) {
      graph_utils::CopyAttribute(key, filter_node, &fused_node);
    }
  }

  NodeDef* tmp_rate = graph_utils::AddScalarConstNode<float>(rate, graph);
  fused_node.add_input(tmp_rate->name());
  NodeDef* tmp_seed = graph_utils::AddScalarConstNode<int64>(seed, graph);
  fused_node.add_input(tmp_seed->name());
  NodeDef* tmp_seed2 = graph_utils::AddScalarConstNode<int64>(seed2, graph);
  fused_node.add_input(tmp_seed2->name());

  return fused_node;
}

const NodeDef* FunctionFindNodeDef(const FunctionDef& function, const string op,
                                   const string func, const string match) {
  for (const NodeDef& func_node : function.node_def()) {
    if (func_node.op() != op) {
      continue;
    }
    if (func_node.name() + match != func) {
      continue;
    }
    return &func_node;
  }
  return nullptr;
}

bool FunctionFindFloatConst(const FunctionDef& function, const string& func,
                            const string& match, float* result) {
  const NodeDef* const_node =
      FunctionFindNodeDef(function, "Const", func, match);
  if (const_node == nullptr) {
    return false;
  }
  if (const_node->attr().at("dtype").type() != DT_FLOAT) {
    return false;
  }
  const auto& value = const_node->attr().at("value").tensor().float_val(0);
  *result = value;
  return true;
}

bool FunctionExpectFloatConst(const FunctionDef& function, const string& func,
                              const string match, const float val) {
  float result;
  if (FunctionFindFloatConst(function, func, match, &result) && result == val) {
    return true;
  } else {
    return false;
  }
}

// This optimization fuses one of the following two forms of
// filter + random_uniform predication into a single data sampling operation:
// fuse:
//   filter
//   |
//   + predication: less [0]
//                  |
//                  + random_uniform [1]
//                  |
//                  + rate
// or:
//   filter
//   |
//   + predication: less
//                  |
//                  + random_uniform[]
//                  |
//                  + rate
// into:
//   sampling(rate)
Status FilterWithRandomUniformFusion::OptimizeAndCollectStats(
    Cluster* cluster, const GrapplerItem& item, GraphDef* output,
    OptimizationStats* stats) {
  *output = item.graph;
  MutableGraphView graph(output);
  absl::flat_hash_set<string> nodes_to_delete;
  float rate;
  int64 seed, seed2;

  for (const NodeDef& node : item.graph.node()) {
    // stage 1 -- recognition
    if (node.op() != "FilterDataset") {
      continue;
    }

    // Use a more descriptive variable name
    const NodeDef& filter_node = node;

    // find predicate function of the node
    const auto& predicate = filter_node.attr().at("predicate");
    const string func_name = predicate.func().name();

    bool function_match = false;
    // find the function that matches func_name
    for (const auto& function : item.graph.library().function()) {
      if (function.signature().name() == func_name) {
        if (function.ret().size() != 1) {
          continue;
        }
        auto it = function.ret().begin();
        string node_name = it->second;
        const NodeDef* func_node =
            FunctionFindNodeDef(function, "Identity", node_name, ":output:0");
        while (func_node != nullptr) {
          node_name = func_node->input(0);
          func_node =
              FunctionFindNodeDef(function, "Identity", node_name, ":output:0");
        }
        func_node = FunctionFindNodeDef(function, "StridedSlice", node_name,
                                        ":output:0");
        const NodeDef* less_node;
        if (func_node != nullptr) {
          // for form one: datasetS = datasetS.filter(lambda x:
          // tf.less(tf.random_uniform([1]), rate)[0])
          less_node = FunctionFindNodeDef(function, "Less", func_node->input(0),
                                          ":z:0");
        } else {
          // for form two: datasetS = datasetS.filter(lambda _:
          // tf.random_uniform([]) < rate)
          less_node = FunctionFindNodeDef(function, "Less", node_name, ":z:0");
        }
        if (less_node == nullptr) {
          continue;
        }

        // check whether the function is actually doing
        // random_uniform[0.0, 1.0) < rate
        // There could be two forms of random_uniform[0.0, 1.0) in the graph
        // * Simple form just have a RandomUniform node which means
        //   random_uniform[0.0, 1.0)
        // * Expanded form is "RandomUniform * (1.0 - 0.0) + 0.0", which is
        //   still random_uniform[0.0, 1.0)
        //
        // First detect whether simple form is used
        const NodeDef* random_uniform_node = FunctionFindNodeDef(
            function, "RandomUniform", less_node->input(0), ":output:0");
        if (random_uniform_node == nullptr) {
          // If expanded form is used, check boundaries
          const NodeDef* random_uniform_result_node =
              FunctionFindNodeDef(function, "Add", less_node->input(0), ":z:0");

          if (!FunctionExpectFloatConst(function,
                                        random_uniform_result_node->input(1),
                                        ":output:0", 0.0f)) {
            continue;
          }

          const NodeDef* random_uniform_mul_node = FunctionFindNodeDef(
              function, "Mul", random_uniform_result_node->input(0), ":z:0");

          const NodeDef* random_uniform_sub_node = FunctionFindNodeDef(
              function, "Sub", random_uniform_mul_node->input(1), ":z:0");

          if (!FunctionExpectFloatConst(function,
                                        random_uniform_sub_node->input(0),
                                        ":output:0", 1.0f)) {
            continue;
          }

          if (!FunctionExpectFloatConst(function,
                                        random_uniform_sub_node->input(1),
                                        ":output:0", 0.0f)) {
            continue;
          }

          random_uniform_node = FunctionFindNodeDef(
              function, "RandomUniform", random_uniform_mul_node->input(0),
              ":output:0");
          if (random_uniform_node == nullptr) {
            continue;
          }
        }

        seed = random_uniform_node->attr().at("seed").i();
        seed2 = random_uniform_node->attr().at("seed2").i();

        if (!FunctionFindFloatConst(function, less_node->input(1), ":output:0",
                                    &rate)) {
          continue;
        }

        function_match = true;
        break;
      }
    }

    if (!function_match) {
      continue;
    }

    // stage 2 -- fuse
    const auto* fused_sampling =
        graph.AddNode(MakeFusedNode(filter_node, rate, seed, seed2, &graph));

    TF_RETURN_IF_ERROR(
        graph.UpdateFanouts(filter_node.name(), fused_sampling->name()));

    // Mark the `Filter` node for removal.
    nodes_to_delete.insert(filter_node.name());
    stats->num_changes++;
  }

  TF_RETURN_IF_ERROR(graph.DeleteNodes(nodes_to_delete));
  return Status::OK();
}

void FilterWithRandomUniformFusion::Feedback(Cluster* cluster,
                                             const GrapplerItem& item,
                                             const GraphDef& optimize_output,
                                             double result) {
  // Nothing to do for FilterWithRandomUniformFusion
}

// TODO(b/131229793): The current implementation of the optimization is brittle
// as it depends on the order of inputs to commutative nodes. Make the
// optimization robust to the input ordering before re-enabling it.
// REGISTER_GRAPH_OPTIMIZER_AS(FilterWithRandomUniformFusion,
//                             "filter_with_random_uniform_fusion");

}  // end namespace grappler
}  // end namespace tensorflow
