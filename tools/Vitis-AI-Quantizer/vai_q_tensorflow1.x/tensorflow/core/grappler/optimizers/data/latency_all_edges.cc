/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/grappler/optimizers/data/latency_all_edges.h"

#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/grappler/clusters/cluster.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/mutable_graph_view.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.h"
#include "tensorflow/core/grappler/optimizers/data/graph_utils.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/kernels/data/stats_utils.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/protobuf.h"

namespace tensorflow {
namespace grappler {
namespace {

constexpr char kInsertOpName[] = "LatencyStatsDataset";

NodeDef MakeLatencyNode(const NodeDef& node, MutableGraphView* graph) {
  NodeDef new_node;
  new_node.set_op(kInsertOpName);
  graph_utils::SetUniqueGraphNodeName(strings::StrCat(kInsertOpName),
                                      graph->graph(), &new_node);
  // Set the input of LatencyDataset node as `node`
  new_node.add_input(node.name());

  string tag_name = strings::StrCat("record_latency",
                                    data::stats_utils::kDelimiter, node.name());
  NodeDef* tag = graph_utils::AddScalarConstNode<StringPiece>(
      StringPiece(tag_name), graph);
  new_node.add_input(tag->name());

  // Set `output_types` and `output_shapes` attributes.
  for (auto key : {"output_shapes", "output_types"}) {
    if (node.attr().find(key) != node.attr().end()) {
      (*new_node.mutable_attr())[key] = node.attr().at(key);
    } else {
      const char* kInferredAttrPrefix = "T";
      if (node.attr().find(strings::StrCat(kInferredAttrPrefix, key)) !=
          node.attr().end()) {
        (*new_node.mutable_attr())[key] =
            node.attr().at(strings::StrCat(kInferredAttrPrefix, key));
      }
    }
  }
  return new_node;
}

}  // namespace

Status LatencyAllEdges::OptimizeAndCollectStats(Cluster* cluster,
                                                const GrapplerItem& item,
                                                GraphDef* output,
                                                OptimizationStats* stats) {
  *output = item.graph;
  MutableGraphView graph(output);

  // Add LatencyDatasetOp node after each node.
  // TODO(shivaniagrawal): Add Op to return Latency for the particular Op than
  // for the edge (e2 - e1?).
  for (const NodeDef& node : item.graph.node()) {
    if (!absl::EndsWith(node.op(), "Dataset") || node.attr().empty()) {
      // TODO(b/111805951): Replace this with non-approximate way to check if
      // node corresponds to a `Dataset` op.
      continue;
    }
    NodeDef* latency_node = graph.AddNode(MakeLatencyNode(node, &graph));
    TF_RETURN_IF_ERROR(graph.UpdateFanouts(node.name(), latency_node->name()));
    stats->num_changes++;
  }
  return Status::OK();
}

void LatencyAllEdges::Feedback(Cluster* cluster, const GrapplerItem& item,
                               const GraphDef& optimize_output, double result) {
  // no-op
}

REGISTER_GRAPH_OPTIMIZER_AS(LatencyAllEdges, "latency_all_edges");

}  // namespace grappler
}  // namespace tensorflow
