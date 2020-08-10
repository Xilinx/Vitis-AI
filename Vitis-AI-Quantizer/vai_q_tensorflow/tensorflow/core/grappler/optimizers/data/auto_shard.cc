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

#include "tensorflow/core/grappler/optimizers/data/auto_shard.h"

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/grappler/clusters/cluster.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/mutable_graph_view.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.h"
#include "tensorflow/core/grappler/optimizers/data/graph_utils.h"
#include "tensorflow/core/grappler/utils/functions.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {
namespace grappler {
namespace {

// clang-format off
constexpr char kShardDatasetOpName[] = "ShardDataset";
constexpr char kShuffleDatasetOpName[] = "ShuffleDataset";
constexpr char kShuffleDatasetV2OpName[] = "ShuffleDatasetV2";

constexpr std::array<const char*, 4> kReaderDatasetOps = {
    "FixedLengthRecordDataset",
    "FixedLengthRecordDatasetV2",
    "TextLineDataset",
    "TFRecordDataset"
};

constexpr std::array<const char*, 2> kMultipleInputsDatasetOps = {
    "ConcatenateDataset",
    "ZipDataset"
};

constexpr std::array<const char*, 25> kPassThroughOps = {
    "_Retval",
    "BatchDataset",
    "BatchDatasetV2",
    "ExperimentalMapAndBatchDataset",
    "PaddedBatchDataset",
    "PaddedBatchDatasetV2",
    "CacheDataset",
    "CacheDatasetV2",
    "FilterDataset",
    "Identity",
    "MapAndBatchDataset",
    "MapDataset",
    "ModelDataset",
    "OptimizeDataset",
    "ParallelMapDataset",
    "PrefetchDataset",
    "ReduceDataset",
    "RepeatDataset",
    "ShardDataset",
    "ShuffleAndRepeatDataset",
    "ShuffleDataset",
    "ShuffleDatasetV2",
    "SkipDataset",
    "TakeDataset",
    "WindowDataset",
};

// TODO(frankchn): Process functions within kFuncDatasetOps as well.
constexpr std::array<const char*, 5> kFuncDatasetOps = {
    "ExperimentalParallelInterleaveDataset",
    "FlatMapDataset",
    "InterleaveDataset",
    "ParallelInterleaveDataset",
    "ParallelInterleaveDatasetV2"
};

constexpr std::array<const char*, 5> kUnshardableSourceDatasetOps = {
    "GeneratorDataset",
    "RangeDataset",
    "SparseTensorsSliceDataset",
    "TensorDataset",
    "TensorSliceDataset",
};
// clang-format on

Status OptimizeGraph(const GrapplerItem& item, int64 num_workers, int64 index,
                     GraphDef* output);

template <std::size_t SIZE>
bool IsDatasetNodeOfType(const NodeDef& node,
                         const std::array<const char*, SIZE>& arr) {
  for (const auto& dataset_op_name : arr) {
    if (node.op() == dataset_op_name) return true;
  }
  return false;
}

Status AddShardNode(MutableGraphView* graph, const NodeDef& add_before,
                    int64 num_workers, int64 index) {
  NodeDef new_node;
  new_node.set_op(kShardDatasetOpName);
  graph_utils::SetUniqueGraphNodeName(kShardDatasetOpName, graph->graph(),
                                      &new_node);

  // Construct argument nodes
  NodeDef* num_shards_node =
      graph_utils::AddScalarConstNode<int64>(num_workers, graph);
  NodeDef* index_node = graph_utils::AddScalarConstNode<int64>(index, graph);

  // Add inputs to new node
  new_node.add_input(add_before.input(0));
  new_node.add_input(num_shards_node->name());
  new_node.add_input(index_node->name());

  // Ensure that each shard will have at least one element.
  (*(new_node.mutable_attr()))["require_non_empty"].set_b(true);

  // Add shapes and other attributes
  NodeDef* add_after = graph->GetNode(add_before.input(0));

  if (absl::EndsWith(add_after->op(), "Dataset") ||
      absl::EndsWith(add_after->op(), "DatasetV2")) {
    // We still may or may not have the right attributes because Datasets like
    // TFRecordDataset doesn't have a output type or shape, and by default we
    // set them to DT_STRING and an unknown shape.
    if (add_after->attr().count("output_shapes") > 0) {
      graph_utils::CopyAttribute("output_shapes", *add_after, &new_node);
    } else {
      tensorflow::TensorShapeProto* shape =
          (*(new_node.mutable_attr()))["output_shapes"]
              .mutable_list()
              ->add_shape();
      shape->set_unknown_rank(true);
    }

    if (add_after->attr().count("output_types") > 0) {
      graph_utils::CopyAttribute("output_types", *add_after, &new_node);
    } else if (add_after->attr().count("Toutput_types") > 0) {
      (*(new_node.mutable_attr()))["output_types"] =
          add_after->attr().at("Toutput_types");
    } else {
      (*(new_node.mutable_attr()))["output_types"].mutable_list()->add_type(
          tensorflow::DataType::DT_STRING);
    }
  } else {
    // TODO(frankchn): Make this work for datasets where input(0) is a Const,
    // and we need to shard the Const.
    // This is probably not a dataset, so we bail because we can't infer the
    // output types and shape.
    LOG(WARNING)
        << "Unable to shard this input. You may need to wrap "
           "the inputs to your reader dataset in a TensorSliceDataset.";
    LOG(WARNING) << "Input node is: " << add_after->DebugString();
    return errors::NotFound("Cannot shard non-dataset node.");
  }

  // Add new node into graph and update edges
  NodeDef* new_node_graph = graph->AddNode(std::move(new_node));
  TF_RETURN_IF_ERROR(
      graph->UpdateFanouts(add_after->name(), new_node_graph->name()));

  return Status::OK();
}

Status AddShuffleNode(MutableGraphView* graph, const NodeDef& add_before,
                      const string& buffer_size_node, const string& seed_node,
                      const string& seed2_node, bool reshuffle_each_iteration) {
  NodeDef* add_after = graph->GetNode(add_before.input(0));
  NodeDef new_node;
  new_node.set_op(kShuffleDatasetOpName);
  graph_utils::SetUniqueGraphNodeName(kShuffleDatasetOpName, graph->graph(),
                                      &new_node);

  new_node.add_input(add_before.input(0));
  new_node.add_input(buffer_size_node);
  new_node.add_input(seed_node);
  new_node.add_input(seed2_node);

  graph_utils::CopyAttribute("output_shapes", *add_after, &new_node);
  graph_utils::CopyAttribute("output_types", *add_after, &new_node);

  AttrValue reshuffle_attr;
  reshuffle_attr.set_b(reshuffle_each_iteration);
  (*new_node.mutable_attr())["reshuffle_each_iteration"] = reshuffle_attr;

  NodeDef* new_node_graph = graph->AddNode(std::move(new_node));

  TF_RETURN_IF_ERROR(
      graph->UpdateFanouts(add_after->name(), new_node_graph->name()));
  return Status::OK();
}

Status AddShuffleV2Node(MutableGraphView* graph, const NodeDef& add_before,
                        const string& buffer_size_node,
                        const string& seed_generator_node) {
  NodeDef* add_after = graph->GetNode(add_before.input(0));
  NodeDef new_node;
  new_node.set_op(kShuffleDatasetV2OpName);
  graph_utils::SetUniqueGraphNodeName(kShuffleDatasetV2OpName, graph->graph(),
                                      &new_node);

  new_node.add_input(add_before.input(0));
  new_node.add_input(buffer_size_node);
  new_node.add_input(seed_generator_node);

  graph_utils::CopyAttribute("output_shapes", *add_after, &new_node);
  graph_utils::CopyAttribute("output_types", *add_after, &new_node);

  NodeDef* new_node_graph = graph->AddNode(std::move(new_node));

  TF_RETURN_IF_ERROR(
      graph->UpdateFanouts(add_after->name(), new_node_graph->name()));
  return Status::OK();
}

bool ReaderOpInFunction(const NodeDef& node,
                        const FunctionLibraryDefinition& flib) {
  const FunctionDef* func = flib.Find(node.attr().at("f").func().name());
  for (int i = 0; i < func->node_def_size(); i++) {
    NodeDef node_in_func = func->node_def(i);
    if (IsDatasetNodeOfType(node_in_func, kReaderDatasetOps) &&
        node_in_func.input_size() > 0 &&
        absl::StartsWith(node_in_func.input(0), "args_0")) {
      return true;
    }
    if (IsDatasetNodeOfType(func->node_def(i), kFuncDatasetOps) &&
        ReaderOpInFunction(func->node_def(i), flib)) {
      return true;
    }
  }
  return false;
}

Status RemoveShuffleDataset(MutableGraphView* graph, const NodeDef& node,
                            absl::flat_hash_set<string>* nodes_to_delete,
                            string* op_name, string* buffer_size_node,
                            string* seed_node, string* seed2_node,
                            bool* reshuffle_each_iteration) {
  if (node.op() == kShuffleDatasetOpName) {
    *op_name = node.op();
    *buffer_size_node = node.input(1);
    *seed_node = node.input(2);
    *seed2_node = node.input(3);
    *reshuffle_each_iteration = node.attr().at("reshuffle_each_iteration").b();
    TF_RETURN_IF_ERROR(graph->UpdateFanouts(node.name(), node.input(0)));
    nodes_to_delete->insert(node.name());
  }

  for (const auto& fanin : graph->GetFanins(node, true)) {
    TF_RETURN_IF_ERROR(RemoveShuffleDataset(
        graph, *fanin.node, nodes_to_delete, op_name, buffer_size_node,
        seed_node, seed2_node, reshuffle_each_iteration));
  }

  // TODO(frankchn): Traverse functions too.
  return Status::OK();
}

Status RemoveShuffleDatasetV2(MutableGraphView* graph, const NodeDef& node,
                              absl::flat_hash_set<string>* nodes_to_delete,
                              string* op_name, string* buffer_size_node,
                              string* seed_generator_node) {
  if (node.op() == kShuffleDatasetV2OpName) {
    *op_name = node.op();
    *buffer_size_node = node.input(1);
    *seed_generator_node = node.input(2);
    TF_RETURN_IF_ERROR(graph->UpdateFanouts(node.name(), node.input(0)));
    nodes_to_delete->insert(node.name());
  }

  for (const auto& fanin : graph->GetFanins(node, true)) {
    TF_RETURN_IF_ERROR(
        RemoveShuffleDatasetV2(graph, *fanin.node, nodes_to_delete, op_name,
                               buffer_size_node, seed_generator_node));
  }

  // TODO(frankchn): Traverse functions too.
  return Status::OK();
}

Status ProcessDatasetSourceNode(MutableGraphView* graph, const NodeDef& node,
                                absl::flat_hash_set<string>* nodes_to_delete,
                                int64 num_workers, int64 index) {
  string shuffle_op_name = "";
  string buffer_size_node = "";
  string seed_node = "";
  string seed2_node = "";
  string seed_generator_node = "";
  bool reshuffle_each_iteration;

  TF_RETURN_IF_ERROR(AddShardNode(graph, node, num_workers, index));
  TF_RETURN_IF_ERROR(RemoveShuffleDataset(
      graph, node, nodes_to_delete, &shuffle_op_name, &buffer_size_node,
      &seed_node, &seed2_node, &reshuffle_each_iteration));
  if (shuffle_op_name.empty()) {
    TF_RETURN_IF_ERROR(
        RemoveShuffleDatasetV2(graph, node, nodes_to_delete, &shuffle_op_name,
                               &buffer_size_node, &seed_generator_node));
  }

  if (shuffle_op_name == kShuffleDatasetOpName) {
    TF_RETURN_IF_ERROR(AddShuffleNode(graph, node, buffer_size_node, seed_node,
                                      seed2_node, reshuffle_each_iteration));
  } else if (shuffle_op_name == kShuffleDatasetV2OpName) {
    TF_RETURN_IF_ERROR(
        AddShuffleV2Node(graph, node, buffer_size_node, seed_generator_node));
  }

  return Status::OK();
}

Status RecursivelyHandleOp(const NodeDef& node, int64 num_workers, int64 index,
                           FunctionLibraryDefinition* flib,
                           MutableGraphView* graph,
                           absl::flat_hash_set<string>* nodes_to_delete) {
  if (IsDatasetNodeOfType(node, kUnshardableSourceDatasetOps)) {
    return errors::NotFound("Found an unshardable source dataset: ",
                            node.DebugString());
  }

  if (IsDatasetNodeOfType(node, kMultipleInputsDatasetOps)) {
    for (int i = 0; i < node.input_size(); ++i) {
      const NodeDef* input_node = graph_utils::GetInputNode(node, *graph, i);
      TF_RETURN_IF_ERROR(RecursivelyHandleOp(*input_node, num_workers, index,
                                             flib, graph, nodes_to_delete));
    }
    return Status::OK();
  }

  // This handles the case where a reader Dataset is contained within a
  // FuncDataset (e.g. FlatMap, ParallelInterleave, etc...). For example:
  //
  // dataset = Dataset.list_files("/path/to/data")
  // dataset = dataset.flat_map(core_readers.TFRecordDataset)
  //
  // where the list of files is passed in one-by-one as an argument to the
  // function in flat_map.
  if (IsDatasetNodeOfType(node, kFuncDatasetOps) &&
      ReaderOpInFunction(node, *flib)) {
    return ProcessDatasetSourceNode(graph, node, nodes_to_delete, num_workers,
                                    index);
  }

  if (IsDatasetNodeOfType(node, kReaderDatasetOps)) {
    // We reached a reader dataset directly and we try to shard input 0.
    return ProcessDatasetSourceNode(graph, node, nodes_to_delete, num_workers,
                                    index);
  }

  if (!IsDatasetNodeOfType(node, kPassThroughOps)) {
    return errors::NotFound(
        "Did not find a shardable source, walked to ",
        "a node which is not a dataset: ", node.DebugString());
  }

  const NodeDef* input_node = graph_utils::GetInputNode(node, *graph, 0);
  return RecursivelyHandleOp(*input_node, num_workers, index, flib, graph,
                             nodes_to_delete);
}

Status OptimizeGraph(const GrapplerItem& item, int64 num_workers, int64 index,
                     GraphDef* output) {
  if (num_workers == 1 && index == 0) {
    return Status::OK();
  }

  *output = item.graph;
  MutableGraphView graph(output);
  FunctionLibraryDefinition flib(OpRegistry::Global(), item.graph.library());

  NodeDef target_node;
  absl::flat_hash_set<string> nodes_to_delete;

  // The basic approach here is to walk the graph from sink to source, and find
  // the latest occurrence of a ReaderDataset (e.g. CSVDataset, TFRecordDataset,
  // etc...). We then add a shard after that dataset to shard the outputs of
  // that dataset, in effect giving a piece to each worker. Finally, we remove
  // occurences from randomness from before that point in the graph (e.g. things
  // like ShuffleDataset) to ensure that `shard` returns a sensible result.
  NodeDef* sink_node;
  TF_RETURN_IF_ERROR(graph_utils::GetFetchNode(graph, item, &sink_node));
  Status s = RecursivelyHandleOp(*sink_node, num_workers, index, &flib, &graph,
                                 &nodes_to_delete);

  if (!s.ok() && errors::IsNotFound(s)) {
    LOG(WARNING) << "Cannot find shardable dataset, adding a shard node at "
                 << "the end of the dataset instead. This may have performance "
                 << "implications.";
    TF_RETURN_IF_ERROR(AddShardNode(&graph, *sink_node, num_workers, index));
  } else if (!s.ok()) {
    return s;
  }

  TF_RETURN_IF_ERROR(graph.DeleteNodes(nodes_to_delete));

  return Status::OK();
}

}  // anonymous namespace

Status AutoShard::Init(
    const tensorflow::RewriterConfig_CustomGraphOptimizer* config) {
  if (!config) return errors::InvalidArgument("RewriterConfig not found.");

  if ((config->parameter_map().find("num_workers") ==
       config->parameter_map().end())) {
    return errors::InvalidArgument("num_workers parameter missing.");
  }

  if ((config->parameter_map().find("index") ==
       config->parameter_map().end())) {
    return errors::InvalidArgument("index parameter missing.");
  }

  num_workers_ = config->parameter_map().at("num_workers").i();
  index_ = config->parameter_map().at("index").i();

  if (num_workers_ < 1) {
    return errors::InvalidArgument("num_workers should be >= 1, currently ",
                                   num_workers_);
  }

  if (index_ < 0 || index_ >= num_workers_) {
    return errors::InvalidArgument("index should be >= 0 and < ", num_workers_,
                                   ", currently ", index_);
  }

  return Status::OK();
}

Status AutoShard::OptimizeAndCollectStats(Cluster* /* cluster */,
                                          const GrapplerItem& item,
                                          GraphDef* output,
                                          OptimizationStats* stats) {
  *output = item.graph;
  TF_RETURN_IF_ERROR(OptimizeGraph(item, num_workers_, index_, output));
  stats->num_changes++;
  return Status::OK();
}

REGISTER_GRAPH_OPTIMIZER_AS(AutoShard, "tf_auto_shard");

}  // namespace grappler
}  // namespace tensorflow
