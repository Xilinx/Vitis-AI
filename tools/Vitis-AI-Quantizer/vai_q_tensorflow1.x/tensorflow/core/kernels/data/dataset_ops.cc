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

#include "tensorflow/core/kernels/data/dataset_ops.h"

#include "tensorflow/core/common_runtime/graph_runner.h"
#include "tensorflow/core/common_runtime/process_function_library_runtime.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/dataset_stateful_op_whitelist.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/kernels/data/dataset_utils.h"

namespace tensorflow {
namespace data {

/* static */ constexpr const char* const DatasetToGraphOp::kStatefulWhitelist;
/* static */ constexpr const char* const DatasetFromGraphOp::kGraphDef;
/* static */ constexpr const char* const DatasetFromGraphOp::kHandle;

// See documentation in ../../ops/dataset_ops.cc for a high-level
// description of the following op.
DatasetToGraphOp::DatasetToGraphOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
  if (ctx->HasAttr(kStatefulWhitelist)) {
    OP_REQUIRES_OK(
        ctx, ctx->GetAttr(kStatefulWhitelist, &whitelisted_stateful_ops_));
  }
}

void DatasetToGraphOp::Compute(OpKernelContext* ctx) {
  DatasetBase* dataset;
  OP_REQUIRES_OK(ctx, GetDatasetFromVariantTensor(ctx->input(0), &dataset));
  std::vector<int> whitelist_indices_to_remove;
  for (int i = 0; i < whitelisted_stateful_ops_.size(); ++i) {
    const string stateful_op = whitelisted_stateful_ops_[i];
    if (!WhitelistedStatefulOpRegistry::Global()->Contains(stateful_op)) {
      whitelist_indices_to_remove.push_back(i);
      // Make sure op is registered first. We maybe don't need this check?
      const OpDef* op_def;
      OP_REQUIRES_OK(ctx,
                     OpRegistry::Global()->LookUpOpDef(stateful_op, &op_def));
      OP_REQUIRES_OK(ctx,
                     WhitelistedStatefulOpRegistry::Global()->Add(stateful_op));
    }
  }
  GraphDef graph_def;
  OP_REQUIRES_OK(
      ctx, AsGraphDef(ctx, dataset, SerializationContext({}), &graph_def));
  Tensor* result;
  OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &result));
  result->scalar<tstring>()() = graph_def.SerializeAsString();
  for (int index : whitelist_indices_to_remove) {
    OP_REQUIRES_OK(ctx, WhitelistedStatefulOpRegistry::Global()->Remove(
                            whitelisted_stateful_ops_[index]));
  }
}

void DatasetCardinalityOp::Compute(OpKernelContext* ctx) {
  DatasetBase* dataset;
  OP_REQUIRES_OK(ctx, GetDatasetFromVariantTensor(ctx->input(0), &dataset));
  Tensor* result;
  OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &result));
  result->scalar<int64>()() = dataset->Cardinality();
}

void DatasetFromGraphOp::Compute(OpKernelContext* ctx) {
  string graph_def_string;
  OP_REQUIRES_OK(ctx,
                 ParseScalarArgument(ctx, kGraphDef, &graph_def_string));
  GraphDef graph_def;
  OP_REQUIRES(ctx, graph_def.ParseFromString(graph_def_string),
              errors::InvalidArgument("Could not parse GraphDef"));
  string output_node;
  for (const auto& node : graph_def.node()) {
    if (node.op() == FunctionLibraryDefinition::kRetOp) {
      output_node = node.input(0);
    }
  }
  Graph graph(OpRegistry::Global());
  OP_REQUIRES_OK(ctx, ImportGraphDef({}, graph_def, &graph, nullptr));

  FunctionLibraryRuntime* flr;
  std::unique_ptr<FunctionLibraryDefinition> flib_def(nullptr);
  std::unique_ptr<ProcessFunctionLibraryRuntime> pflr(nullptr);
  OP_REQUIRES_OK(ctx,
                 ctx->function_library()->Clone(&flib_def, &pflr, &flr, true));

  // Some function names may be duplicated (for example, if the serialized
  // graph has an optimized function that retains its original name). We
  // override functions in flib_def in the event of conflict. It is
  // safe to assume that any node in the serialized graph is referring to the
  // serialized function when there is a conflict.
  OP_REQUIRES_OK(ctx,
                 AddToFunctionLibrary(flib_def.get(), graph_def.library()));

  std::vector<Tensor> outputs;
  GraphRunner graph_runner(flr->device());
  OP_REQUIRES_OK(ctx,
                 graph_runner.Run(&graph, flr, {}, {output_node}, &outputs));
  OP_REQUIRES_OK(ctx, ctx->set_output(kHandle, outputs[0]));
}

REGISTER_KERNEL_BUILDER(Name("DatasetToGraph").Device(DEVICE_CPU),
                        DatasetToGraphOp);

REGISTER_KERNEL_BUILDER(Name("DatasetCardinality").Device(DEVICE_CPU),
                        DatasetCardinalityOp);
REGISTER_KERNEL_BUILDER(
    Name("ExperimentalDatasetCardinality").Device(DEVICE_CPU),
    DatasetCardinalityOp);

REGISTER_KERNEL_BUILDER(Name("DatasetFromGraph").Device(DEVICE_CPU),
                        DatasetFromGraphOp);

}  // namespace data
}  // namespace tensorflow
