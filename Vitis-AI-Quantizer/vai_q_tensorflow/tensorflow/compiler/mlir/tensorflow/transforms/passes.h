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

#ifndef TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSFORMS_PASSES_H_
#define TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSFORMS_PASSES_H_

#include "mlir/Pass/Pass.h"  // TF:local_config_mlir

namespace mlir {
namespace TF {
// Transforms functional control flow operations in the standard TensorFlow
// dialect to MLIR Control Flow Graph (CFG) form.
std::unique_ptr<FunctionPassBase> CreateTFFunctionalControlFlowToCFG();

// Optimizes Tensorflow graph.
std::unique_ptr<FunctionPassBase> CreateTFOptimizePass();

}  // namespace TF

namespace TFControlFlow {
// Raises from the "TensorFlow Control Flow" dialect to the standard TensorFlow
// dialect.
std::unique_ptr<FunctionPassBase> CreateRaiseTFControlFlowPass();

}  // namespace TFControlFlow

namespace tf_executor {
class GraphOp;

// Create a pass to merge IslandOps from TFExecutor dialect.
std::unique_ptr<FunctionPassBase> CreateTFExecutorIslandCoarseningPass();

// Create a pass to prune tf_executor.graph from dead nodes.
FunctionPassBase* CreateTFExecutorGraphPruningPass();

// Prune a tf_executor.graph operation from dead nodes.
void prune_graph(GraphOp graph);

}  // namespace tf_executor

namespace TFDevice {
// Creates a pass that forms clusters from instructions that are assigned to
// same device.
std::unique_ptr<FunctionPassBase> CreateClusterFormationPass();

// Creates a pass that outlines regions of tf_device.launch operations.
std::unique_ptr<ModulePassBase> CreateClusterOutliningPass();
}  // namespace TFDevice

}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSFORMS_PASSES_H_
