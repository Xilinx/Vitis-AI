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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_HLO_REPLICATION_ANALYSIS_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_HLO_REPLICATION_ANALYSIS_H_

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/statusor.h"

namespace xla {

// An HLO pass that determines whether each instruction in the module outputs
// the same value across replicas. It propagates sources of replicated values to
// the rest of the module, where sources include cross-replica-sum, annotated
// entry parameters, and constants.
class HloReplicationAnalysis {
 public:
  // Runs the analysis on module and returns the result or an error.
  static StatusOr<std::unique_ptr<HloReplicationAnalysis>> Run(
      const HloModule* module);

  // Same as above, but the caller can provide additional annotations: a set of
  // while loops that are known to have the same iteration counts across
  // replicas.
  static StatusOr<std::unique_ptr<HloReplicationAnalysis>> Run(
      const HloModule* module, const absl::flat_hash_set<const HloInstruction*>*
                                   loops_known_with_same_iterations);

  // Returns if the HLO instruction outputs the same value (i.e., replicated) at
  // the given index across all replicas.
  bool HloInstructionIsReplicatedAt(const HloInstruction* inst,
                                    const ShapeIndex& index) const;

 private:
  HloReplicationAnalysis(const HloModule* module,
                         const absl::flat_hash_set<const HloInstruction*>*
                             loops_known_with_same_iterations)
      : module_(module),
        loops_known_with_same_iterations_(*loops_known_with_same_iterations) {}

  // Computes hlo_replication_.
  void ComputeHloReplication();

  // A helper function to recursively compute hlo_replication on a computation.
  // Returns whether hlo_replication_ is changed.
  bool ComputeHloReplicationOnComputation(const HloComputation* computation,
                                          bool mark_everything_not_replicated);

  const HloModule* module_;

  // A set of while loops that are known to have the same iteration counts
  // across replicas. This is provided by the caller as additional annotations.
  const absl::flat_hash_set<const HloInstruction*>&
      loops_known_with_same_iterations_;

  // A map from each analyzed HLO instruction to a shape tree that represents
  // whether the instruction outputs the same value across replicas at each
  // shape index.
  absl::flat_hash_map<const HloInstruction*, ShapeTree<bool>> hlo_replication_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_HLO_REPLICATION_ANALYSIS_H_
