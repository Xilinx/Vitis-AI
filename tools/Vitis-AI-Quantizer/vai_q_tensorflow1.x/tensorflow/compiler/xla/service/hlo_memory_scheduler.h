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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_HLO_MEMORY_SCHEDULER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_HLO_MEMORY_SCHEDULER_H_

#include <vector>

#include "absl/container/flat_hash_map.h"
#include "tensorflow/compiler/xla/service/hlo_alias_analysis.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_ordering.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"
#include "tensorflow/compiler/xla/service/hlo_schedule.h"
#include "tensorflow/compiler/xla/service/logical_buffer.h"
#include "tensorflow/compiler/xla/service/tuple_points_to_analysis.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"

namespace xla {

// A memory scheduler computes an execution sequence for the HLO instructions in
// 'computation' that minimizes peak memory, given a points-to analysis result
// that describes buffer aliasing, together with a target-specific size function
// that maps a tensor's logical size to its padded size. peak_memory (may be
// nullptr) is set to the peak memory of the resulting schedule according to the
// HeapSimulator.
//
// TODO(yunxing): Cleanup usage of TuplePointsToAnalysis.
typedef std::function<StatusOr<HloInstructionSequence>(
    HloComputation*, const TuplePointsToAnalysis&, const HloAliasAnalysis&,
    const LogicalBuffer::SizeFunction&,
    const absl::flat_hash_map<const HloComputation*, int64>&,
    /*peak_memory*/ int64*)>
    MemorySchedulerAlgorithm;

// Scheduler for the entire module.
typedef std::function<StatusOr<HloSchedule>(
    HloModule*, const TuplePointsToAnalysis&, const HloAliasAnalysis&,
    const LogicalBuffer::SizeFunction&,
    /*peak_memory*/ int64*)>
    ModuleSchedulerAlgorithm;

// Lift a computation scheduler into a module scheduler by calling the
// computation scheduler on all computations in a module.
ModuleSchedulerAlgorithm ComputationSchedulerToModuleScheduler(
    const MemorySchedulerAlgorithm&);

// List scheduler
StatusOr<HloInstructionSequence> ListMemoryScheduler(
    HloComputation* computation,
    const TuplePointsToAnalysis& points_to_analysis,
    const HloAliasAnalysis& alias_analysis,
    const LogicalBuffer::SizeFunction& size_function,
    const absl::flat_hash_map<const HloComputation*, int64>&
        memory_by_computation,
    int64* peak_memory);

// DFS-order scheduler
StatusOr<HloInstructionSequence> DFSMemoryScheduler(
    HloComputation* computation,
    const TuplePointsToAnalysis& points_to_analysis,
    const HloAliasAnalysis& alias_analysis,
    const LogicalBuffer::SizeFunction& size_function,
    const absl::flat_hash_map<const HloComputation*, int64>&
        memory_by_computation,
    int64* peak_memory);

// Naive Post Order scheduler
StatusOr<HloInstructionSequence> PostOrderMemoryScheduler(
    HloComputation* computation,
    const TuplePointsToAnalysis& points_to_analysis,
    const HloAliasAnalysis& alias_analysis,
    const LogicalBuffer::SizeFunction& size_function,
    const absl::flat_hash_map<const HloComputation*, int64>&
        memory_by_computation,
    int64* peak_memory);

// The default scheduling algorithm. Runs the list scheduler, the DFS scheduler,
// and the post-order scheduler and chooses whichever returns a lower min-
// memory, not accounting for fragmentation. peak_memory (may be nullptr) is set
// to the peak memory of the resulting schedule according to the HeapSimulator.
StatusOr<HloInstructionSequence> DefaultMemoryScheduler(
    HloComputation* computation,
    const TuplePointsToAnalysis& points_to_analysis,
    const HloAliasAnalysis& alias_analysis,
    const LogicalBuffer::SizeFunction& size_function,
    const absl::flat_hash_map<const HloComputation*, int64>&
        memory_by_computation,
    int64* peak_memory);

StatusOr<HloSchedule> DefaultModuleScheduler(
    HloModule* module, const TuplePointsToAnalysis& points_to_analysis,
    const HloAliasAnalysis& alias_analysis,
    const LogicalBuffer::SizeFunction& size_function, int64* peak_memory);

// Returns an HloSchedule which seeks to minimize the memory required for the
// module. size_function is the function returning the number of bytes required
// for a LogicalBuffer. peak_memory (if not nullptr) is set to the largest peak
// memory (according to the HeapSimulator) of all computations in the module.
StatusOr<HloSchedule> ScheduleModule(
    HloModule* module, const LogicalBuffer::SizeFunction& size_function,
    const ModuleSchedulerAlgorithm& algorithm = {},
    int64* peak_memory = nullptr);

// Computes the schedule for a single computation.
// Currently only used by the GPU backend.
StatusOr<HloInstructionSequence> ScheduleComputation(
    HloComputation* computation,
    const LogicalBuffer::SizeFunction& size_function);

// A pass which schedules the HLO instructions in a module. The HloModule's
// schedule field is set to the resulting HloSchedule using
// HloModule::set_schedule.
class HloMemoryScheduler : public HloModulePass {
 public:
  // size_function is the function returning the number of bytes required for a
  // LogicalBuffer. algorithm is the memory scheduling algorithm to use. If not
  // specified, then DefaultMemoryScheduler is used.
  HloMemoryScheduler(const LogicalBuffer::SizeFunction& size_function,
                     const ModuleSchedulerAlgorithm& algorithm = {});

  ~HloMemoryScheduler() override = default;

  absl::string_view name() const override { return "hlo-memory-scheduler"; }

  StatusOr<bool> Run(HloModule* module) override;

 private:
  LogicalBuffer::SizeFunction size_function_;

  ModuleSchedulerAlgorithm algorithm_;
};

// A pass which produces a naive, but correct schedule. The schedule is produced
// using a DFS traversal of the graph with no attempt to minimize memory use.
class HloTrivialScheduler : public HloModulePass {
 public:
  absl::string_view name() const override { return "hlo-trivial-scheduler"; }

  StatusOr<bool> Run(HloModule* module) override;
};

// A trivial pass which clears the schedule currently set on the
// HloModule. After this pass runs HloModule::has_schedule will return false.
class HloDescheduler : public HloModulePass {
 public:
  HloDescheduler() = default;
  ~HloDescheduler() override = default;
  absl::string_view name() const override { return "hlo-descheduler"; }

  StatusOr<bool> Run(HloModule* module) override;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_HLO_MEMORY_SCHEDULER_H_
