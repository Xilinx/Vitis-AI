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

#include "tensorflow/compiler/xla/service/all_reduce_simplifier.h"

#include <vector>

#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/hlo_replication_analysis.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/statusor.h"

namespace xla {

StatusOr<bool> AllReduceSimplifier::Run(HloModule* module) {
  TF_ASSIGN_OR_RETURN(auto replication, HloReplicationAnalysis::Run(module));
  std::vector<HloInstruction*> all_reduces_to_replace;
  for (auto computation : module->computations()) {
    for (HloInstruction* inst : computation->MakeInstructionPostOrder()) {
      if (!inst->shape().IsArray()) {
        // We currently do not change tuple-shaped all-reduce.
        continue;
      }
      if (inst->IsCrossReplicaAllReduce() &&
          replication->HloInstructionIsReplicatedAt(inst->operand(0), {})) {
        all_reduces_to_replace.push_back(inst);
      }
    }
  }

  bool changed = false;
  if (all_reduces_to_replace.empty()) {
    return changed;
  }

  // Returns the size of a replica group if all groups have the same size, or -1
  // if they have different sizes.
  auto get_replica_group_size =
      [this](const HloInstruction* all_reduce) -> int64 {
    if (all_reduce->replica_groups().empty()) {
      return replica_count_;
    }
    int64 replica_group_size = -1;
    for (const auto& group : all_reduce->replica_groups()) {
      if (replica_group_size == -1) {
        replica_group_size = group.replica_ids_size();
      } else if (replica_group_size != group.replica_ids_size()) {
        return -1;
      }
    }
    return replica_group_size;
  };

  for (auto all_reduce : all_reduces_to_replace) {
    if (all_reduce->to_apply()->instruction_count() != 3 ||
        all_reduce->to_apply()->num_parameters() != 2) {
      continue;
    }
    HloInstruction* replacement;
    switch (all_reduce->to_apply()->root_instruction()->opcode()) {
      case HloOpcode::kAdd: {
        int64 replica_group_size = get_replica_group_size(all_reduce);
        if (replica_group_size == -1) {
          continue;
        }
        // Create the multiplier:
        //   broadcast(convert_to_matching_type(s32 group size))
        auto multiplier =
            all_reduce->parent()->AddInstruction(HloInstruction::CreateConstant(
                LiteralUtil::CreateR0<int32>(replica_group_size)));
        if (all_reduce->shape().element_type() != S32) {
          multiplier = all_reduce->parent()->AddInstruction(
              HloInstruction::CreateConvert(
                  ShapeUtil::ChangeElementType(
                      multiplier->shape(), all_reduce->shape().element_type()),
                  multiplier));
        }
        if (all_reduce->shape().rank() > 0) {
          multiplier = all_reduce->parent()->AddInstruction(
              HloInstruction::CreateBroadcast(all_reduce->shape(), multiplier,
                                              {}));
        }
        replacement =
            all_reduce->parent()->AddInstruction(HloInstruction::CreateBinary(
                all_reduce->shape(), HloOpcode::kMultiply,
                all_reduce->mutable_operand(0), multiplier));
        break;
      }
      case HloOpcode::kMinimum:
      case HloOpcode::kMaximum:
      case HloOpcode::kOr:
      case HloOpcode::kAnd:
        replacement = all_reduce->mutable_operand(0);
        break;
      default:
        continue;
    }
    VLOG(2) << "Replacing " << all_reduce->ToString() << " with "
            << replacement->ToString();
    TF_RETURN_IF_ERROR(all_reduce->ReplaceAllUsesWith(replacement));
    changed = true;
  }
  return changed;
}

}  // namespace xla
