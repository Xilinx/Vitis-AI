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

#include "tensorflow/compiler/xla/service/ar_crs_combiner.h"

#include <string>
#include <utility>
#include <vector>

#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/call_graph.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/types.h"

namespace xla {

namespace m = match;

// Checks if the argument instruction is an AllReduce, followed by a certain
// sequence of instructions and then a CRS. It must be possible to move
// the AR past each instruction in the sequence.
absl::optional<ArCrsCombiner::ArCrsPair> ArCrsCombiner::MatchesArCrsPattern(
    HloInstruction* instruction) {
  auto can_ar_move_past_instruction = [](HloInstruction* instruction) -> bool {
    if (instruction->user_count() != 1) {
      return false;
    }
    switch (instruction->opcode()) {
      case HloOpcode::kBitcast:
      case HloOpcode::kTranspose:
      case HloOpcode::kReshape:
        return true;
      case HloOpcode::kConvert:
        // Can be moved across if both input and output is either float or
        // integer (e.g. S32<->U32 or F32<->BF16)
        return ShapeUtil::ElementIsFloating(instruction->shape()) ==
               ShapeUtil::ElementIsFloating(instruction->operand(0)->shape());
      case HloOpcode::kAdd:
      case HloOpcode::kSubtract:
      case HloOpcode::kMultiply:
        // Only supported for floating point operands.
        return ShapeUtil::ElementIsFloating(instruction->shape());
      default:
        return false;
    }
  };

  auto computation_is_addition = [](HloComputation* c) {
    return c->instruction_count() == 3 &&
           Match(c->root_instruction(), m::Add(m::Parameter(), m::Parameter()));
  };

  // We only support combining cross-partition all-reduce where each replica
  // belongs to its own group, since the later cross-replica all-reduce combines
  // along the replica dimension.
  if (instruction->IsCrossModuleAllReduce() &&
      instruction->replica_groups().size() == num_replicas_ &&
      computation_is_addition(instruction->called_computations()[0]) &&
      instruction->user_count() == 1) {
    auto next = instruction->users()[0];
    int64 distance = 1;
    while (!next->IsCrossReplicaAllReduce()) {
      if (can_ar_move_past_instruction(next)) {
        next = next->users()[0];
      } else {
        return absl::nullopt;
      }
      ++distance;
    }
    if (!Cast<HloAllReduceInstruction>(next)->IsNoop() &&
        computation_is_addition(next->called_computations()[0])) {
      ArCrsPair pair(instruction, next, distance);
      VLOG(2) << "ArCrsPair matching pattern: " << pair.ToString();
      return pair;
    }
  }
  return absl::nullopt;
}

absl::optional<HloInstruction*> ArCrsCombiner::WhileFromBodyParameter(
    HloInstruction* instruction) {
  CHECK_EQ(HloOpcode::kParameter, instruction->opcode());
  HloComputation* computation = instruction->parent();
  auto caller_instructions = call_graph_->GetComputationCallers(computation);
  if (caller_instructions.size() == 1) {
    auto caller_instruction = caller_instructions[0];
    if (caller_instruction->opcode() == HloOpcode::kWhile) {
      return caller_instruction;
    }
  }
  return absl::nullopt;
}

absl::optional<HloInstruction*> ArCrsCombiner::ConditionalFromBodyParameter(
    HloInstruction* instruction) {
  CHECK_EQ(HloOpcode::kParameter, instruction->opcode());
  HloComputation* computation = instruction->parent();
  auto caller_instructions = call_graph_->GetComputationCallers(computation);
  if (caller_instructions.size() == 1) {
    auto caller_instruction = caller_instructions[0];
    if (caller_instruction->opcode() == HloOpcode::kConditional) {
      return caller_instruction;
    }
  }
  return absl::nullopt;
}

absl::optional<std::vector<HloInstruction*>> ArCrsCombiner::GetAllTuples(
    HloInstruction* instruction,
    absl::flat_hash_set<HloInstruction*>* visited) {
  if (visited->find(instruction) != visited->end()) {
    return std::vector<HloInstruction*>();
  }
  visited->insert(instruction);

  switch (instruction->opcode()) {
    case HloOpcode::kTuple: {
      return std::vector<HloInstruction*>({instruction});
    }
    case HloOpcode::kDomain: {
      return GetAllTuples(instruction->operands()[0], visited);
    }
    case HloOpcode::kParameter: {
      auto maybe_while = WhileFromBodyParameter(instruction);
      if (maybe_while) {
        auto while_instr = *maybe_while;
        auto init_tuples = GetAllTuples(while_instr->while_init(), visited);
        auto body_tuples = GetAllTuples(
            while_instr->while_body()->root_instruction(), visited);
        if (!init_tuples || !body_tuples) {
          return absl::nullopt;
        }
        auto result = *init_tuples;
        result.insert(result.end(), body_tuples->begin(), body_tuples->end());
        return result;
      }
      auto maybe_conditional = ConditionalFromBodyParameter(instruction);
      if (maybe_conditional) {
        auto cond_instr = *maybe_conditional;
        std::vector<HloInstruction*> tuples;
        for (int64 i = 0; i < cond_instr->branch_computations().size(); ++i) {
          if (cond_instr->branch_computation(i)->parameter_instruction(0) ==
              instruction) {
            // If the same computation is used for more than one branch of the
            // conditional, we collect the arguments that flow to the
            // computation from all branches.
            auto branch_tuples =
                GetAllTuples(cond_instr->mutable_operand(i + 1), visited);
            if (!branch_tuples) {
              return absl::nullopt;
            }
            tuples.insert(tuples.end(), branch_tuples->begin(),
                          branch_tuples->end());
          }
        }
        return tuples;
      }
      return absl::nullopt;
    }
    case HloOpcode::kGetTupleElement: {
      std::vector<HloInstruction*> result_tuples;
      auto tuples = GetAllTuples(instruction->operands()[0], visited);
      if (!tuples) {
        return absl::nullopt;
      }
      for (auto tuple : *tuples) {
        auto tmp_tuples = GetAllTuples(
            tuple->mutable_operand(instruction->tuple_index()), visited);
        if (!tmp_tuples) {
          return absl::nullopt;
        }
        result_tuples.insert(result_tuples.end(), tmp_tuples->begin(),
                             tmp_tuples->end());
      }
      return result_tuples;
    }
    case HloOpcode::kConditional: {
      std::vector<HloInstruction*> result_tuples;
      for (HloComputation* body : instruction->branch_computations()) {
        if (body->root_instruction()->opcode() != HloOpcode::kTuple) {
          return absl::nullopt;
        }
        result_tuples.push_back(body->root_instruction());
      }
      return result_tuples;
    }
    case HloOpcode::kWhile: {
      auto init_tuples = GetAllTuples(instruction->while_init(), visited);
      auto body_tuples =
          GetAllTuples(instruction->while_body()->root_instruction(), visited);
      if (!init_tuples || !body_tuples) {
        return absl::nullopt;
      }
      auto result = *init_tuples;
      result.insert(result.end(), body_tuples->begin(), body_tuples->end());
      return result;
    }
    default:
      return absl::nullopt;
  }
}

bool ArCrsCombiner::TupleElementsComputeSameValue(
    HloInstruction* tuple_shaped_instruction, int64 i1, int64 i2,
    absl::flat_hash_map<int64, int64>* visited_pairs) {
  absl::flat_hash_set<HloInstruction*> visited;
  auto tuples = GetAllTuples(tuple_shaped_instruction, &visited);
  if (!tuples) {
    return false;
  }
  for (auto tuple : *tuples) {
    CHECK_EQ(tuple->opcode(), HloOpcode::kTuple);
    if (!InstructionsComputeSameValue(tuple->mutable_operand(i1),
                                      tuple->mutable_operand(i2),
                                      visited_pairs)) {
      return false;
    }
  }
  return true;
}

/* static */
bool ArCrsCombiner::TestInstructionsComputeSameValue(HloInstruction* i1,
                                                     HloInstruction* i2) {
  ArCrsCombiner combiner(/*num_spatial_partitions=*/2, /*num_replicas=*/1);
  auto module = i1->parent()->parent();
  CHECK_EQ(module, i2->parent()->parent());
  combiner.call_graph_ = CallGraph::Build(module);
  absl::flat_hash_map<int64, int64> visited_pairs;
  return combiner.InstructionsComputeSameValue(i1, i2, &visited_pairs);
}

bool ArCrsCombiner::InstructionsComputeSameValue(
    HloInstruction* i1, HloInstruction* i2,
    absl::flat_hash_map<int64, int64>* visited_pairs) {
  if (i1 == i2) {
    return true;
  }
  auto uid1 = i1->unique_id();
  auto uid2 = i2->unique_id();
  auto min_uid = std::min(uid1, uid2);
  auto max_uid = std::max(uid1, uid2);
  auto it = visited_pairs->find(min_uid);
  if (it != visited_pairs->end() && max_uid == it->second) {
    return true;
  }
  auto opcode1 = i1->opcode();
  auto operands1 = i1->operands();
  if (opcode1 != i2->opcode() || operands1.size() != i2->operands().size()) {
    return false;
  }
  auto eq_computations = [](const HloComputation* a, const HloComputation* b) {
    return *a == *b;
  };
  if (i1->IsCrossModuleAllReduce()) {
    return i1->Identical(*i2,
                         /*eq_operands=*/std::equal_to<const HloInstruction*>(),
                         eq_computations,
                         /*layout_sensitive=*/false);
  }
  visited_pairs->emplace(min_uid, max_uid);
  for (int i = 0; i < operands1.size(); ++i) {
    auto operand1 = operands1[i];
    auto operand2 = i2->operands()[i];
    if (!InstructionsComputeSameValue(operand1, operand2, visited_pairs)) {
      return false;
    }
  }
  if (opcode1 == HloOpcode::kParameter) {
    // In the general case, we don't try to prove equality of parameters.
    // We only try in the context of get-tuple-element
    // (see TupleElementsComputeSameValue).
    return false;
  }
  if (opcode1 == HloOpcode::kGetTupleElement) {
    return i1->tuple_index() == i2->tuple_index() ||
           TupleElementsComputeSameValue(operands1[0], i1->tuple_index(),
                                         i2->tuple_index(), visited_pairs);
  }
  // Don't check that the operands are identical, because Identical can
  // return false for instructions that compute the same value but are not
  // identical, which we don't want. We have checked the arguments with
  // InstructionsComputeSameValue earlier.
  auto eq_instructions = [](const HloInstruction* i1,
                            const HloInstruction* i2) -> bool { return true; };
  return i1->Identical(*i2, eq_instructions, eq_computations,
                       /*layout_sensitive=*/false);
}

void ArCrsCombiner::GroupAllReducesById(HloModule* module) {
  // Say that two or more ARs lead to the same CRS: (AR1, CRS), (AR2, CRS),
  // ... , (ARn, CRS).
  // If as we traverse the HLO graph we start tracking the pair (AR2, CRS),
  // and later find that AR1's distance from the CRS is longer, we discard
  // AR2 and start tracking AR1. We put the discarded ids in this set, in order
  // to skip processing of short paths when we encounter the other ARs that
  // have the same id as AR2.
  absl::flat_hash_set<int64> discarded_ar_ids;
  for (HloComputation* computation : module->MakeNonfusionComputations()) {
    for (HloInstruction* instruction : computation->instructions()) {
      auto maybe_pair = MatchesArCrsPattern(instruction);
      if (maybe_pair) {
        auto pair = *maybe_pair;
        int64 ar_id = *(instruction->channel_id());
        if (discarded_ar_ids.find(ar_id) != discarded_ar_ids.end()) {
          continue;
        }
        auto it = crs_reserved_map_.find(pair.crs);
        if (it != crs_reserved_map_.end()) {
          auto prev_ar_id = it->second;
          // Since there is another AR paired with CRS,
          // all_reduce_map_[prev_ar_id] should exist, but
          // all_reduce_map_[ar_id] shouldn't.
          CHECK(all_reduce_map_.find(ar_id) == all_reduce_map_.end());
          CHECK_NE(prev_ar_id, ar_id);
          auto prev_pair = all_reduce_map_[prev_ar_id].back();
          int64 prev_distance = prev_pair.distance;
          if (prev_distance < pair.distance) {
            // The current AR's distance to CRS is longer than the previously
            // tracked AR, so we discard the previous AR.
            VLOG(2) << "Replacing ArCrsPair: " << prev_pair.ToString()
                    << " with ArCrsPair: " << pair.ToString();
            all_reduce_map_.erase(prev_ar_id);
            discarded_ar_ids.insert(prev_ar_id);
            all_reduce_map_[ar_id].push_back(pair);
            crs_reserved_map_[pair.crs] = ar_id;
          } else {
            // Discard the current AR id because we are keeping the previously
            // tracked AR.
            discarded_ar_ids.insert(ar_id);
          }
        } else {
          if (all_reduce_map_.find(ar_id) != all_reduce_map_.end()) {
            int64 prev_distance = all_reduce_map_[ar_id].back().distance;
            CHECK_EQ(prev_distance, pair.distance)
                << "All ARs with the same AR ID must have the same distance "
                   "from the corresponding CRSs. Found: "
                << prev_distance << " and " << pair.distance;
          }
          all_reduce_map_[ar_id].push_back(pair);
          crs_reserved_map_[pair.crs] = ar_id;
        }
      }
    }
  }
}

void ArCrsCombiner::KeepProvablyEqualInstructionGroups() {
  for (auto it : all_reduce_map_) {
    auto channel_id = it.first;
    VLOG(2)
        << "KeepProvablyEqualInstructionGroups. Checking AllReduce channel id: "
        << channel_id << "\n";
    auto pairs_vec = it.second;
    CHECK_EQ(pairs_vec.size(), num_spatial_partitions_);
    auto instr_0 = pairs_vec[0].ar;
    for (int i = 1; i < pairs_vec.size(); ++i) {
      auto instr_i = pairs_vec[i].ar;
      auto next_0 = instr_0->users()[0];
      auto next_i = instr_i->users()[0];
      absl::flat_hash_map<int64, int64> visited_pairs;
      while (true) {
        if (!InstructionsComputeSameValue(next_0, next_i, &visited_pairs)) {
          all_reduce_map_.erase(channel_id);
          VLOG(2) << "KeepProvablyEqualInstructionGroups. Erased AllReduce "
                     "channel id: "
                  << channel_id << "\n";
          break;
        }
        if (next_0->IsCrossReplicaAllReduce()) {
          break;
        }
        next_0 = next_0->users()[0];
        next_i = next_i->users()[0];
      }
    }
  }
}

StatusOr<bool> ArCrsCombiner::RewriteGraph() {
  if (all_reduce_map_.empty()) {
    return false;
  }
  for (auto it : all_reduce_map_) {
    auto pairs_vec = it.second;
    for (auto pair : pairs_vec) {
      auto all_reduce = pair.ar;
      auto parent_computation = all_reduce->parent();
      auto channel_id = all_reduce->channel_id();
      auto prev = all_reduce->mutable_operand(0);
      auto next = all_reduce->users()[0];
      TF_CHECK_OK(all_reduce->ReplaceUseWith(next, prev));
      TF_CHECK_OK(parent_computation->RemoveInstruction(all_reduce));
      while (!next->IsCrossReplicaAllReduce()) {
        switch (next->opcode()) {
          case HloOpcode::kBitcast:
          case HloOpcode::kTranspose:
          case HloOpcode::kReshape:
          case HloOpcode::kConvert:
          case HloOpcode::kMultiply:
            break;
          case HloOpcode::kAdd:
          case HloOpcode::kSubtract: {
            auto other_operand = (next->operands()[0] == prev)
                                     ? next->operands()[1]
                                     : next->operands()[0];
            // To move the AR past the addition/subtraction, we need to divide
            // other_operand by the number of spatial partitions, except if
            // other_operand is a cross-module AR, which can be eliminated.
            if (other_operand->IsCrossModuleAllReduce() &&
                other_operand->user_count() == 1) {
              TF_CHECK_OK(other_operand->ReplaceAllUsesWith(
                  other_operand->mutable_operand(0)));
            } else {
              auto shape = other_operand->shape();
              Literal lit(shape);
              lit.PopulateWithValue<float>(num_spatial_partitions_);
              auto divisor = parent_computation->AddInstruction(
                  HloInstruction::CreateConstant(lit.Clone()));
              auto division = parent_computation->AddInstruction(
                  HloInstruction::CreateBinary(shape, HloOpcode::kDivide,
                                               other_operand, divisor));
              TF_CHECK_OK(other_operand->ReplaceUseWith(next, division));
            }
            break;
          }
          default:
            LOG(FATAL) << "Unexpected instruction: " << next->ToShortString();
        }
        prev = next;
        next = next->users()[0];
      }
      // The AllReduce and the CRS are combined to an all-core AllReduce.
      next->set_channel_id(channel_id);
    }
  }
  return true;
}

StatusOr<bool> ArCrsCombiner::Run(HloModule* module) {
  call_graph_ = CallGraph::Build(module);

  GroupAllReducesById(module);

  KeepProvablyEqualInstructionGroups();

  return RewriteGraph();
}

}  // namespace xla
