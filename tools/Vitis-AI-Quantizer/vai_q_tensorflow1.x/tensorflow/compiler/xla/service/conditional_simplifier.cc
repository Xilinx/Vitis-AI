/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/conditional_simplifier.h"

#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/service/call_graph.h"
#include "tensorflow/compiler/xla/service/call_inliner.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/errors.h"

namespace xla {

namespace {
// Tries to replace a conditional with a call operation of the corresponding
// computation. If the given conditional has a constant branch_index, tries to
// replace it with a call to its corresponding branch computation and then
// inline that computation.
//
// Returns true if it made a change to the graph.
StatusOr<bool> TryRemoveConditional(HloInstruction* conditional) {
  CHECK_EQ(conditional->opcode(), HloOpcode::kConditional);
  // Do not remove conditionals that contain side-effecting instructions or
  // have control predecessors/successors in either true/false computation.
  if (!conditional->parent()->IsSafelyRemovable(conditional) ||
      conditional->HasSideEffect()) {
    VLOG(2) << "Not attempting to remove conditional as it is not removable or "
               "has side effect: "
            << conditional->ToShortString();
    return false;
  }

  // We can always inline a 1-branch conditional due to default branch fallback.
  auto computation = conditional->parent();
  auto create_call = [&](int64 branch) {
    auto call = computation->AddInstruction(HloInstruction::CreateCall(
        conditional->shape(), {conditional->mutable_operand(1 + branch)},
        conditional->branch_computation(branch)));
    conditional->SetupDerivedInstruction(call);
    return call;
  };

  if (conditional->branch_count() == 1) {
    HloInstruction* call_op = create_call(0);
    TF_RETURN_IF_ERROR(computation->ReplaceInstruction(conditional, call_op));
    TF_RETURN_IF_ERROR(CallInliner::Inline(call_op).status());
    return true;
  }

  if (conditional->operand(0)->opcode() == HloOpcode::kConstant) {
    int branch_index = 0;
    if (conditional->operand(0)->shape().element_type() == PRED) {
      branch_index = conditional->operand(0)->literal().Get<bool>({}) ? 0 : 1;
    } else {
      branch_index = conditional->operand(0)->literal().Get<int32>({});
      if (branch_index < 0 || branch_index >= conditional->branch_count()) {
        branch_index = conditional->branch_count() - 1;
      }
    }
    HloInstruction* call_op = create_call(branch_index);
    TF_RETURN_IF_ERROR(computation->ReplaceInstruction(conditional, call_op));
    TF_RETURN_IF_ERROR(CallInliner::Inline(call_op).status());

    return true;
  }

  auto instruction_is_expensive = [](const HloInstruction* hlo) {
    switch (hlo->opcode()) {
      case HloOpcode::kBroadcast:
      case HloOpcode::kConcatenate:
      case HloOpcode::kDynamicSlice:
      case HloOpcode::kDynamicUpdateSlice:
      case HloOpcode::kGetTupleElement:
      case HloOpcode::kReduce:
      case HloOpcode::kReshape:
      case HloOpcode::kPad:
      case HloOpcode::kParameter:
      case HloOpcode::kSlice:
      case HloOpcode::kTuple:
        return false;
      default:
        return !hlo->IsElementwise();
    }
  };

  if (conditional->branch_count() != 2 ||
      conditional->operand(0)->shape().element_type() != PRED ||
      absl::c_any_of(conditional->branch_computation(0)->instructions(),
                     instruction_is_expensive) ||
      absl::c_any_of(conditional->branch_computation(1)->instructions(),
                     instruction_is_expensive)) {
    VLOG(2)
        << "Not attempting  to remove conditional as its branch_index is not a "
           "compile-time constant or contains expensive instructions: "
        << conditional->ToShortString();
    return false;
  }

  HloInstruction* true_call_op = create_call(0);
  HloInstruction* false_call_op = create_call(1);
  auto condition_broadcast = [&](const Shape& shape) {
    if (ShapeUtil::IsScalar(shape)) {
      return conditional->mutable_operand(0);
    }
    return computation->AddInstruction(HloInstruction::CreateBroadcast(
        ShapeUtil::ChangeElementType(shape, PRED),
        conditional->mutable_operand(0), {}));
  };

  auto gte = [&](HloInstruction* hlo, int64 i) {
    return computation->AddInstruction(HloInstruction::CreateGetTupleElement(
        hlo->shape().tuple_shapes(i), hlo, i));
  };
  std::function<HloInstruction*(HloInstruction*, HloInstruction*)> select =
      [&](HloInstruction* t, HloInstruction* f) {
        if (f->shape().IsArray()) {
          return computation->AddInstruction(HloInstruction::CreateTernary(
              f->shape(), HloOpcode::kSelect, condition_broadcast(f->shape()),
              t, f));
        }
        std::vector<HloInstruction*> selects;
        const int64 tuple_element_count =
            ShapeUtil::TupleElementCount(f->shape());
        selects.reserve(tuple_element_count);
        for (int64 i = 0; i < tuple_element_count; ++i) {
          selects.push_back(select(gte(t, i), gte(f, i)));
        }
        return computation->AddInstruction(
            HloInstruction::CreateTuple(selects));
      };

  TF_RETURN_IF_ERROR(computation->ReplaceInstruction(
      conditional, select(true_call_op, false_call_op)));

  TF_RETURN_IF_ERROR(CallInliner::Inline(false_call_op).status());
  TF_RETURN_IF_ERROR(CallInliner::Inline(true_call_op).status());
  return true;
}
StatusOr<bool> TryRemoveUnusedConditionalOperands(
    HloInstruction* conditional,
    std::map<HloComputation*, std::set<int64>>* changed_computations) {
  // Avoid dealing with sharding.
  if (conditional->has_sharding()) {
    return false;
  }
  std::vector<std::set<int64>> tuple_indices_to_keep(
      conditional->branch_count());
  bool will_change = false;
  for (int64 i = 0; i < conditional->branch_count(); ++i) {
    HloComputation* computation = conditional->branch_computation(i);
    if (changed_computations->count(computation) > 0) {
      will_change = true;
      break;
    }
    HloInstruction* param = computation->parameter_instruction(0);
    // Do not remove the root instruction.
    if (param == computation->root_instruction()) {
      return false;
    }
    // There is nothing to be removed for non-tuple operands.
    if (!param->shape().IsTuple()) {
      return false;
    }
    for (HloInstruction* user : param->users()) {
      // If the user is not a get tuple element, assume it is unsafe to remove
      // elemnts from the tuple.
      if (user->opcode() != HloOpcode::kGetTupleElement) {
        return false;
      }
      tuple_indices_to_keep[i].insert(user->tuple_index());
    }
    // If not all tuple elements are used in this conditional branch, some can
    // removed from the computation.
    if (tuple_indices_to_keep[i].size() !=
        ShapeUtil::TupleElementCount(param->shape())) {
      will_change = true;
    }
  }

  if (!will_change) {
    return false;
  }

  for (int64 branch = 0; branch < conditional->branch_count(); ++branch) {
    const Shape& old_shape = conditional->operand(branch + 1)->shape();
    int64 old_tuple_element_count = ShapeUtil::TupleElementCount(old_shape);
    // Clone the computation in case it is called by another instruction.
    HloComputation* computation = conditional->branch_computation(branch);
    if (changed_computations
            ->insert({computation, tuple_indices_to_keep[branch]})
            .second) {
      HloInstruction* param = computation->parameter_instruction(0);

      // Create a new tuple shape based on the indices actually used by this
      // branch.
      std::vector<Shape> new_tuple_shapes;
      new_tuple_shapes.reserve(tuple_indices_to_keep[branch].size());
      std::vector<int64> map(old_tuple_element_count, -1);
      for (int64 i : tuple_indices_to_keep[branch]) {
        map[i] = new_tuple_shapes.size();
        new_tuple_shapes.push_back(old_shape.tuple_shapes(i));
      }
      Shape tuple_shape = ShapeUtil::MakeTupleShape(new_tuple_shapes);
      // Reset the parameter shape of the computation.
      *param->mutable_shape() = tuple_shape;

      // Reroute the GTE instructions to new tuple indices.
      for (HloInstruction* user : param->users()) {
        user->set_tuple_index(map[user->tuple_index()]);
      }
    }

    // Reroute the operand tuple through a tuple of gte instructions of the
    // original operand tuple.
    const auto& to_keep = (*changed_computations)[computation];
    std::vector<HloInstruction*> new_tuple_operands;
    new_tuple_operands.reserve(to_keep.size());
    for (int64 i : to_keep) {
      new_tuple_operands.push_back(conditional->parent()->AddInstruction(
          HloInstruction::CreateGetTupleElement(
              old_shape.tuple_shapes(i),
              conditional->mutable_operand(branch + 1), i)));
    }
    HloInstruction* new_tuple = conditional->parent()->AddInstruction(
        HloInstruction::CreateTuple(new_tuple_operands));
    TF_RETURN_IF_ERROR(
        conditional->ReplaceOperandWithDifferentShape(branch + 1, new_tuple));
  }
  return true;
}

// Replaces the roots of all branches with an empty tuple if the conditional op
// has no users. Returns if anything is changed.
bool ReplaceRootWithEmptyTupleIfNoUsers(HloInstruction* conditional_op) {
  const Shape empty_tuple = ShapeUtil::MakeTupleShape({});
  if (conditional_op->user_count() == 0 &&
      conditional_op != conditional_op->parent()->root_instruction() &&
      !ShapeUtil::Compatible(empty_tuple, conditional_op->shape())) {
    for (int64 branch_id = 0; branch_id < conditional_op->branch_count();
         ++branch_id) {
      auto branch_computation =
          conditional_op->GetModule()->AddEmbeddedComputation(
              conditional_op->branch_computation(branch_id)->Clone());
      conditional_op->set_branch_computation(branch_id, branch_computation);
      auto new_empty_root =
          branch_computation->AddInstruction(HloInstruction::CreateTuple({}));
      branch_computation->set_root_instruction(new_empty_root,
                                               /*accept_different_shape=*/true);
    }
    *conditional_op->mutable_shape() = empty_tuple;
    return true;
  }
  return false;
}

}  // namespace

StatusOr<bool> ConditionalSimplifier::Run(HloModule* module) {
  XLA_VLOG_LINES(
      3, "ConditionalSimplifier::Run(), before:\n" + module->ToString());
  bool changed = false;

  // Gather all the conditional ops in our module. We do this ahead of time so
  // we don't have to worry about mutating the lists of computations or
  // instructions as we iterate.
  std::vector<HloInstruction*> conditional_ops;
  for (auto* comp : module->computations()) {
    for (auto* instr : comp->MakeInstructionPostOrder()) {
      if (instr->opcode() == HloOpcode::kConditional) {
        conditional_ops.push_back(instr);
      }
    }
  }

  std::map<HloComputation*, std::set<int64>> changed_computations;
  for (HloInstruction* conditional_op : conditional_ops) {
    changed |= ReplaceRootWithEmptyTupleIfNoUsers(conditional_op);
    TF_ASSIGN_OR_RETURN(bool result, TryRemoveConditional(conditional_op));
    if (!result) {
      TF_ASSIGN_OR_RETURN(result, TryRemoveUnusedConditionalOperands(
                                      conditional_op, &changed_computations));
    }
    changed |= result;
  }

  XLA_VLOG_LINES(3,
                 "ConditionalSimplifier::Run(), after:\n" + module->ToString());
  return changed;
}

}  // namespace xla
