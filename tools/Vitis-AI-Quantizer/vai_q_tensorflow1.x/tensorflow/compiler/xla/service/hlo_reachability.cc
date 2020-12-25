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

#include <queue>

#include "tensorflow/compiler/xla/service/hlo_reachability.h"

namespace xla {

HloReachabilityMap::HloReachabilityMap(
    absl::Span<const HloInstruction* const> instructions)
    : size_(instructions.size()) {
  bit_vectors_.reserve(size_);
  for (const HloInstruction* hlo : instructions) {
    indices_[GetKey(hlo)] = bit_vectors_.size();
    bit_vectors_.emplace_back(size_);
  }
  CHECK_EQ(size_, indices_.size());  // instructions should be unique
}

bool HloReachabilityMap::SetReachabilityToUnion(
    absl::Span<const HloInstruction* const> inputs,
    const HloInstruction* instruction) {
  BitVector& bit_vector = GetBitVector(instruction);
  tmp_bit_vector_ = bit_vector;
  SetReachabilityToUnionHelper(inputs, instruction, &bit_vector);
  return bit_vector != tmp_bit_vector_;
}

void HloReachabilityMap::FastSetReachabilityToUnion(
    absl::Span<const HloInstruction* const> inputs,
    const HloInstruction* instruction) {
  SetReachabilityToUnionHelper(inputs, instruction, &GetBitVector(instruction));
}

void HloReachabilityMap::SetReachabilityToUnionHelper(
    absl::Span<const HloInstruction* const> inputs,
    const HloInstruction* instruction, BitVector* bit_vector) {
  // If instruction is part of inputs, don't reset the bit_vector.
  if (!absl::c_linear_search(inputs, instruction)) {
    bit_vector->SetToZero();
  }
  bit_vector->Set(GetIndex(instruction));
  for (const HloInstruction* input : inputs) {
    if (input != instruction) {
      bit_vector->OrWith(GetBitVector(input));
    }
  }
}

void HloReachabilityMap::Replace(const HloInstruction* original,
                                 const HloInstruction* replacement) {
  if (GetKey(original) == GetKey(replacement)) {
    return;
  }
  indices_[GetKey(replacement)] = GetIndex(original);
  indices_.erase(GetKey(original));
}

void HloReachabilityMap::SetReachable(const HloInstruction* a,
                                      const HloInstruction* b) {
  GetBitVector(b).Set(GetIndex(a));
}

bool HloReachabilityMap::IsReachable(const HloInstruction* a,
                                     const HloInstruction* b) const {
  return GetBitVector(b).Get(GetIndex(a));
}

bool HloReachabilityMap::IsConnected(const HloInstruction* a,
                                     const HloInstruction* b) const {
  return IsReachable(a, b) || IsReachable(b, a);
}

std::unique_ptr<HloReachabilityMap> HloReachabilityMap::Build(
    const HloComputation* computation) {
  const auto& all = computation->MakeInstructionPostOrder();
  auto result = absl::make_unique<HloReachabilityMap>(all);
  auto channel_group = computation->ComputeChannelDependencies();

  for (const HloInstruction* hlo : all) {
    std::vector<HloInstruction*> inputs;
    const auto add_input = [&channel_group, &inputs](HloInstruction* input) {
      inputs.push_back(input);
      if (input->opcode() == HloOpcode::kAllReduce && input->channel_id()) {
        auto it = channel_group.find(*input->channel_id());
        if (it != channel_group.end()) {
          inputs.insert(inputs.end(), it->second.begin(), it->second.end());
        }
      }
    };

    const auto add_dependencies = [&add_input](const HloInstruction* hlo) {
      for (HloInstruction* operand : hlo->operands()) {
        add_input(operand);
      }
      for (HloInstruction* predecessor : hlo->control_predecessors()) {
        add_input(predecessor);
      }
    };

    add_dependencies(hlo);

    switch (hlo->opcode()) {
      case HloOpcode::kRecvDone: {
        auto it = channel_group.find(*hlo->channel_id());
        if (it != channel_group.end()) {
          for (HloInstruction* channel : it->second) {
            if (channel->opcode() == HloOpcode::kSend) {
              add_input(channel);
            }
          }
        }
        break;
      }
      case HloOpcode::kAllReduce: {
        auto channel_id = hlo->channel_id();
        if (channel_id) {
          auto it = channel_group.find(channel_id.value());
          if (it != channel_group.end()) {
            for (HloInstruction* all_reduce : it->second) {
              add_dependencies(all_reduce);
            }
          }
        }
        break;
      }
      default:
        break;
    }

    result->FastSetReachabilityToUnion(inputs, hlo);
  }
  return result;
}

void HloReachabilityMap::UpdateReachabilityThroughInstruction(
    const HloInstruction* instruction) {
  std::queue<const HloInstruction*> worklist;
  worklist.push(instruction);

  std::vector<HloInstruction*> inputs;

  while (!worklist.empty()) {
    const HloInstruction* item = worklist.front();
    worklist.pop();

    inputs.assign(item->operands().begin(), item->operands().end());
    inputs.insert(inputs.end(), item->control_predecessors().begin(),
                  item->control_predecessors().end());

    if (SetReachabilityToUnion(inputs, item)) {
      // Add immediate successors to worklist.
      for (const HloInstruction* user : item->users()) {
        worklist.push(user);
      }
      for (const HloInstruction* succ : item->control_successors()) {
        worklist.push(succ);
      }
    }
  }
}

}  // namespace xla
