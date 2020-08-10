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

#include "tensorflow/compiler/xla/service/hlo_replication_analysis.h"

#include <memory>
#include <string>

#include "absl/memory/memory.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {
namespace {

class HloReplicationAnalysisTest : public HloTestBase {};

TEST_F(HloReplicationAnalysisTest, NoControlFlow) {
  const string module_str = R"(
HloModule NoControlFlow

sum {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT add.2 = f32[] add(a, b)
}

ENTRY entry {
  param = (f32[4096,4096]{1,0}, f32[4096,4096]{1,0}) parameter(0)
  get-tuple-element.2 = f32[4096,4096]{1,0} get-tuple-element(param), index=0
  get-tuple-element.3 = f32[4096,4096]{1,0} get-tuple-element(param), index=1
  after-all.1 = token[] after-all()
  infeed = (f32[4096,4096]{1,0}, token[]) infeed(after-all.1)
  get-tuple-element.5 = f32[4096,4096]{1,0} get-tuple-element(infeed), index=0
  dot = f32[4096,4096]{1,0} dot(get-tuple-element.5, get-tuple-element.3), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  all-reduce = f32[4096,4096]{1,0} all-reduce(dot), replica_groups={}, to_apply=sum
  subtract = f32[4096,4096]{1,0} subtract(get-tuple-element.3, all-reduce)
  ROOT add = f32[4096,4096]{1,0} add(get-tuple-element.2, subtract)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule(module_str));
  auto param = module->entry_computation()->parameter_instruction(0);
  param->set_parameter_replicated_at_leaf_buffers(
      absl::Span<const bool>{false, true});
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloReplicationAnalysis> analysis,
                          HloReplicationAnalysis::Run(module.get()));
  EXPECT_FALSE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "get-tuple-element.2"), {}));
  EXPECT_TRUE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "get-tuple-element.3"), {}));
  EXPECT_FALSE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "get-tuple-element.5"), {}));
  EXPECT_FALSE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "dot"), {}));
  EXPECT_TRUE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "all-reduce"), {}));
  EXPECT_TRUE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "subtract"), {}));
  EXPECT_FALSE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "add"), {}));
}

TEST_F(HloReplicationAnalysisTest, NestedCall) {
  const string module_str = R"(
HloModule NestedCall

fusion_computation {
  fusion_p0 = f32[] parameter(0)
  fusion_p1 = f32[] parameter(1)
  add = f32[] add(fusion_p0, fusion_p0)
  multiply = f32[] multiply(add, fusion_p1)
  ROOT tuple = (f32[], f32[]) tuple(add, multiply)
}

call_body {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT fusion = (f32[], f32[]) fusion(a, b), kind=kLoop, calls=fusion_computation
}

ENTRY entry {
  param = (f32[], f32[]) parameter(0)
  get-tuple-element = f32[] get-tuple-element(param), index=0
  get-tuple-element.1 = f32[] get-tuple-element(param), index=1
  ROOT call = (f32[], f32[]) call(get-tuple-element, get-tuple-element.1), to_apply=call_body
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule(module_str));
  auto param = module->entry_computation()->parameter_instruction(0);
  param->set_parameter_replicated_at_leaf_buffers(
      absl::Span<const bool>{true, false});
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloReplicationAnalysis> analysis,
                          HloReplicationAnalysis::Run(module.get()));
  EXPECT_TRUE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "get-tuple-element"), {}));
  EXPECT_FALSE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "get-tuple-element.1"), {}));
  EXPECT_TRUE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "add"), {}));
  EXPECT_FALSE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "multiply"), {}));
  EXPECT_TRUE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "fusion"), {0}));
  EXPECT_FALSE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "fusion"), {1}));
  EXPECT_TRUE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "call"), {0}));
  EXPECT_FALSE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "call"), {1}));
}

TEST_F(HloReplicationAnalysisTest, SimpleWhileLoop) {
  const string module_str = R"(
HloModule SimpleWhileLoop

cond {
  cond_param = (f32[4096,4096]{1,0}, u32[]) parameter(0)
  get-tuple-element = u32[] get-tuple-element(cond_param), index=1
  constant.3 = u32[] constant(5)
  ROOT greater-than = pred[] compare(get-tuple-element, constant.3), direction=LT
}

body {
  body_param = (f32[4096,4096]{1,0}, u32[]) parameter(0)
  get-tuple-element.1 = f32[4096,4096]{1,0} get-tuple-element(body_param), index=0
  multiply = f32[4096,4096]{1,0} multiply(get-tuple-element.1, get-tuple-element.1)
  get-tuple-element.6 = u32[] get-tuple-element(body_param), index=1
  constant.1 = u32[] constant(1)
  add = u32[] add(get-tuple-element.6, constant.1)
  ROOT tuple = (f32[4096,4096]{1,0}, u32[]) tuple(multiply, add)
}

ENTRY SimpleWhileLoop {
  param = (f32[4096,4096]{1,0}, u32[]) parameter(0)
  ROOT while = (f32[4096,4096]{1,0}, u32[]) while(param), condition=cond, body=body
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule(module_str));
  auto param = module->entry_computation()->parameter_instruction(0);
  param->set_parameter_replicated_at_leaf_buffers(
      absl::Span<const bool>{true, true});
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloReplicationAnalysis> analysis,
                          HloReplicationAnalysis::Run(module.get()));
  EXPECT_TRUE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "tuple"), {0}));
  EXPECT_TRUE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "tuple"), {1}));
  EXPECT_TRUE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "while"), {0}));
  EXPECT_TRUE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "while"), {1}));
}

TEST_F(HloReplicationAnalysisTest,
       WhileLoopParameterAliasingNonReplicatedOutput) {
  const string module_str = R"(
HloModule WhileLoopParameterAliasingNonReplicatedOutput

cond {
  cond_param = (f32[4096,4096]{1,0}, u32[]) parameter(0)
  get-tuple-element = u32[] get-tuple-element(cond_param), index=1
  constant.3 = u32[] constant(5)
  ROOT greater-than = pred[] compare(get-tuple-element, constant.3), direction=LT
}

body {
  body_param = (f32[4096,4096]{1,0}, u32[]) parameter(0)
  get-tuple-element.1 = f32[4096,4096]{1,0} get-tuple-element(body_param), index=0
  multiply = f32[4096,4096]{1,0} multiply(get-tuple-element.1, get-tuple-element.1)
  after-all.1 = token[] after-all()
  infeed = (f32[4096,4096]{1,0}, token[]) infeed(after-all.1)
  get-tuple-element.5 = f32[4096,4096]{1,0} get-tuple-element(infeed), index=0
  subtract = f32[4096,4096]{1,0} subtract(get-tuple-element.5, multiply)
  get-tuple-element.6 = u32[] get-tuple-element(body_param), index=1
  constant.1 = u32[] constant(1)
  add = u32[] add(get-tuple-element.6, constant.1)
  ROOT tuple = (f32[4096,4096]{1,0}, u32[]) tuple(subtract, add)
}

ENTRY WhileLoopParameterAliasingNonReplicatedOutput {
  param = (f32[4096,4096]{1,0}, u32[]) parameter(0)
  ROOT while = (f32[4096,4096]{1,0}, u32[]) while(param), condition=cond, body=body
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule(module_str));
  auto param = module->entry_computation()->parameter_instruction(0);
  param->set_parameter_replicated_at_leaf_buffers(
      absl::Span<const bool>{true, true});
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloReplicationAnalysis> analysis,
                          HloReplicationAnalysis::Run(module.get()));
  EXPECT_FALSE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "multiply"), {}));
  EXPECT_FALSE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "tuple"), {0}));
  EXPECT_TRUE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "tuple"), {1}));
  EXPECT_FALSE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "while"), {0}));
  EXPECT_TRUE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "while"), {1}));
}

TEST_F(HloReplicationAnalysisTest, WhileLoopDifferentCondition) {
  const string module_str = R"(
HloModule WhileLoopDifferentCondition

cond {
  cond_param = (f32[4096,4096]{1,0}, u32[]) parameter(0)
  get-tuple-element = u32[] get-tuple-element(cond_param), index=1
  constant.3 = u32[] constant(5)
  ROOT greater-than = pred[] compare(get-tuple-element, constant.3), direction=LT
}

body {
  body_param = (f32[4096,4096]{1,0}, u32[]) parameter(0)
  get-tuple-element.1 = f32[4096,4096]{1,0} get-tuple-element(body_param), index=0
  multiply = f32[4096,4096]{1,0} multiply(get-tuple-element.1, get-tuple-element.1)
  get-tuple-element.6 = u32[] get-tuple-element(body_param), index=1
  replica-id = u32[] replica-id()
  add = u32[] add(get-tuple-element.6, replica-id)
  ROOT tuple = (f32[4096,4096]{1,0}, u32[]) tuple(multiply, add)
}

ENTRY WhileLoopDifferentCondition {
  param = (f32[4096,4096]{1,0}, u32[]) parameter(0)
  ROOT while = (f32[4096,4096]{1,0}, u32[]) while(param), condition=cond, body=body
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule(module_str));
  auto param = module->entry_computation()->parameter_instruction(0);
  param->set_parameter_replicated_at_leaf_buffers(
      absl::Span<const bool>{true, true});
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloReplicationAnalysis> analysis,
                          HloReplicationAnalysis::Run(module.get()));
  EXPECT_FALSE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "while"), {0}));
  EXPECT_FALSE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "while"), {1}));
}

TEST_F(HloReplicationAnalysisTest, SimpleConditional) {
  const string module_str = R"(
HloModule SimpleConditional

Negate {
  x = (f32[], f32[]) parameter(0)
  get-tuple-element = f32[] get-tuple-element(x), index=0
  negate = f32[] negate(get-tuple-element)
  get-tuple-element.1 = f32[] get-tuple-element(x), index=1
  negate.1 = f32[] negate(get-tuple-element.1)
  ROOT tuple = (f32[], f32[]) tuple(negate, negate.1)
}

Identity {
  ROOT y = (f32[], f32[]) parameter(0)
}

Floor {
  z = (f32[], f32[]) parameter(0)
  get-tuple-element.2 = f32[] get-tuple-element(z), index=0
  floor = f32[] floor(get-tuple-element.2)
  get-tuple-element.3 = f32[] get-tuple-element(z), index=1
  floor.1 = f32[] floor(get-tuple-element.3)
  ROOT tuple.1 = (f32[], f32[]) tuple(floor, floor.1)
}

ENTRY entry {
  param = ((f32[], f32[]), (f32[], f32[]), (f32[], f32[]), s32[]) parameter(0)
  get-tuple-element.4 = (f32[], f32[]) get-tuple-element(param), index=0
  get-tuple-element.5 = (f32[], f32[]) get-tuple-element(param), index=1
  get-tuple-element.6 = (f32[], f32[]) get-tuple-element(param), index=2
  get-tuple-element.7 = s32[] get-tuple-element(param), index=3
  ROOT conditional = (f32[], f32[]) conditional(get-tuple-element.7, get-tuple-element.4, get-tuple-element.5, get-tuple-element.6), branch_computations={Negate, Identity, Floor}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule(module_str));
  auto param = module->entry_computation()->parameter_instruction(0);
  param->set_parameter_replicated_at_leaf_buffers(
      absl::Span<const bool>{true, true, true, true, false, true, true});
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloReplicationAnalysis> analysis,
                          HloReplicationAnalysis::Run(module.get()));
  EXPECT_TRUE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "tuple"), {0}));
  EXPECT_TRUE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "tuple"), {1}));
  EXPECT_TRUE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "y"), {0}));
  EXPECT_TRUE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "y"), {1}));
  EXPECT_FALSE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "tuple.1"), {0}));
  EXPECT_TRUE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "tuple.1"), {1}));
  EXPECT_FALSE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "conditional"), {0}));
  EXPECT_TRUE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "conditional"), {1}));
}

TEST_F(HloReplicationAnalysisTest, ConditionalWithDifferentPredicates) {
  const string module_str = R"(
HloModule ConditionalWithDifferentPredicates

Negate {
  x = (f32[], f32[]) parameter(0)
  get-tuple-element = f32[] get-tuple-element(x), index=0
  negate = f32[] negate(get-tuple-element)
  get-tuple-element.1 = f32[] get-tuple-element(x), index=1
  negate.1 = f32[] negate(get-tuple-element.1)
  ROOT tuple = (f32[], f32[]) tuple(negate, negate.1)
}

Identity {
  ROOT y = (f32[], f32[]) parameter(0)
}

Floor {
  z = (f32[], f32[]) parameter(0)
  get-tuple-element.2 = f32[] get-tuple-element(z), index=0
  floor = f32[] floor(get-tuple-element.2)
  get-tuple-element.3 = f32[] get-tuple-element(z), index=1
  floor.1 = f32[] floor(get-tuple-element.3)
  ROOT tuple.1 = (f32[], f32[]) tuple(floor, floor.1)
}

ENTRY entry {
  param = ((f32[], f32[]), (f32[], f32[]), (f32[], f32[])) parameter(0)
  get-tuple-element.4 = (f32[], f32[]) get-tuple-element(param), index=0
  get-tuple-element.5 = (f32[], f32[]) get-tuple-element(param), index=1
  get-tuple-element.6 = (f32[], f32[]) get-tuple-element(param), index=2
  replica-id = u32[] replica-id()
  id = s32[] bitcast-convert(replica-id)
  ROOT conditional = (f32[], f32[]) conditional(id, get-tuple-element.4,
    get-tuple-element.5, get-tuple-element.6),
    branch_computations={Negate, Identity, Floor}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule(module_str));
  auto param = module->entry_computation()->parameter_instruction(0);
  param->set_parameter_replicated_at_leaf_buffers(
      absl::Span<const bool>{true, true, true, true, true, true});
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloReplicationAnalysis> analysis,
                          HloReplicationAnalysis::Run(module.get()));
  EXPECT_FALSE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "tuple"), {0}));
  EXPECT_FALSE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "tuple"), {1}));
  EXPECT_FALSE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "y"), {0}));
  EXPECT_FALSE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "y"), {1}));
  EXPECT_FALSE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "tuple.1"), {0}));
  EXPECT_FALSE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "tuple.1"), {1}));
  EXPECT_FALSE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "conditional"), {0}));
  EXPECT_FALSE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "conditional"), {1}));
}

TEST_F(HloReplicationAnalysisTest, SimpleTupleSelect) {
  const string module_str = R"(
HloModule SimpleTupleSelect

ENTRY entry {
  param = ((f32[], f32[]), (f32[], f32[]), pred[]) parameter(0)
  get-tuple-element.4 = (f32[], f32[]) get-tuple-element(param), index=0
  get-tuple-element.5 = (f32[], f32[]) get-tuple-element(param), index=1
  get-tuple-element.6 = pred[] get-tuple-element(param), index=2
  ROOT tuple-select = (f32[], f32[]) tuple-select(get-tuple-element.6, get-tuple-element.4, get-tuple-element.5)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule(module_str));
  auto param = module->entry_computation()->parameter_instruction(0);
  param->set_parameter_replicated_at_leaf_buffers(
      absl::Span<const bool>{true, false, true, true, true});
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloReplicationAnalysis> analysis,
                          HloReplicationAnalysis::Run(module.get()));
  EXPECT_TRUE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "tuple-select"), {0}));
  EXPECT_FALSE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "tuple-select"), {1}));
}

TEST_F(HloReplicationAnalysisTest, TupleSelectWithDifferentPredicates) {
  const string module_str = R"(
HloModule TupleSelectWithDifferentPredicates

ENTRY entry {
  param = ((f32[], f32[]), (f32[], f32[]), pred[]) parameter(0)
  get-tuple-element.4 = (f32[], f32[]) get-tuple-element(param), index=0
  get-tuple-element.5 = (f32[], f32[]) get-tuple-element(param), index=1
  get-tuple-element.6 = pred[] get-tuple-element(param), index=2
  ROOT tuple-select = (f32[], f32[]) tuple-select(get-tuple-element.6, get-tuple-element.4, get-tuple-element.5)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule(module_str));
  auto param = module->entry_computation()->parameter_instruction(0);
  param->set_parameter_replicated_at_leaf_buffers(
      absl::Span<const bool>{true, true, true, true, false});
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloReplicationAnalysis> analysis,
                          HloReplicationAnalysis::Run(module.get()));
  EXPECT_FALSE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "tuple-select"), {0}));
  EXPECT_FALSE(analysis->HloInstructionIsReplicatedAt(
      FindInstruction(module.get(), "tuple-select"), {1}));
}

}  // namespace
}  // namespace xla
