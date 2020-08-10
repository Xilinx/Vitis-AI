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
#include "tensorflow/compiler/xla/service/gpu/gpu_debug_info_manager.h"

#include "tensorflow/compiler/xla/service/buffer_assignment.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"

namespace xla {
namespace gpu {

using ::testing::UnorderedElementsAre;

class GpuDebugInfoManagerTest : public HloTestBase {
 protected:
  struct DebugMetadata {
    // We allow same id to be registered multiple times. we need unique id to
    // know which program is referenced (such as in UnregisterProgram).
    int unique_id;
    string id;
    std::shared_ptr<HloModule> module;
    std::shared_ptr<BufferAssignment> buffer_assignment;
  };

  // Return unique id of this module.
  int RegisterProgram(const string& module_id) {
    DebugMetadata debug_info;
    HloModuleConfig config;
    debug_info.unique_id = ++serial_;
    debug_info.id = module_id;
    debug_info.module = std::make_shared<HloModule>(module_id, config);
    debug_info.buffer_assignment = nullptr;
    gpu_debug_info_manager_.RegisterModule(module_id, debug_info.module,
                                           debug_info.buffer_assignment);
    external_references_.push_back(std::move(debug_info));
    return serial_;
  }

  void UnregisterProgram(int unique_id) {
    for (int i = 0; i < external_references_.size(); i++) {
      if (external_references_[i].unique_id == unique_id) {
        gpu_debug_info_manager_.UnregisterModule(
            external_references_[i].id, external_references_[i].module,
            external_references_[i].buffer_assignment);
        external_references_.erase(external_references_.begin() + i);
        break;
      }
    }
  }

  void StartProgram(int unique_id) {
    for (int i = 0; i < external_references_.size(); i++) {
      if (external_references_[i].unique_id == unique_id) {
        gpu_debug_info_manager_.OnModuleStart(external_references_[i].id);
        break;
      }
    }
  }

  void StopProgram(int unique_id) {
    for (int i = 0; i < external_references_.size(); i++) {
      if (external_references_[i].unique_id == unique_id) {
        gpu_debug_info_manager_.OnModuleStop(external_references_[i].id);
        break;
      }
    }
  }

  void StartAndStopProgram(int unique_id) {
    StartProgram(unique_id);
    StopProgram(unique_id);
  }

  std::set<ModuleIdentifier> GetRunningModule() {
    return gpu_debug_info_manager_.GetRunningModules();
  }
  std::set<ModuleIdentifier> GetActiveModule() {
    return gpu_debug_info_manager_.GetActiveModules();
  }

  void StartTrace() { gpu_debug_info_manager_.StartTracing(); }

  std::set<ModuleIdentifier> StopTrace() {
    std::vector<GpuModuleDebugInfo> module_debug_info;
    gpu_debug_info_manager_.StopTracing(&module_debug_info);
    std::set<ModuleIdentifier> serialized;
    for (const auto& module : module_debug_info) {
      serialized.insert(module.module_id);
    }
    return serialized;
  }

  int serial_ = 0;

  // Simulation of compilation cache.
  std::vector<DebugMetadata> external_references_;

  // Use an instance per test instead of singleton to avoid interferences.
  GpuDebugInfoManager gpu_debug_info_manager_;
};

// Test the cases where no trace session is involved.
TEST_F(GpuDebugInfoManagerTest, NoTraceBasic) {
  auto program0 = RegisterProgram("program0");
  EXPECT_THAT(GetActiveModule(), UnorderedElementsAre("program0"));
  EXPECT_TRUE(GetRunningModule().empty());

  auto program1 = RegisterProgram("program1");
  EXPECT_THAT(GetActiveModule(), UnorderedElementsAre("program0", "program1"));

  StartAndStopProgram(program0);
  EXPECT_TRUE(GetRunningModule().empty());
  StartProgram(program0);
  EXPECT_THAT(GetRunningModule(), UnorderedElementsAre("program0"));
  StopProgram(program0);
  UnregisterProgram(program0);
  EXPECT_THAT(GetActiveModule(), UnorderedElementsAre("program1"));
  StartAndStopProgram(program1);
  EXPECT_TRUE(GetRunningModule().empty());
  StartProgram(program1);
  EXPECT_THAT(GetRunningModule(), UnorderedElementsAre("program1"));
  StopProgram(program1);
  UnregisterProgram(program1);
  EXPECT_TRUE(GetActiveModule().empty());
}

TEST_F(GpuDebugInfoManagerTest, NoTraceDuplicateIds) {
  auto program0A = RegisterProgram("program0");
  auto program0B = RegisterProgram("program0");  // duplicates
  auto program1 = RegisterProgram("program1");
  EXPECT_THAT(GetActiveModule(), UnorderedElementsAre("program0", "program1"));

  StartProgram(program0A);
  EXPECT_THAT(GetRunningModule(), UnorderedElementsAre("program0"));
  StartProgram(program0B);
  EXPECT_THAT(GetRunningModule(), UnorderedElementsAre("program0"));
  StartProgram(program1);
  EXPECT_THAT(GetRunningModule(), UnorderedElementsAre("program0", "program1"));
  StopProgram(program0A);
  EXPECT_THAT(GetRunningModule(), UnorderedElementsAre("program0", "program1"));
  StopProgram(program0B);
  EXPECT_THAT(GetRunningModule(), UnorderedElementsAre("program1"));
  StopProgram(program1);
  EXPECT_TRUE(GetRunningModule().empty());

  UnregisterProgram(program1);
  EXPECT_THAT(GetActiveModule(), UnorderedElementsAre("program0"));
  UnregisterProgram(program0A);
  EXPECT_THAT(GetActiveModule(), UnorderedElementsAre("program0"));
  UnregisterProgram(program0B);
  EXPECT_TRUE(GetActiveModule().empty());
}

// Test the cases where an active trace session is involved.
TEST_F(GpuDebugInfoManagerTest, ActiveTrace) {
  auto program0A = RegisterProgram("program0");
  auto program0B = RegisterProgram("program0");  // duplicates
  auto program1 = RegisterProgram("program1");

  // Case 1: Trace starts when no program is running.
  StartAndStopProgram(program0A);
  StartTrace();
  StartAndStopProgram(program1);
  auto program2 = RegisterProgram("program2");
  StartAndStopProgram(program0B);
  EXPECT_THAT(StopTrace(), UnorderedElementsAre("program0", "program1"));

  // Case 1: Trace starts during program is running.
  StartProgram(program0A);
  StartTrace();
  StopProgram(program0A);
  StartAndStopProgram(program1);
  EXPECT_THAT(StopTrace(), UnorderedElementsAre("program0", "program1"));
  EXPECT_THAT(GetActiveModule(),
              UnorderedElementsAre("program0", "program1", "program2"));

  UnregisterProgram(program2);
  EXPECT_THAT(GetActiveModule(), UnorderedElementsAre("program0", "program1"));
  UnregisterProgram(program0A);
  EXPECT_THAT(GetActiveModule(), UnorderedElementsAre("program0", "program1"));
  UnregisterProgram(program0B);
  EXPECT_THAT(GetActiveModule(), UnorderedElementsAre("program1"));
  UnregisterProgram(program1);
  EXPECT_TRUE(GetActiveModule().empty());
}

TEST_F(GpuDebugInfoManagerTest, UnregisterDuringTrace) {
  auto program0A = RegisterProgram("program0");
  auto program0B = RegisterProgram("program0");  // duplicates
  auto program1 = RegisterProgram("program1");

  StartTrace();
  StartAndStopProgram(program1);
  EXPECT_THAT(GetRunningModule(), UnorderedElementsAre("program1"));
  UnregisterProgram(program1);
  UnregisterProgram(program0B);
  EXPECT_THAT(StopTrace(), UnorderedElementsAre("program1"));
  EXPECT_THAT(GetActiveModule(), UnorderedElementsAre("program0"));

  UnregisterProgram(program0A);
}

}  // namespace gpu
}  // namespace xla
