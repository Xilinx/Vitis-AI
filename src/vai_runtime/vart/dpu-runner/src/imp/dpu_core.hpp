/*
 * Copyright 2022-2023 Advanced Micro Devices Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once
#include <memory>
#include <mutex>
#include <array>

#include "../dpu_reg.hpp"
#include "hbm_config.hpp"
#include "hbm_manager.hpp"

namespace vart {
namespace dpu {

class DpuCoreWorkspace;
class DpuCore {
 public:
  explicit DpuCore(size_t core_id);
  DpuCore(const DpuCore&) = delete;
  DpuCore& operator=(const DpuCore& other) = delete;

  virtual ~DpuCore();

 public:
  std::unique_ptr<DpuCoreWorkspace> lock_workspace();
  size_t get_num_of_engines() const { return num_of_engines_; }

 private:
  size_t core_id_;
  std::array<std::mutex, 2> workspace_mutex_;
  std::mutex core_mutex_;
  // nullptr mean it is not in use
  // workspace_chunks_[workspace_id][engine_id]
  std::array<std::vector<HbmChunk>, 2> workspace_chunks_;

  volatile int next_workspace_;
  friend class DpuCoreWorkspace;
  size_t num_of_engines_;
};

class DpuCoreWorkspace {
 public:
  explicit DpuCoreWorkspace(std::unique_lock<std::mutex>&& lock,
                            size_t current_workspace, DpuCore* core);
  DpuCoreWorkspace(const DpuCoreWorkspace&) = delete;
  DpuCoreWorkspace& operator=(const DpuCoreWorkspace& other) = delete;
  virtual ~DpuCoreWorkspace() = default;

 public:
  std::unique_lock<std::mutex> lock_core();
  // reg id as subscribe
  // sizes[reg_id] = 0 means not in use
  // return_value[engine_id][reg_id]
  std::vector<std::map<std::string, std::unique_ptr<HbmChunk>>> get_workspaces(
      const std::vector<DpuReg>& sizes);

  size_t get_workspace_id() const { return current_workspace_; }

 private:
  // use unique_lock rather than unique_lock because unique_lock is not movable.
  std::unique_lock<std::mutex> lock_;
  const size_t current_workspace_;

  DpuCore* core_;
};
}  // namespace dpu
}  // namespace vart
