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
#include "dpu_core.hpp"

#include <glog/logging.h>

#include <cstring>
#include <iostream>
#include <vitis/ai/env_config.hpp>
#include <vitis/ai/profiling.hpp>
#include <vitis/ai/xxd.hpp>
#include <UniLog/UniLog.hpp>

#include "dpu_cloud.hpp"
#include "hbm_config.hpp"
DEF_ENV_PARAM(DEBUG_DPU_RUNNER, "0");
DEF_ENV_PARAM(DEBUG_DPU_CONTROLLER, "0")
DEF_ENV_PARAM(XLNX_SHOW_DPU_COUNTER, "0")
DEF_ENV_PARAM(XLNX_DPU_TIMEOUT, "10000")
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

namespace vart {
namespace dpu {
// ret[workspace_id][engine_id]
static std::vector<HbmChunk> init_workspace(size_t core_id,
                                            size_t workspace_id) {
  // hbms[engine_id] is a chunk_def_t
  const auto& hbms = get_engine_hbm(core_id);
  auto ret = std::vector<HbmChunk>();
  auto num_of_engines = hbms.size();
  ret.reserve(num_of_engines);
  uint64_t total = 0;
  for (auto engine_id = 0u; engine_id < num_of_engines; ++engine_id) {
    auto& arg = hbms[engine_id];
    // swap A/B workspace
    UNI_LOG_CHECK(arg.size() == 1u, VART_OUT_OF_RANGE)
        << "for workspace, only single hbm channel supported yet.";
    LOG_IF(INFO, ENV_PARAM(DEBUG_DPU_RUNNER)) << " arg = " << arg;
    auto hbm_ch = arg[0];
    auto workspace_size = hbm_ch.capacity / 2;
    auto workspace_base = hbm_ch.offset + workspace_id * workspace_size;
    LOG_IF(INFO, ENV_PARAM(DEBUG_DPU_RUNNER))
        << "initialize workspace[" << workspace_id << "] for core[" << core_id
        << "] size = " << std::hex << "0x" << workspace_size << " base= 0x"
        << workspace_base << std::dec;
    ret.emplace_back(workspace_base, workspace_size);
    total = total + workspace_size;
  }
  UNI_LOG_CHECK(total > 0u, VART_SIZE_MISMATCH)
    << "workspace size for engine must not empty. core_id="
    << core_id << " workspace id = " << workspace_id;
  return ret;
}

DpuCore::DpuCore(size_t core_id)
    : core_id_{core_id},
      workspace_mutex_{},
      core_mutex_{},
      workspace_chunks_{
          init_workspace(core_id, 0 /* workspace_id */),
          init_workspace(
              core_id,
              1 /* workspace_id */)},  // workspace_chunks_[workspace_id][engine_id]
      next_workspace_{0u},
      num_of_engines_{workspace_chunks_[0].size()} {
  UNI_LOG_CHECK(num_of_engines_ == workspace_chunks_[1].size(), VART_SIZE_MISMATCH);
}

DpuCore::~DpuCore() {
  LOG_IF(INFO, ENV_PARAM(DEBUG_DPU_RUNNER))
      << "destroying dpu core[" << core_id_ << "] @" << (void*)this;
}

std::unique_ptr<DpuCoreWorkspace> DpuCore::lock_workspace() {
  LOG_IF(INFO, ENV_PARAM(DEBUG_DPU_RUNNER))
      << "try locking dpu core[" << core_id_ << "] @ " << (void*)this << " "
      << "next_workspace_ " << next_workspace_ << " ";

  std::unique_lock<std::mutex> lock(workspace_mutex_[next_workspace_]);
  auto current_workspace = next_workspace_;
  next_workspace_ = (next_workspace_ + 1) % workspace_mutex_.size();
  LOG_IF(INFO, ENV_PARAM(DEBUG_DPU_RUNNER))
      << "current_workspace " << current_workspace << " "  //
      << "next_workspace_ " << next_workspace_ << " "      //
      ;
  return std::make_unique<DpuCoreWorkspace>(std::move(lock), current_workspace,
                                            this);
}
DpuCoreWorkspace::DpuCoreWorkspace(std::unique_lock<std::mutex>&& lock,
                                   size_t current_workspace, DpuCore* core)
    : lock_{move(lock)},                      //
      current_workspace_{current_workspace},  //
      core_{core} {}

std::unique_lock<std::mutex> DpuCoreWorkspace::lock_core() {
  return std::unique_lock<std ::mutex>(core_->core_mutex_);
}
static inline uint64_t align(uint64_t a, uint64_t b) {
  return (a / b + (a % b ? 1 : 0)) * b;
}

// ret[engine_id][reg_id]
std::vector<std::map<std::string, std::unique_ptr<HbmChunk>>>
DpuCoreWorkspace::get_workspaces(const std::vector<DpuReg>& regs) {
  auto& workspaces = core_->workspace_chunks_[current_workspace_];
  auto num_of_engines = workspaces.size();
  auto ret = std::vector<std::map<std::string, std::unique_ptr<HbmChunk>>>(
      num_of_engines);
  for (size_t engine_id = 0; engine_id < num_of_engines; ++engine_id) {
    const auto& workspace = workspaces[engine_id];
    auto& engine_regs = ret[engine_id];
    size_t reg_offset = 0ull;
    for (const auto& reg : regs) {
      const auto& reg_id = reg.name_;
      engine_regs.insert(std::make_pair(
          reg_id,
          std::make_unique<HbmChunk>(workspace.get_offset() + reg_offset,  //
                                     reg.size_)));
      UNI_LOG_CHECK(reg.size_ < workspace.get_capacity(), VART_OUT_OF_RANGE)
          << "out of HBM memory"
          << ";reg.name=" << reg.name_;
      reg_offset = reg_offset + align(reg.size_, _4K);
    }
  }
  return ret;
}
}  // namespace dpu
}  // namespace vart
