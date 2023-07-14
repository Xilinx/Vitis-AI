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
#include <glog/logging.h>
#include <functional>
#include <map>
#include <memory>
#include <vitis/ai/env_config.hpp>
#include "dpu_session.hpp"

DEF_ENV_PARAM(DEBUG_DPU_SESSION, "0");
namespace vart {
namespace dpu {
std::unique_ptr<DpuSession> DpuSession::create(const std::string& filename,
                                               const std::string& kernel) {
  return DpuSession::create0(filename, kernel);
}
std::unique_ptr<DpuSession> DpuSession::create(const xir::Subgraph* subgraph,
                                               xir::Attrs* attrs) {
  return DpuSession::create0(subgraph, attrs);
}
}  // namespace dpu
}  // namespace vart
