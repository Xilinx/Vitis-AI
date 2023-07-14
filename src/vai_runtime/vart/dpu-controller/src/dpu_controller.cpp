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
#include "xir/dpu_controller.hpp"

#include <glog/logging.h>

#include <iostream>
#include <map>

#include "vitis/ai/env_config.hpp"
DEF_ENV_PARAM(DEBUG_DPU_CONTROLLER, "0");
DEF_ENV_PARAM_2(DPU_KERNEL_NAME, "unknown", std::string);
DEF_ENV_PARAM_2(DPU_INSTANCE_NAME, "dpu0", std::string);

namespace xir {
static std::map<std::string, std::function<std::shared_ptr<DpuController>()>>& get_factory_methods()
{
    static std::map<std::string, std::function<std::shared_ptr<DpuController>()>>
        the_factory_methods;
    return the_factory_methods;
}

void DpuController::registar(
    const std::string& name,
    std::function<std::shared_ptr<DpuController>()> m) {
  auto it = get_factory_methods().begin();
  auto ok = false;
  std::tie(it, ok) = get_factory_methods().emplace(std::make_pair(name, m));
  LOG_IF(INFO, ENV_PARAM(DEBUG_DPU_CONTROLLER))
      << "add factory method " << name;
  CHECK(ok);
}

bool DpuController::exist_dpu_devices() {
  return get_factory_methods().empty() ? false : true;
}

std::shared_ptr<DpuController> DpuController::get_instance() {
  CHECK(!get_factory_methods().empty());
  auto ret = get_factory_methods().begin()->second();
  // one dpu controllers per sessions
  // each dpu controller has its own xrt_cu
  // xrt_cu shares the xrt_device_handle
  LOG_IF(INFO, ENV_PARAM(DEBUG_DPU_CONTROLLER))
      << "create dpu controller via " << get_factory_methods().begin()->first
      << " ret= " << (void*)ret.get();
  return ret;
}
DpuController::DpuController() {}
DpuController::~DpuController() {}

std::string DpuController::get_full_name(size_t device_core_id) const {
  return get_kernel_name(device_core_id) + ":" +
         get_instance_name(device_core_id);
}
std::string DpuController::get_kernel_name(size_t device_core_id) const {
  return ENV_PARAM(DPU_KERNEL_NAME);
}
std::string DpuController::get_instance_name(size_t device_core_id) const {
  return ENV_PARAM(DPU_INSTANCE_NAME);
}
}  // namespace xir
