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
#include "./dpu_control_fake.hpp"
#include "vitis/ai/env_config.hpp"
#include "vitis/ai/weak.hpp"
#include <glog/logging.h>
#include <sstream>
DEF_ENV_PARAM(DISABLE_DPU_CONTROLLER_FAKE, "0");
DEF_ENV_PARAM(DEBUG_DPU_CONTROLLER, "0");
namespace {

DpuControllerFake::DpuControllerFake() : xir::DpuController{} {}
DpuControllerFake::~DpuControllerFake() {}

void DpuControllerFake::run(const uint64_t code,
                            const std::vector<uint64_t> &gen_reg,
                            int device_id /*not used*/) {
  std::ostringstream str;
  str << std::hex;
  for (const auto &v : gen_reg) {
    str << " 0x" << v;
  }
  LOG(INFO) << std::hex                               //
            << "code 0x" << code << " "               //
            << "gen_reg: 0x" << str.str() << std::dec //
      ;
}

static struct Registar {
  Registar() {
    auto disabled = ENV_PARAM(DISABLE_DPU_CONTROLLER_FAKE);
    if (!disabled) {
      xir::DpuController::registar("10_fake", []() {
        return std::shared_ptr<xir::DpuController>(
            vitis::ai::WeakSingleton<DpuControllerFake>::create());
      });
      LOG_IF(INFO, ENV_PARAM(DEBUG_DPU_CONTROLLER))
          << "register the fake dpu controller";
    } else {
      LOG_IF(INFO, ENV_PARAM(DEBUG_DPU_CONTROLLER))
          << "cancel register the fake dpu controller, because "
             "DISABLE_DPU_CONTROLLER_FAKE=1";
    }
  }
} g_registar;

} // namespace
