/*
 * Copyright 2021 Xilinx Inc.
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
#include "xrnn_controller.hpp"

#include <glog/logging.h>
#include <sys/mman.h>

#include <chrono>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <thread>

#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

namespace vart {
namespace xrnn {

std::map<std::string, GenFunc>& RnnControllerCreate::getRegistry() {
  static std::map<std::string, GenFunc> registry;
  return registry;
}

void RnnControllerCreate::Register(std::string device_name, GenFunc func) {
  LOG_IF(INFO, ENV_PARAM(DEBUG_XRNN_CONTROLLER))
      << "Registering : " << device_name << std::endl;
  getRegistry()[device_name] = func;
}

std::unique_ptr<XrnnController> RnnControllerCreate::Create(
    std::string device_name, unsigned int device_core_id, std::string device) {
  auto& registry = getRegistry();
  if (registry.find(device_name) != registry.end()) {
    return registry[device_name](device_core_id, device);
  } else {  // TODO : abidk : Throw proper error here.
    LOG_IF(INFO, ENV_PARAM(DEBUG_XRNN_CONTROLLER))
        << "Couldn't find : " << device_name << std::endl
        << "Available Devices : ";
    for (auto& item : registry) {
      LOG_IF(INFO, ENV_PARAM(DEBUG_XRNN_CONTROLLER)) << item.first << std::endl;
    }
    return nullptr;
  }
}

}  // namespace xrnn
}  // namespace vart
