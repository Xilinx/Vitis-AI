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
#include "device_memory_cloud.hpp"

#include <glog/logging.h>

#include "vitis/ai/env_config.hpp"
DEF_ENV_PARAM(DEBUG_DEVICE_MEMORY, "0");
namespace {
DeviceMemoryCloud::DeviceMemoryCloud(size_t device_id)
    : device_id_{device_id},  //
      handle_{xclOpen(device_id_, NULL, XCL_INFO)} {
  LOG_IF(INFO, ENV_PARAM(DEBUG_DEVICE_MEMORY))
      << "device_id_ " << device_id_ << " "  //
      << "handle_ " << handle_ << " "        //
      << std::endl;
}
DeviceMemoryCloud::~DeviceMemoryCloud() {
  LOG_IF(INFO, ENV_PARAM(DEBUG_DEVICE_MEMORY))
      << "close the handle " << handle_;
  xclClose(handle_);
}

bool DeviceMemoryCloud::upload(const void* data, uint64_t offset, size_t size) {
  LOG_IF(INFO, ENV_PARAM(DEBUG_DEVICE_MEMORY)) << "data " << data << " "      //
                                               << "offset " << offset << " "  //
                                               << "size " << size << " "      //
      ;
  auto flags = 0;
  auto ok = xclUnmgdPwrite(handle_, flags, data, size, offset);
  PCHECK(ok == 0) << " upload data has error! ";
  return ok == 0;
}
bool DeviceMemoryCloud::download(void* data, uint64_t offset, size_t size) {
  LOG_IF(INFO, ENV_PARAM(DEBUG_DEVICE_MEMORY)) << "data " << data << " "      //
                                               << "offset " << offset << " "  //
                                               << "size " << size << " "      //
      ;
  auto flags = 0;
  auto ok = xclUnmgdPread(handle_, flags, data, size, offset);
  PCHECK(ok == 0) << " download data has error!";
  return ok == 0;
}

}  // namespace

REGISTER_INJECTION_BEGIN(xir::DeviceMemory, 1, DeviceMemoryCloud, size_t&) {
  return true;
}
REGISTER_INJECTION_END
