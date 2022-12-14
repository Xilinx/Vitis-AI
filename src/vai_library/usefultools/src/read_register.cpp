/*
 * Copyright 2019 Xilinx Inc.
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

#include <vitis/ai/env_config.hpp>
#include <vitis/ai/profiling.hpp>

#include "tools_extra_ops.hpp"

DEF_ENV_PARAM(DEBUG_TOOLS, "0");

#ifdef ENABLE_XRT
#include <iomanip>
#include <iostream>
#include <xir/xrt_device_handle.hpp>

#include "parse_value.hpp"
#include "xrt_xcl_read.hpp"

uint32_t get_reg(xclDeviceHandle xcl_handle, uint32_t ip_index,
                 uint64_t cu_base_addr, uint32_t offset) {
  uint32_t value = 0;
  auto read_result =
      xrtXclRead(xcl_handle, ip_index, offset, cu_base_addr, &value);

  CHECK_EQ(read_result, 0) << "xrtXclRead has error!";
  return value;
}

std::vector<uint32_t> read_register(void* handle, uint32_t ip_index,
                                    uint64_t cu_base_addr,
                                    const std::vector<uint32_t>& addrs) {
  __TIC__(READ_REGISTER)

  std::vector<uint32_t> values;
  for (auto addr : addrs) {
    values.push_back(get_reg(handle, ip_index, cu_base_addr, addr));
  }
  __TOC__(READ_REGISTER)
  return values;
}
#else
std::vector<uint32_t> read_register(void* handle, uint32_t ip_index,
                                    uint64_t cu_base_addr,
                                    const std::vector<uint32_t>& addrs) {
  LOG(INFO) << "xrt not found ";
  return std::vector<uint32_t>();
}
#endif
