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
#pragma once
#include <experimental/xrt-next.h>
#include <xrt.h>

static int xrtXclRead(xclDeviceHandle handle, uint32_t ipIndex, uint64_t offset,
                      uint64_t offsetbase, uint32_t* datap) {
#ifdef ENABLE_CLOUD
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
  auto read_result = xclRead(handle, XCL_ADDR_KERNEL_CTRL, offsetbase + offset,
                             datap, sizeof(uint32_t));
  return (read_result == sizeof(uint32_t)) ? 0 : 1;
#else
  return xclRegRead(handle, ipIndex, offset, datap);
#endif
}
