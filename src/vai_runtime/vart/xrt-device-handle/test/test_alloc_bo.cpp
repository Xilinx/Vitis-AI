
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
#include <xrt.h>

#include <array>
#include <iostream>
using namespace std;

static uint64_t get_physical_address(const xclDeviceHandle& handle,
                                     xclBufferHandle bo) {
  xclBOProperties p{};
  auto error_code = xclGetBOProperties(handle, bo, &p);
  uint64_t phy = 0u;
  if (error_code != 0) {
    LOG(INFO) << "cannot xclGetBOProperties !";
  }
  phy = error_code == 0 ? p.paddr : -1;
  return phy;
}
int main(int argc, char* argv[]) {
  auto h = xclOpen(0, NULL, XCL_INFO);
  auto bo1 = xclAllocBO(h, 25u * 1024 * 1024, 0, 0);
  LOG(INFO) << "bo " << std::hex << "0x" << bo1 << std::dec;

  auto phy1 = get_physical_address(h, bo1);
  LOG(INFO) << "phy " << std::hex << "0x" << phy1 << std::dec << " ";
  auto bo2 = xclAllocBO(h, 4u * 1024 * 1024, 0, 0);
  LOG(INFO) << "bo " << std::hex << "0x" << bo2 << std::dec;

  auto phy2 = get_physical_address(h, bo2);
  LOG(INFO) << "phy " << std::hex << "0x" << phy2 << std::dec << " ";
  xclFreeBO(h, bo1);
  xclFreeBO(h, bo2);
  xclClose(h);
  return 0;
}
