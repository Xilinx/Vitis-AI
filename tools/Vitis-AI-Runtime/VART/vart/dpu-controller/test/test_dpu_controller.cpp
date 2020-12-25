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
#include <xrt.h>

#include <iostream>

#include "vitis/ai/parse_value.hpp"
#include "vitis/ai/profiling.hpp"
#include "xir/dpu_controller.hpp"
#include "xir/xrt_device_handle.hpp"
using namespace std;
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

static uint32_t get_reg(xclDeviceHandle xcl_handle, uint64_t cu_addr) {
  uint32_t value = 0;
  size_t size = sizeof(value);
  auto read_result =
      xclRead(xcl_handle, XCL_ADDR_KERNEL_CTRL, cu_addr, &value, size);
  CHECK_EQ(read_result, size)
      << "xclRead has error!"                              //
      << "read_result " << read_result << " "              //
      << "cu_addr " << std::hex << "0x" << cu_addr << " "  //
      ;
  return value;
}

int main(int argc, char* argv[]) {
  auto h = xir::XrtDeviceHandle::get_instance();

  auto dpu = xir::DpuController::get_instance();
  auto w = 0ul;
  vitis::ai::parse_value(argv[1], w);
  auto p = 0ul;
  vitis::ai::parse_value(argv[2], p);
  auto c = 0ul;
  vitis::ai::parse_value(argv[3], c);
  auto idx = 0ul;
  vitis::ai::parse_value(argv[4], idx);
  auto count = std::stoi(argv[5]);

  //
  auto cu_name = std::string("dpu_xrt_top");
  auto handle = h->get_handle(cu_name, idx);
  auto cu_addr = h->get_cu_addr(cu_name, idx);
  for (auto i = 0; i < count; ++i) {
    auto start = std::chrono::steady_clock::now();
    auto start_cycle = get_reg(handle, cu_addr + 0x1a0);
    dpu->run(idx, c, {p, w});
    auto end = std::chrono::steady_clock::now();
    auto end_cycle = get_reg(handle, cu_addr + 0x1a0);
    auto ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start)
                  .count();
    auto cycle = end_cycle - start_cycle;
    cout << ms << " " << start_cycle << " " << end_cycle << " " << cycle
         << "\n";
  }
  return 0;
}
