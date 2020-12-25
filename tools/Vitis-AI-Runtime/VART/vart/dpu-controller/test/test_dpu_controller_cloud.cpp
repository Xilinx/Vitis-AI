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
#include <iostream>

#include "vitis/ai/parse_value.hpp"
#include "vitis/ai/profiling.hpp"
#include "xir/dpu_controller.hpp"

using namespace std;
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

/*static uint32_t get_reg(xclDeviceHandle xcl_handle, uint64_t cu_addr) {
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
*/
int main(int argc, char* argv[]) {
  auto dpu = xir::DpuController::get_instance();
  //
  auto idx = 0ul;
  auto cu_name = std::string("dpu");
  // auto handle = h->get_handle(cu_name, idx);
  // auto cu_addr = h->get_cu_addr(cu_name, idx);
  auto count = 1;
  for (auto i = 0; i < count; ++i) {
    auto start = std::chrono::steady_clock::now();
    // auto start_cycle = get_reg(handle, cu_addr + 0x1a0);
    dpu->run(
        idx, 0x80000000,
        {0x40000000, 0x0,         0x40007000, 0x50000000, 0x0, 0x0, 0x0,
         0x0,  //
         0x40000000, 0x100000000, 0x40007000, 0x50000000, 0x0, 0x0, 0x0,
         0x0,  //
         0x40000000, 0x110000000, 0x40007000, 0x50000000, 0x0, 0x0, 0x0, 0x0});
    auto end = std::chrono::steady_clock::now();
    // auto end_cycle = get_reg(handle, cu_addr + 0x1a0);
    auto ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start)
                  .count();
    std::cout << "ms " << ms << " "  //
              << std::endl;
  }
  return 0;
}
