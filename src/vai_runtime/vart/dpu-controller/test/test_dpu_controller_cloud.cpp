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
#include <iostream>

#include "vitis/ai/parse_value.hpp"
#include "vitis/ai/profiling.hpp"
#include "xir/dpu_controller.hpp"

using namespace std;
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

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
    dpu->run(
        idx, 0x80000000,
        {0x40000000, 0x0,         0x40007000, 0x50000000, 0x0, 0x0, 0x0,
         0x0,  //
         0x40000000, 0x100000000, 0x40007000, 0x50000000, 0x0, 0x0, 0x0,
         0x0,  //
         0x40000000, 0x110000000, 0x40007000, 0x50000000, 0x0, 0x0, 0x0, 0x0});
    auto end = std::chrono::steady_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start)
                  .count();
    std::cout << "ms " << ms << " "  //
              << std::endl;
  }
  return 0;
}
