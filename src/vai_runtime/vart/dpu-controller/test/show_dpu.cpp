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

int main(int argc, char* argv[]) {
  auto dpu = xir::DpuController::get_instance();
  auto num = dpu->get_num_of_dpus();
  for (auto i = 0u; i < num; ++i) {
    auto core_id = dpu->get_core_id(i);
    auto device_id = dpu->get_device_id(i);
    auto device_core_id = i;
    auto fingerprint = dpu->get_fingerprint(i);
    auto batch = dpu->get_batch_size(i);
    cout << "device_core_id=" << device_core_id << " "                        //
         << "device= " << device_id                                           //
         << " core = " << core_id                                             //
         << " fingerprint = " << std::hex << "0x" << fingerprint << std::dec  //
         << " batch = " << batch                                              //
         << " full_cu_name=" << dpu->get_full_name(device_core_id) << "\n";
  }
  cout << endl;
}
