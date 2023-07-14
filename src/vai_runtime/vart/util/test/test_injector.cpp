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
using namespace std;
#include "device_scheduler.hpp"
int main(int argc, char* argv[]) {  //
  cout << "HELLO" << endl;
  auto x1 = vart::dpu::DeviceScheduler::create(4);
  auto x2 = vart::dpu::DeviceScheduler::create((const char*)"hello", 4);
  // auto x3 = vart::dpu::DeviceScheduler::create(
  //     vart::dpu::DeviceScheduler::so_name_t{"test_injector_lib"},
  //     (const char*)"hello", 4);
  std::cout << "DONE" << vart::dpu::hello << 
  std::endl;
  for (auto i = 0; i < 40; ++i) {
    auto v1 = x1->next();
    auto v2 = x2->next();
    auto v3 = x2->next();
    std::cout << "next = "
              << "{" << v1 << "," << v2 << "," << v3 << "}" << std::endl;
  }
  // testing for initialize once;
  /*
  using T = vart::dpu::DeviceScheduler;
  auto value = std::is_base_of<vitis::ai::WithInjection<T>, T>::value;
  std::cerr << __FILE__ << ":" << __LINE__ << ": [" << __FUNCTION__ << "]"  //
            << "value " << value << " "                                     //
            << std::endl;
  auto x4 = vitis::ai::WeakStore<int, vart::dpu::DeviceScheduler>::create(0, 4);
  std::cout << "x4 " << x4->next() << std::endl;
  */
  return 0;
}
