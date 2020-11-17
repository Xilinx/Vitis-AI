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
#include <vector>

#include "vitis/ai/parse_value.hpp"
#include "xir/buffer_object.hpp"
using namespace std;
#include <vitis/ai/env_config.hpp>
DEF_ENV_PARAM(DEVICE_ID, "0");
DEF_ENV_PARAM_2(CU_NAME, "DPU", std::string);
int main(int argc, char* argv[]) {
  size_t sz = 0ul;
  std::vector<std::unique_ptr<xir::BufferObject>> all;
  for (auto i = 1; i < argc; ++i) {
    vitis::ai::parse_value(std::string{argv[i]}, sz);
    size_t device_id = (size_t)ENV_PARAM(DEVICE_ID);
    const string& cu_name = ENV_PARAM(CU_NAME);
    auto bo = xir::BufferObject::create(sz, device_id, cu_name);
    std::cout << __FILE__ << ":" << __LINE__ << ": [" << __FUNCTION__ << "]"  //
              << "bo->data_r() " << bo->data_r() << " "                       //
              << "bo->data_r() " << bo->data_w() << " "                       //
              << "bo->phy() " << std::hex << "0x" << bo->phy() << " "         //
              << std::dec                                                     //
              << std::endl;
    bo->sync_for_read(100, 200);
    bo->sync_for_write(200, 300);
    all.emplace_back(std::move(bo));
  }
  std::cout << "press enter to releaes memory and continue ... \n"
               "you can use xbutil query -d 0 to check memory usage ...\n";
  char c = 0;
  std::cin >> c;
  cout << "BYEBYE";
  return 0;
}
