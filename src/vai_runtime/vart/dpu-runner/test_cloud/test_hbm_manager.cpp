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
#include <glog/logging.h>

#include <iostream>
#include <vector>
using namespace std;

#include "../src/imp/hbm_config.hpp"
#include "../src/imp/hbm_manager.hpp"
int main(int argc, char* argv[]) {  //
  uint64_t from = 1 * 256 * 1024 * 1024;
  uint64_t size = 256 * 1024 * 1024;
  uint64_t chunk_size = size / 3;
  uint64_t alignment = 4 * 1024;
  const vart::dpu::chunk_def_t args = {{from, size, alignment},
                                       {from + size, size, alignment}};
  auto x1 = vart::dpu::HbmManager::create(args);
  vector<unique_ptr<vart::dpu::HbmChunk>> v;
  for (auto i = 0; i < 40; ++i) {
    auto c = x1->allocate(chunk_size);
    if (c != nullptr) {
      LOG(INFO) << "C = " << c->to_string();
      v.emplace_back(move(c));
    } else {
      break;
    }
  }
  for (const auto& x : vart::dpu::HBM_CHANNELS()) {
    LOG(INFO) << " x= " << x;
  }
  for (auto core_id = 0u; core_id < 2; ++core_id) {
    auto workspace = vart::dpu::get_engine_hbm(core_id);
    for (const auto& w : workspace) {
      LOG(INFO) << "core[" << core_id << "] w = " << w;
    }
  }
  return 0;
}
