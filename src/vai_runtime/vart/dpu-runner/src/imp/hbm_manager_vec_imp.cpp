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

#include "./hbm_manager_vec_imp.hpp"

#include <glog/logging.h>

#include <cstdlib>
#include <vitis/ai/env_config.hpp>
DEF_ENV_PARAM(DEBUG_HBM_MANAGER, "0");

namespace {
std::vector<std::unique_ptr<vart::dpu::HbmManager>> create_managers(
    const vart::dpu::chunk_def_t& args) {
  auto ret = std::vector<std::unique_ptr<vart::dpu::HbmManager>>(args.size());
  int c = 0;
  for (const auto& arg : args) {
    ret[c++] =
        vart::dpu::HbmManager::create(arg.offset, arg.capacity, arg.alignment);
  }
  return ret;
}

HbmManagerVecImp::HbmManagerVecImp(const vart::dpu::chunk_def_t& args)
    : vart::dpu::HbmManager(),  //
      cursor_{0},
      managers_(create_managers(args)){};

HbmManagerVecImp::~HbmManagerVecImp() {  //
}

void HbmManagerVecImp::release(const vart::dpu::HbmChunk* chunk) {  //
  LOG(FATAL) << "not valid";
}

std::unique_ptr<vart::dpu::HbmChunk> HbmManagerVecImp::allocate(uint64_t size) {
  auto ret = std::unique_ptr<vart::dpu::HbmChunk>{};
  auto total = managers_.size();
  for (auto i = 0u; i < total; ++i) {
    auto idx = (cursor_ + i) % total;
    ret = managers_[idx]->allocate(size);
    if (ret) {
      break;
    }
  }
  LOG_IF(INFO, ENV_PARAM(DEBUG_HBM_MANAGER) >= 5)
      << "cursor_ " << cursor_ << " "  //
      << "return: "
      << (ret == nullptr ? std::string("nullptr") : ret->to_string());
  cursor_ = (cursor_ + 1) % total;
  return ret;
}
}  // namespace

DECLARE_INJECTION(vart::dpu::HbmManager, HbmManagerVecImp,
                  vart::dpu::chunk_def_t&);
DECLARE_INJECTION(vart::dpu::HbmManager, HbmManagerVecImp,
                  const vart::dpu::chunk_def_t&);
