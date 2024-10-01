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

#include "./hbm_manager_imp.hpp"

#include <glog/logging.h>

#include <cstdlib>
#include <vitis/ai/env_config.hpp>
DEF_ENV_PARAM(DEBUG_HBM_MANAGER, "0");
namespace {
static inline uint64_t align(uint64_t a, uint64_t b) {
  return (a / b + (a % b ? 1 : 0)) * b;
}

HbmManagerImp::HbmManagerImp(uint64_t from, uint64_t size,
                             uint64_t alignment)
    : vart::dpu::HbmManager(),  //
      from_{from},
      size_{size},
      alignment_{alignment},
      used_{} {}

HbmManagerImp::~HbmManagerImp() {  //
  CHECK(used_.empty()) << "MEMORY LEAK!";
}

void HbmManagerImp::release(const vart::dpu::HbmChunk* chunk) {  //
  auto it = used_.find(chunk);
  CHECK(it != used_.end()) << "LOGICIAL ERROR! bo is not found";
  used_.erase(it);
}

static std::string to_string(
    std::set<const vart::dpu::HbmChunk*, HbmManagerImp::CompareBO>& used) {
  std::ostringstream str;
  int x = 0;
  str << "{";
  for (auto c : used) {
    if (x++ != 0) {
      str << ",";
    }
    /*    str << std::hex << "(0x" << c->get_offset() << std::dec << ","  //
            << std::hex << "0x" << c->get_size() << std::dec << ")"     //
            ;*/
    str << c->to_string();
  }
  str << "}";
  return str.str();
}

std::unique_ptr<vart::dpu::HbmChunk> HbmManagerImp::allocate(uint64_t size0) {
  const uint64_t capacity = align(size0, alignment_);
  uint64_t base = align(from_, alignment_);
  for (const auto bo : used_) {
    const auto offset = bo->get_offset();
    if (base + capacity <= offset) {
      break;
    }
    base = offset + bo->get_capacity();
  }
  auto out_of_range = (base + capacity > from_ + size_);
  LOG_IF(INFO, ENV_PARAM(DEBUG_HBM_MANAGER) >= 2 || out_of_range)
      << (out_of_range ? "out of memory! " : "")      //
      << "base "                                      //
      << "0x" << std::hex << base << std::dec << " "  //
      << "size " << size0 << " "                      //
      << "capacity " << capacity << " "               //
      << "from_ "
      << "0x" << std::hex << from_ << std::dec << " "              //
      << "size_ " << std::hex << "0x" << size_ << std::dec << " "  //
      << " used: " << to_string(used_);
  ;
  auto ret = std::unique_ptr<vart::dpu::HbmChunk>();
  if (!out_of_range) {
    ret = std::make_unique<vart::dpu::HbmChunk>(this, base, size0, capacity,
                                                alignment_);
    used_.emplace(ret.get());
  }
  LOG_IF(INFO, ENV_PARAM(DEBUG_HBM_MANAGER) >= 5 || ret == nullptr)
      << " used: " << to_string(used_) << "return: "
      << (ret == nullptr ? std::string("nullptr") : ret->to_string());
  return ret;
}
bool HbmManagerImp::CompareBO::operator()(const vart::dpu::HbmChunk* a,
                                          const vart::dpu::HbmChunk* b) const {
  return a->get_offset() < b->get_offset();
}
}  // namespace

DECLARE_INJECTION(vart::dpu::HbmManager, HbmManagerImp, uint64_t&, uint64_t&,
                  uint64_t&)
