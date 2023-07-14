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
#pragma once
#include <cstdint>
#include <set>

#include "./hbm_manager.hpp"
namespace {
class HbmManagerImp : public vart::dpu::HbmManager {
 public:
  explicit HbmManagerImp(uint64_t from, uint64_t size,
                         uint64_t alignment = 4 * 1024 * 1024);
  HbmManagerImp(const HbmManagerImp&) = delete;
  HbmManagerImp& operator=(const HbmManagerImp& other) = delete;
  virtual ~HbmManagerImp();

 private:
  virtual void release(const vart::dpu::HbmChunk* chunk) override;
  virtual std::unique_ptr<vart::dpu::HbmChunk> allocate(uint64_t size) override;

 public:
  struct CompareBO {
    bool operator()(const vart::dpu::HbmChunk* a,
                    const vart::dpu::HbmChunk* b) const;
  };

 private:
  const uint64_t from_;
  const uint64_t size_;
  const uint64_t alignment_;
  std::set<const vart::dpu::HbmChunk*, CompareBO> used_;
};
}  // namespace
