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
#include <tuple>
#include <vector>
#include "./hbm_config.hpp"
#include "./hbm_manager.hpp"
namespace {
class HbmManagerVecImp : public vart::dpu::HbmManager {
 public:
  explicit HbmManagerVecImp(const vart::dpu::chunk_def_t& args);
  HbmManagerVecImp(const HbmManagerVecImp&) = delete;
  HbmManagerVecImp& operator=(const HbmManagerVecImp& other) = delete;
  virtual ~HbmManagerVecImp();

 private:
  virtual void release(const vart::dpu::HbmChunk* chunk) override;
  virtual std::unique_ptr<vart::dpu::HbmChunk> allocate(uint64_t size) override;

 private:
  size_t cursor_;
  std::vector<std::unique_ptr<vart::dpu::HbmManager>> managers_;
};
}  // namespace
