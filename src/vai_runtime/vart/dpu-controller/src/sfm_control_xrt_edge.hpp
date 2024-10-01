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
#include <memory>
#include <mutex>
#include <xir/buffer_object.hpp>

#include "./softmax.hpp"
#include "./xrt_cu.hpp"
// #include "vart/dpu/device_scheduler.hpp"
#include "xir/sfm_controller.hpp"
namespace {

class SfmControllerXrtEdge : public xir::SfmController {
 public:
  explicit SfmControllerXrtEdge(size_t core_idx,
                                std::unique_ptr<xir::XrtCu>&& xrt_cu);
  virtual ~SfmControllerXrtEdge();
  SfmControllerXrtEdge(const SfmControllerXrtEdge& other) = delete;
  SfmControllerXrtEdge& operator=(const SfmControllerXrtEdge& rhs) = delete;

 public:
  virtual void run(const int8_t* input, float scale, unsigned int cls,
                   unsigned int group, float* output) override;
  virtual bool supported(float scale, unsigned int cls,
                         unsigned int group) const override;

 public:
  virtual void run_xrt_cu(size_t core_idx, const uint64_t input,
                          const unsigned int cls, const unsigned int group,
                          const int scale, uint64_t output,
                          uint32_t offset) override;

 private:
  const size_t MAX_GROUP = 65535u;
  const size_t MAX_CLS = 1023u;
  const int CUR_SCALE = 2;

  const size_t core_idx_;
  int page_size_;
  std::unique_ptr<xir::XrtCu> xrt_cu_;
  std::unique_ptr<xir::BufferObject> workspace_;
  std::mutex mutex_;
};

class SfmControllerXrtEdgeWithScheduler : public xir::SfmController {
 public:
  explicit SfmControllerXrtEdgeWithScheduler();
  virtual ~SfmControllerXrtEdgeWithScheduler() = default;
  SfmControllerXrtEdgeWithScheduler(
      const SfmControllerXrtEdgeWithScheduler& other) = delete;
  SfmControllerXrtEdgeWithScheduler& operator=(
      const SfmControllerXrtEdgeWithScheduler& rhs) = delete;

 public:
  virtual void run(const int8_t* input, float scale, unsigned int cls,
                   unsigned int group, float* output) override;
  virtual void run_xrt_cu(size_t core_idx, const uint64_t input,
                          const unsigned int cls, const unsigned int group,
                          const int scale, uint64_t output,
                          uint32_t offset) override;

  virtual bool supported(float scale, unsigned int cls,
                         unsigned int group) const override;

 private:
  std::vector<std::unique_ptr<SfmControllerXrtEdge>> controllers_;
  // std::unique_ptr<vart::dpu::DeviceScheduler> scheduler_;
};

}  // namespace
