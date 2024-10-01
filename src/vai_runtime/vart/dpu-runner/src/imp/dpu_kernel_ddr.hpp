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
#include <xir/buffer_object.hpp>
#include <xir/dpu_controller.hpp>

#include "../dpu_kernel.hpp"

namespace vart {
namespace dpu {

class DpuKernelDdr : public vart::dpu::DpuKernel {
 public:
  DpuKernelDdr(const xir::Subgraph& sg, xir::Attrs* attrs,
               xir::DpuController* dpu_controller, size_t device_core_id);
  DpuKernelDdr(const std::string& filename, const std::string& kernel,
               xir::DpuController* dpu_controller, size_t device_core_id);

 public:
  ~DpuKernelDdr();

 public:
  virtual std::map<std::string, uint64_t> get_parameter(
      size_t device_core_id) const override;
  //  return segments_;
  virtual std::vector<vart::dpu::DpuKernel::SubgraphCode> get_code(
      size_t device_core_id) const override;

 private:
  virtual void load_parameter(
      const std::vector<vart::dpu::DpuReg>& parameters) override;
  virtual void load_code(const vart::dpu::DpuReg& code) override;

 public:
  virtual void initialize() override;

 private:
  const size_t device_core_id_;
  const std::string cu_full_name_;
  const size_t device_id_;
  std::vector<std::unique_ptr<xir::BufferObject>> codes_;
};

}  // namespace dpu
}  // namespace vart
