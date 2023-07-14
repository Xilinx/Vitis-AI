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
#include <xir/device_memory.hpp>
#include <xir/dpu_controller.hpp>

#include "../dpu_kernel.hpp"
#include "hbm_config.hpp"
#include "hbm_manager.hpp"
namespace vart {
namespace dpu {

class DpuKernelHbm : public vart::dpu::DpuKernel {
 public:
  DpuKernelHbm(const std::string& filename, const std::string& kernel,
               xir::DpuController* dpu_controller, size_t device_core_id);
  DpuKernelHbm(const xir::Subgraph& sg, xir::Attrs* attrs,
               xir::DpuController* dpu_controller, size_t device_core_id);

 public:
  virtual ~DpuKernelHbm();

 public:
  // key: "REG_0", "REG_1", or "REG_2" etc
  // TODO: rename
  virtual std::map<std::string, uint64_t> get_parameter(
      size_t device_core_id) const override;
  virtual std::vector<vart::dpu::DpuKernel::SubgraphCode> get_code(
      size_t device_core_id) const override;

 private:
  virtual void load_parameter(
      const std::vector<vart::dpu::DpuReg>& parameters) override;
  virtual void load_code(const vart::dpu::DpuReg& code) override;

 private:
  // get_code_hbm_managers(device_core_id)
  vart::dpu::HbmManager* get_code_hbm_manager();
  // get_parameter_hbm_managers(device_core_id, reg_id W0 or W1);
  vart::dpu::HbmManager* get_parameter_hbm_managers(
      const std::string& reg_id /*W0 or W1*/);

 private:
  const size_t device_core_id_;
  const std::string cu_full_name_;
  const size_t device_id_;
  std::shared_ptr<xir::DeviceMemory> device_memory_;

  // multiple kernel shared the same hbm manager, one per dpu core;
  // code_hbm_managers_[reg_id];
  std::map<std::string, std::shared_ptr<vart::dpu::HbmManager>> hbm_managers_;

  // one per dpu core
  // code_chunks_[subgraph_id], must not be nullptr
  std::vector<std::unique_ptr<vart::dpu::HbmChunk>> code_chunks_;
  // one vector of parameters per dpu core,
  // parameter_chunks_[device_core_id][reg_id], never be nullptr
  std::map<std::string, std::unique_ptr<vart::dpu::HbmChunk>> parameter_chunks_;
};
}  // namespace dpu
}  // namespace vart
