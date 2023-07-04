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

#include "./xrt_cu.hpp"
#include "xir/dpu_controller.hpp"
#include "xir/xrt_device_handle.hpp"
#include "xir/buffer_object.hpp"
class ert_start_kernel_cmd;

class DpuControllerXrtXv2Dpu : public xir::DpuController {
 public:
  DpuControllerXrtXv2Dpu(std::unique_ptr<xir::XrtCu>&& xrt_cu);
  virtual ~DpuControllerXrtXv2Dpu();
  DpuControllerXrtXv2Dpu(const DpuControllerXrtXv2Dpu& other) = delete;
  DpuControllerXrtXv2Dpu& operator=(const DpuControllerXrtXv2Dpu& rhs) = delete;

 public:
  virtual void run(size_t core_idx, const uint64_t code,
                   const std::vector<uint64_t>& gen_reg) override;

  virtual size_t get_num_of_dpus() const override;
  virtual size_t get_device_id(size_t device_core_id) const override;
  virtual size_t get_core_id(size_t device_core_id) const override;
  virtual uint64_t get_fingerprint(size_t device_core_id) const override;
  virtual size_t get_batch_size(size_t device_core_id) const override;
  virtual size_t get_size_of_gen_regs(size_t device_core_id) const;
  virtual std::string get_full_name(size_t device_core_id) const override;
  virtual std::string get_kernel_name(size_t device_core_id) const override;
  virtual std::string get_instance_name(size_t device_core_id) const override;

 private:
  std::string xdpu_get_counter(size_t device_core_id);
  virtual uint64_t get_device_hwconuter(size_t device_core_id) const override;

 private:
  std::unique_ptr<xir::XrtCu> xrt_cu_;
  std::vector<std::vector<std::unique_ptr<xir::BufferObject>>> bo_;
};
