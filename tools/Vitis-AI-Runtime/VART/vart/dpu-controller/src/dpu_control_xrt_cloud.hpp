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
#pragma once
#include <memory>

#include "./dpu_cloud.hpp"
#include "./xrt_cu.hpp"
#include "xir/dpu_controller.hpp"
#include "xir/xrt_device_handle.hpp"
class ert_start_kernel_cmd;

class DpuControllerXrtCloud : public xir::DpuController {
 public:
  DpuControllerXrtCloud(std::unique_ptr<xir::XrtCu>&& xrt_cu,
                        DPU_CLOUD_TYPE dpu_cloud_type);
  virtual ~DpuControllerXrtCloud();
  DpuControllerXrtCloud(const DpuControllerXrtCloud& other) = delete;
  DpuControllerXrtCloud& operator=(const DpuControllerXrtCloud& rhs) = delete;

 private:
  virtual void run(size_t core_idx, const uint64_t code,
                   const std::vector<uint64_t>& gen_reg) override;
  virtual size_t get_num_of_dpus() const override;
  virtual size_t get_device_id(size_t device_core_id) const override;
  virtual size_t get_core_id(size_t device_core_id) const override;
  virtual uint64_t get_fingerprint(size_t device_core_id) const override;
  virtual size_t get_batch_size(size_t device_core_id) const override;
  virtual std::string get_full_name(size_t device_core_id) const override;
  virtual std::string get_kernel_name(size_t device_core_id) const override;
  virtual std::string get_instance_name(size_t device_core_id) const override;

 private:
  std::string xdpu_get_counter(size_t device_core_id);

 private:
  std::unique_ptr<xir::XrtCu> xrt_cu_;
  DPU_CLOUD_TYPE dpu_cloud_type_;
};
