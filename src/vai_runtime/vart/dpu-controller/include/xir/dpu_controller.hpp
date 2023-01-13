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
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace xir {

class DpuController {
 public:
  /** @brief register a new factory method for creating a buffer
   * object, only one factory method is working. Invoking this
   * function second time will overwrite the last factory method.
   */
  static void registar(const std::string& name,
                       std::function<std::shared_ptr<DpuController>()>);
  static bool exist_dpu_devices();

  static std::shared_ptr<DpuController> get_instance();

 protected:
  explicit DpuController();

 public:
  DpuController(const DpuController& rhs) = delete;
  DpuController& operator=(const DpuController& rhs) = delete;

 public:
  virtual ~DpuController();

 public:
  virtual size_t get_num_of_dpus() const = 0;
  // convert a device_core_idx to a device_id, i.e. board idx
  virtual size_t get_device_id(size_t device_core_id) const = 0;
  virtual size_t get_core_id(size_t device_core_id) const = 0;
  virtual uint64_t get_fingerprint(size_t device_core_id) const = 0;
  virtual uint64_t get_device_hwconuter(size_t device_core_id) const { return 0; };
  virtual size_t get_batch_size(size_t device_core_id) const { return 1; }
  virtual std::string get_full_name(size_t device_core_id) const;
  virtual std::string get_kernel_name(size_t device_core_id) const;
  virtual std::string get_instance_name(size_t device_core_id) const;
  virtual size_t get_size_of_gen_regs(size_t device_core_id) const {
    return 8u;
  }
  // device_core_id in [0, get_num_of_dpus());
  virtual void run(size_t device_core_idx, const uint64_t code,
                   const std::vector<uint64_t>& gen_reg) = 0;
};
}  // namespace xir
