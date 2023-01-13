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
#include <xrt.h>

#include <cstdlib>
#include <functional>
#include <array>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include "xir/xrt_device_handle.hpp"
class ert_start_kernel_cmd;
namespace xir {
struct my_bo_handle {
  xclDeviceHandle handle;
#ifdef _WIN32
  void* bo_handle;
#else
  unsigned int bo_handle;
#endif
  void* bo_addr;
  size_t cu_index;
  size_t ip_index;
  unsigned int cu_mask;
  uint64_t cu_addr;
  size_t device_id;
  size_t core_id;
  uint64_t fingerprint;
  std::string name;
  std::string kernel_name;
  std::string full_name;
  std::array<unsigned char, SIZE_OF_UUID> uuid;
  ert_start_kernel_cmd* get() {
    return reinterpret_cast<ert_start_kernel_cmd*>(bo_addr);
  }
};
class XrtCu {
 public:
  explicit XrtCu(const std::string& cu_name);
  virtual ~XrtCu();
  XrtCu(const XrtCu& other) = delete;
  XrtCu& operator=(const XrtCu& rhs) = delete;
  using prepare_ecmd_t = std::function<void(ert_start_kernel_cmd*)>;
  using callback_t = std::function<void(xclDeviceHandle, uint64_t)>;

 public:
  void run(size_t core_idx, prepare_ecmd_t prepare, callback_t on_success,
           callback_t on_failure);

  size_t get_num_of_cu() const;
  std::string get_full_name(size_t device_core_idx) const;
  std::string get_kernel_name(size_t device_core_idx) const;
  std::string get_instance_name(size_t device_core_idx) const;
  size_t get_device_id(size_t device_core_idx) const;
  size_t get_core_id(size_t device_core_idx) const;
  uint64_t get_fingerprint(size_t device_core_idx) const;
  uint32_t read_register(size_t device_core_idx, uint32_t offset) const;
  ert_start_kernel_cmd* get_cmd(size_t device_core_id);

 private:
  void init_cmd(size_t device_core_id);

 private:
  std::string cu_name_;
  std::shared_ptr<xir::XrtDeviceHandle> handle_;
  std::vector<my_bo_handle> bo_handles_;
};

}  // namespace xir
