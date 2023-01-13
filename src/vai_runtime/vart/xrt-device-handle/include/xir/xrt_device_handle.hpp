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
#include <glog/logging.h>

#include <cstring>
#include <fstream>
#include <functional>
#include <memory>
#include <string>
static constexpr size_t SIZE_OF_UUID = 16u;
namespace xir {
class XrtDeviceHandle {
 public:
  /** @brief register a new factory method for creating a buffer
   * object, only one factory method is working. Invoking this
   * function second time will overwrite the last factory method.
   */
  static void registar(const std::string& name,
                       std::function<std::shared_ptr<XrtDeviceHandle>()>);

  // dirty hack, for dp_buffer_object, use get_instance(), shared among all cu.
  // for dp_dpu_controller, use create(), every controller own a handle.
  /** @brief return the install of xrt device handle
   */
  static std::shared_ptr<XrtDeviceHandle> get_instance();

  // static std::unique_ptr<XrtDeviceHandle> create();

 protected:
  explicit XrtDeviceHandle() = default;

 public:
  virtual ~XrtDeviceHandle() = default;
  // dirty hack, avoid include xrt.h
  // typedef void * xclDeviceHandle;
  virtual void* get_handle(const std::string& cu_name, size_t core_idx) = 0;
  virtual size_t get_cu_index(const std::string& cu_name,
                              size_t core_idx) const = 0;
  virtual size_t get_ip_index(const std::string& cu_name,
                              size_t core_idx) const = 0;
  virtual unsigned int get_cu_mask(const std::string& cu_name,
                                   size_t core_idx) const = 0;

  virtual uint64_t get_cu_addr(const std::string& cu_name,
                               size_t core_idx) const = 0;

  virtual unsigned int get_num_of_cus(const std::string& cu_name) const = 0;

  virtual std::string get_cu_full_name(const std::string& cu_name,
                                       size_t core_idx) const = 0;
  virtual std::string get_cu_kernel_name(const std::string& cu_name,
                                         size_t core_idx) const = 0;
  virtual std::string get_instance_name(const std::string& cu_name,
                                        size_t core_idx) const = 0;
  virtual size_t get_device_id(const std::string& cu_name,
                               size_t core_idx) const = 0;
  virtual size_t get_core_id(const std::string& cu_name,
                             size_t core_idx) const = 0;

  virtual uint64_t get_fingerprint(const std::string& cu_name,
                                   size_t core_idx) const = 0;
  virtual unsigned int get_bank_flags(const std::string& cu_name,
                                      size_t core_idx) const {
    return 0u;
  }
  virtual std::array<unsigned char, SIZE_OF_UUID> get_uuid(
      const std::string& cu_name, size_t core_idx) const = 0;

 private:
  XrtDeviceHandle(const XrtDeviceHandle& rhs) = delete;
  XrtDeviceHandle& operator=(const XrtDeviceHandle& rhs) = delete;
};
}  // namespace xir
