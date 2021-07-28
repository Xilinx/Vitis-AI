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
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vitis/ai/lock.hpp>

#include "./xrt_bin_stream.hpp"
#include "xir/xrt_device_handle.hpp"
namespace {
struct DeviceObject {
  uint64_t cu_base_addr;
  size_t cu_index;
  size_t ip_index;
  unsigned int cu_mask;
  xclDeviceHandle handle;
  std::string full_name;
  std::string kernel_name;  // cu_name?
  std::string instance_name;
  size_t device_id;
  size_t core_id;
  uint64_t fingerprint;
  unsigned int bank_flags;
  std::array<unsigned char, SIZE_OF_UUID> uuid;
};

class XrtDeviceHandleImp : public xir::XrtDeviceHandle {
 public:
  XrtDeviceHandleImp();
  virtual ~XrtDeviceHandleImp();

  virtual xclDeviceHandle get_handle(const std::string& cu_name,
                                     size_t core_idx) override;
  virtual size_t get_cu_index(const std::string& cu_name,
                              size_t core_idx) const override;
  virtual size_t get_ip_index(const std::string& cu_name,
                              size_t core_idx) const override;
  virtual unsigned int get_cu_mask(const std::string& cu_name,
                                   size_t core_idx) const override;
  virtual uint64_t get_cu_addr(const std::string& cu_name,
                               size_t core_idx) const override;
  virtual unsigned int get_num_of_cus(
      const std::string& cu_name) const override;

  virtual std::string get_cu_full_name(const std::string& cu_name,
                                       size_t core_idx) const override;
  virtual std::string get_cu_kernel_name(const std::string& cu_name,
                                         size_t core_idx) const override;
  virtual std::string get_instance_name(const std::string& cu_name,
                                        size_t core_idx) const override;
  virtual size_t get_device_id(const std::string& cu_name,
                               size_t core_idx) const override;
  virtual size_t get_core_id(const std::string& cu_name,
                             size_t core_idx) const override;

  virtual uint64_t get_fingerprint(const std::string& cu_name,
                                   size_t core_idx) const override;
  virtual unsigned int get_bank_flags(const std::string& cu_name,
                                      size_t core_idx) const override;
  virtual std::array<unsigned char, SIZE_OF_UUID> get_uuid(
      const std::string& cu_name, size_t core_idx) const override;

 private:
  XrtDeviceHandleImp(const XrtDeviceHandleImp& rhs) = delete;
  XrtDeviceHandleImp& operator=(const XrtDeviceHandleImp& rhs) = delete;

 private:
  DeviceObject& find_cu(const std::string& cu_name, size_t core_idx);
  const DeviceObject& find_cu(const std::string& cu_name,
                              size_t core_idx) const;

 private:
  std::map<std::string, DeviceObject> handles_;
  std::unique_ptr<xir::XrtBinStream> binstream_;

  std::vector<std::unique_ptr<vitis::ai::Lock>> mtx_;
  std::vector<std::unique_ptr<std::unique_lock<vitis::ai::Lock>>> locks_;
};
}  // namespace
