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
#include <glog/logging.h>

#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <string>

#include "./xrt_bin_stream.hpp"
#include "butler_client.h"
#include "butler_dev.h"
#include "xir/xrt_device_handle.hpp"

namespace {
struct DeviceObject {
  uint64_t cu_base_addr;
  size_t cu_index;
  unsigned int cu_mask;
  xclDeviceHandle handle;
  std::string full_name;
  std::string kernel_name;
  std::string name;
  size_t device_id;
  size_t core_id;
  uint64_t fingerprint;
  butler::handle butler_handle;
};
inline std::string to_string(const DeviceObject& x) {
  std::stringstream str;
  str << " cu_handle " << x.handle                                      //
      << " cu_mask " << x.cu_mask                                       //
      << " cu_index " << x.cu_index                                     //
      << " cu_addr " << std::hex << "0x" << x.cu_base_addr << std::dec  //
      << " cu_full_name " << x.full_name << " "                         //
      << " device_id " << x.device_id << " "                            //
      << " core_id " << x.core_id << " "                                //
      ;
  return str.str();
}
class XrtDeviceHandleImp : public xir::XrtDeviceHandle {
 public:
  XrtDeviceHandleImp();
  virtual ~XrtDeviceHandleImp();

  virtual xclDeviceHandle get_handle(const std::string& cu_name,
                                     size_t core_idx) override;
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
  std::unique_ptr<butler::ButlerClient> client_;
};
}  // namespace
