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
#include <xclbin.h>
#include <xrt.h>

#include <array>
#include <string>
#include <vector>
namespace xir {
class XrtBinStream {
 public:
  explicit XrtBinStream(const std::string filename);
  ~XrtBinStream();

 public:
  void burn(int device_id = 0);
  void burn(xclDeviceHandle handle);
  void dump_layout() const;
  void dump_mem_topology() const;
  size_t get_num_of_cu() const;
  std::string get_cu(size_t cu_idx) const;
  std::string get_dsa() const;
  uint64_t get_cu_base_addr(size_t cu_idx) const;
  std::array<unsigned char, sizeof(xuid_t)> get_uuid() const;

 private:
  void init_fd(const std::string filename);
  void init_top();
  void init_uuid();
  void init_ip_layout();
  void init_mem_topology();
  void init_cu_names();
  void init_cu_indices();

 private:
  int fd_;
  void* data_;
  const axlf* top_;
  xuid_t uuid_;
  ip_layout* ip_layout_;
  mem_topology* topology_;
  std::vector<std::string> cu_names_;
  std::vector<size_t> indices_;
  std::string dsa_;
};
}  // namespace xir
