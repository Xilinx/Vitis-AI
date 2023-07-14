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
#include <xrt.h>
#include "xir/device_memory.hpp"
namespace {
class DeviceMemoryCloud : public xir::DeviceMemory {
 public:
  DeviceMemoryCloud(size_t device_id);

  DeviceMemoryCloud(const DeviceMemoryCloud&) = delete;
  DeviceMemoryCloud& operator=(const DeviceMemoryCloud& other) = delete;

  virtual ~DeviceMemoryCloud();

 public:
  virtual bool upload(const void* data, uint64_t offset, size_t size) override;
  virtual bool download(void* data, uint64_t offset, size_t size) override;

 private:
  const size_t device_id_;
  xclDeviceHandle handle_;
};
}  // namespace
