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
#include "xir/device_memory.hpp"
namespace {
class DeviceMemoryEdge : public xir::DeviceMemory {
 public:
  DeviceMemoryEdge(size_t device_id);

  DeviceMemoryEdge(const DeviceMemoryEdge&) = delete;
  DeviceMemoryEdge& operator=(const DeviceMemoryEdge& other) = delete;

  virtual ~DeviceMemoryEdge();

 public:
  virtual bool upload(const void* data, uint64_t offset, size_t size) override;
  virtual bool download(void* data, uint64_t offset, size_t size) override;
  //  virtual void save(const std::string& filename, uint64_t offset,
  //                    size_t size) override;
 private:
};
}  // namespace
