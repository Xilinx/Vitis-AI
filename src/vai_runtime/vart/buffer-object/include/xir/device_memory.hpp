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
#include <string>
#include <vitis/ai/with_injection.hpp>

namespace xir {
/**
 *@brief device memory operation (upload and download)
 */
class DeviceMemory : public vitis::ai::WithInjection<DeviceMemory> {
 public:
  explicit DeviceMemory() = default;
  static std::unique_ptr<DeviceMemory> create(size_t);
 public:
  DeviceMemory(const DeviceMemory&) = delete;
  DeviceMemory& operator=(const DeviceMemory& other) = delete;

  virtual ~DeviceMemory() = default;

 public:
  virtual bool upload(const void* data, uint64_t offset, size_t size) = 0;

  virtual bool download(void* data, uint64_t offset, size_t size) = 0;

 public:
  virtual bool save(const std::string& filename, uint64_t offset,
                    size_t size) final;

 private:
};

}  // namespace xir
