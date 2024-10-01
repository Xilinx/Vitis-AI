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

#include <cstdint>
#include <set>
#include <string>
#include <vitis/ai/with_injection.hpp>
#include <xir/device_memory.hpp>

namespace vart {
namespace dpu {
constexpr uint64_t _1M = 1024ull * 1024ull;
constexpr uint64_t _256M = 256ull * _1M;
constexpr uint64_t _4K = 4ull * 1024ull;
struct hbm_channel_def_t {
  uint64_t offset;
  uint64_t capacity = _256M;
  uint64_t alignment = _4K;
};

using chunk_def_t = std::vector<hbm_channel_def_t>;
class HbmChunk;
class HbmManager : public vitis::ai::WithInjection<HbmManager> {
 public:
  static std::unique_ptr<HbmManager> create(const vart::dpu::chunk_def_t& def);
  static std::unique_ptr<HbmManager> create(uint64_t from, uint64_t size,
                                            uint64_t alignment);
  explicit HbmManager() = default;
  HbmManager(const HbmManager&) = delete;
  HbmManager& operator=(const HbmManager& other) = delete;

  virtual ~HbmManager() = default;

 public:
  virtual void release(const HbmChunk* chunk) = 0;
  virtual std::unique_ptr<HbmChunk> allocate(uint64_t size) = 0;
};
class HbmChunk {
 public:
  friend class HbmManager;
  explicit HbmChunk(HbmManager* manager, uint64_t offset, uint64_t size,
                    uint64_t capacity, uint64_t alignment)
      : manager_{manager},
        offset_{offset},
        size_{size},
        capacity_{capacity},
        alignment_{alignment} {}
  explicit HbmChunk(uint64_t offset, uint64_t size)
      : manager_{nullptr},
        offset_{offset},
        size_{size},
        capacity_{size},
        alignment_{size} {}

 public:
  ~HbmChunk() {
    if (manager_) {
      manager_->release(this);
    }
  }

 public:
  uint64_t get_offset() const { return offset_; };
  uint64_t get_size() const { return size_; }
  uint64_t get_capacity() const { return capacity_; }
  uint64_t get_alignment() const { return alignment_; }
  std::string to_string() const;
  void upload(xir::DeviceMemory* dm, const void* data, size_t offset,
              size_t size) const;
  bool download(xir::DeviceMemory* dm, void* data, size_t offset, size_t size,
                bool ignore_error = false) const;

 private:
  HbmManager* manager_;
  const uint64_t offset_;
  const uint64_t size_;
  const uint64_t capacity_;
  const uint64_t alignment_;
};
}  // namespace dpu
}  // namespace vart
