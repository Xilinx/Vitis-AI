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

#include "./buffer_object_fd.hpp"
#include "./buffer_object_map.hpp"
#include "xir/buffer_object.hpp"
namespace /* anonympous */ {
class BufferObjectDpCma : public ::xir::BufferObject {
 public:
  explicit BufferObjectDpCma(size_t size, size_t device_id = 0u,
                             const std::string& cu_name = std::string(""));
  BufferObjectDpCma(const BufferObjectDpCma&) = delete;
  BufferObjectDpCma& operator=(const BufferObjectDpCma& other) = delete;

  virtual ~BufferObjectDpCma();

 private:
  virtual void* data_w() override;
  virtual const void* data_r() const override;

  virtual size_t size() override;
  virtual uint64_t phy(size_t offset = 0) override;
  virtual void sync_for_read(uint64_t offset, size_t size) override;
  virtual void sync_for_write(uint64_t offset, size_t size) override;
  virtual void copy_from_host(const void* buf, size_t size,
                              size_t offset) override;
  virtual void copy_to_host(void* buf, size_t size, size_t offset) override;

 private:
  size_t size_;
  bool cache_;
  std::shared_ptr<vitis::xir::buffer_object_fd> fd_;
  std::shared_ptr<vitis::xir::buffer_object_fd> mm_fd_;
  size_t capacity_;
  void* data_;
  void* data_dev_mem_;
  uint64_t phy_;
};
}  // namespace
