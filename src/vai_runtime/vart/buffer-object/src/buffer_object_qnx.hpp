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
#include "./buffer_object_fd.hpp"
#include "./buffer_object_map.hpp"
#include "xir/buffer_object.hpp"
#include <cstdint>
#include <sys/cache.h>
namespace /* anonympous */ {
class BufferObjectQnx : public xir::BufferObject {
public:
  explicit BufferObjectQnx(size_t size);
  BufferObjectQnx(const BufferObjectQnx &) = delete;
  BufferObjectQnx &operator=(const BufferObjectQnx &other) = delete;

  virtual ~BufferObjectQnx();

public:
  virtual void *data_w() override;
  virtual const void *data_r() const override;

  virtual size_t size() override;
  virtual uint64_t phy(size_t offset = 0) override;
  virtual void sync_for_read(uint64_t offset, size_t size) override;
  virtual void sync_for_write(uint64_t offset, size_t size) override;

private:
  size_t size_;
  size_t capacity_;
  void *data_;
  off64_t phy_;
  struct cache_ctrl cache_ctl_;
};
} // namespace
