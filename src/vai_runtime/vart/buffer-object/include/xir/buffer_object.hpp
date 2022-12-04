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
#include <functional>
#include <memory>
#include <vitis/ai/with_injection.hpp>
#include "./buffer_object_export.hpp"
namespace xir {
struct XclBo {
  void* xcl_handle;
#ifdef _WIN32
  void* bo_handle;
#else
  unsigned int bo_handle;
#endif
};

/**
 * @brief a device memory management
 */
class BufferObject : public vitis::ai::WithInjection<BufferObject> {
 public:
  explicit BufferObject() = default;
  static VART_BUFFER_OBJECT_DLLSPEC std::unique_ptr<BufferObject> create(
      size_t size, size_t device_id, const std::string& cu_name);

 public:
  BufferObject(const BufferObject&) = delete;
  BufferObject& operator=(const BufferObject& other) = delete;

  virtual ~BufferObject() = default;

 public:
  virtual size_t size() = 0;
  virtual void* data_w() = 0;
  virtual const void* data_r() const = 0;
  virtual uint64_t phy(size_t offset = 0) = 0;
  virtual XclBo get_xcl_bo() const;
  /// sync_for_read before reading
  virtual void sync_for_read(uint64_t offset, size_t size) = 0;
  /// sync_for_write after write
  virtual void sync_for_write(uint64_t offset, size_t size) = 0;

  /// copy from host to this buffer object
  virtual void copy_from_host(const void* buf, size_t size, size_t offset) = 0;

  /// copy from this buffer object to host
  virtual void copy_to_host(void* buf, size_t size, size_t offset) = 0;

  /// todo: copy from other buffer object to this buffer object.
  /// virtual void copy_from_other_bo (BufferObject * other, size_t size, size_t
  /// offset) = 0;

 public:
  template <typename T>
  T* get_w(size_t offset = 0) {
    return reinterpret_cast<T*>(data_w()) + offset;
  }
  template <typename T>
  const T* get_r(size_t offset = 0) const {
    return reinterpret_cast<const T*>(data_r()) + offset;
  }
};
}  // namespace xir
