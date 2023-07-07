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
//#include <xrt/xclhal2.h>

#ifdef ENABLE_CLOUD
#include <xrt.h>
#else
#include <xrt/xrt.h>
#endif

#include <cstdint>

#include "xir/buffer_object.hpp"
#include "xir/xrt_device_handle.hpp"
namespace {
struct device_info_t {
  const size_t device_id;
  const xclDeviceHandle handle;
  const unsigned int flags;
};

class BufferObjectXrtEdgeImp : public xir::BufferObject {
 public:
  BufferObjectXrtEdgeImp(size_t size, size_t device_id,
                         const std::string& cu_name);
  BufferObjectXrtEdgeImp(const BufferObjectXrtEdgeImp&) = delete;
  BufferObjectXrtEdgeImp& operator=(const BufferObjectXrtEdgeImp& other) =
      delete;

  virtual ~BufferObjectXrtEdgeImp();

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
  virtual xir::XclBo get_xcl_bo() const override;

 private:
  const std::shared_ptr<xir::XrtDeviceHandle> holder_;
  const device_info_t xrt_;
  const size_t size_;
#ifdef _WIN32
  void* bo_;
#else
  unsigned int bo_;
#endif
  int* data_;
  uint64_t phy_;
};
}  // namespace
