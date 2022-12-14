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
#if _WIN32
// for type conversion
#  pragma warning(disable : 4267)
#endif

#include "./buffer_object_xrt_imp.hpp"

#include <glog/logging.h>

#include <cstring>
#include <string>

#include "vitis/ai/env_config.hpp"
#include "vitis/ai/weak.hpp"
DEF_ENV_PARAM(DEBUG_BUFFER_OBJECT, "0")

namespace {
static uint64_t get_physical_address(const xclDeviceHandle& handle,
                                     const xclBufferHandle bo) {
  xclBOProperties p;
  auto error_code = xclGetBOProperties(handle, bo, &p);
  CHECK_EQ(error_code, 0) << "cannot xclGetBOProperties !";
  auto phy = error_code == 0 ? p.paddr : -1;
  LOG_IF(INFO, ENV_PARAM(DEBUG_BUFFER_OBJECT))
      << "error_code " << error_code << " "        //
      << "handle " << handle << " "                //
      << "bo " << bo << " "                        //
      << "phy " << std::hex << "0x" << phy << " "  //
      << std::dec << std::endl;
  CHECK_NE(phy, (decltype(phy))(-1)) << "cannot xclGetBOProperties ! "  //
                                     << " error_code=" << error_code    //
                                     << " handle " << handle << " "
                                     << " bo=" << bo;
  return phy;
}

static device_info_t find_xrt_info(xir::XrtDeviceHandle* device_handle,
                                   size_t device_id,
                                   const std::string& cu_name) {
  auto num_of_device_cores = device_handle->get_num_of_cus(cu_name);
  for (auto i = 0u; i < num_of_device_cores; ++i) {
    if (device_id == device_handle->get_device_id(cu_name, i)) {
      return device_info_t{
          device_id,                                 //
          device_handle->get_handle(cu_name, i),     // xrt handle
          device_handle->get_bank_flags(cu_name, i)  // flags
      };
    }
  }
  LOG(FATAL) << "cannot find xrt device handle for device id " << device_id;
  return device_info_t{};
}

BufferObjectXrtEdgeImp::BufferObjectXrtEdgeImp(size_t size, size_t device_id,
                                               const std::string& cu_name)
    : BufferObject(),                                          //
      holder_{xir::XrtDeviceHandle::get_instance()},           //
      xrt_{find_xrt_info(holder_.get(), device_id, cu_name)},  //
      size_{size}                                              //
{
  bo_ = xclAllocBO(xrt_.handle, size, 0 /* not used according to xrt.h*/,
                   xrt_.flags);
  CHECK(bo_ != XRT_NULL_BO)
      << " allocation failure: "
      << " xrt_.handle " << xrt_.handle << " "
      << "xrt_.device_id " << xrt_.device_id << " "                          //
      << "size  " << size << " "                                             //
      << "xrt_.flags " << std::hex << "0x" << xrt_.flags << std::dec << " "  //
      ;
  LOG_IF(INFO, ENV_PARAM(DEBUG_BUFFER_OBJECT))
      << " xrt_.handle " << xrt_.handle << " "
      << "xrt_.device_id " << xrt_.device_id << " "                          //
      << "xrt_.flags " << std::hex << "0x" << xrt_.flags << std::dec << " "  //
      ;
  if (xrt_.flags & XCL_BO_FLAGS_DEV_ONLY) {
    data_ = nullptr;
  } else {
    data_ = (int*)xclMapBO(xrt_.handle, bo_, true);  //
    // XRT memory allocation is dynamic. After xclMapBO() , only the vritual
    // address is returned, and the physical memory is not immediately applied.
    // Only when a 'page falut' is generated when the memory is written for the
    // first time, the physical memory is applied for and then the write
    // operation is performed.
    // It takes 26ms to apply for 106MB of physical memory.
    // In order not to affect the performance measurement of the first write
    // operation, mmset is executed after mmap to actively apply for physical
    // memory.
    std::memset(data_, 0, size_);  //
    // cache flush.
    // When the cache is enabled, after memset is executed, the time for cpu to
    // write back to the cache is not fixed, If it alternats with DPU writing,
    // dirty data will appear in the cache, causeing some very difficult to
    // debug errors. eg. DPU outputs random 64 bits zero .
    // Execute cache flush ， write data to cache immediately.
    xclSyncBO(xrt_.handle, bo_, XCL_BO_SYNC_BO_TO_DEVICE, size_, 0);
  }
  phy_ = get_physical_address(xrt_.handle, bo_);  //
  LOG_IF(INFO, ENV_PARAM(DEBUG_BUFFER_OBJECT))
      << "create bufferobject "                                  //
      << "phy_ " << std::hex << "0x" << phy_ << std::dec << " "  //
      << "size " << size << " "                                  //
      << " ptr " << (void*)data_ << " ";
}

BufferObjectXrtEdgeImp::~BufferObjectXrtEdgeImp() {
  if (data_ != nullptr) {
    xclUnmapBO(xrt_.handle, bo_, data_);
  }
  xclFreeBO(xrt_.handle, bo_);
}

const void* BufferObjectXrtEdgeImp::data_r() const {  //
  return data_;
}

void* BufferObjectXrtEdgeImp::data_w() {  //
  return data_;
}

size_t BufferObjectXrtEdgeImp::size() { return size_; }

uint64_t BufferObjectXrtEdgeImp::phy(size_t offset) { return phy_ + offset; }

void BufferObjectXrtEdgeImp::sync_for_read(uint64_t offset, size_t size) {
  if (data_ == nullptr) {
    LOG_IF(INFO, ENV_PARAM(DEBUG_BUFFER_OBJECT))
        << " meaningless for sync a device only memory";
    return;
  }
  auto sync =
      xclSyncBO(xrt_.handle, bo_, XCL_BO_SYNC_BO_FROM_DEVICE, size, offset);

  LOG_IF(INFO, ENV_PARAM(DEBUG_BUFFER_OBJECT))
      << "phy " << std::hex << "0x" << phy_ << std::dec << " "  //
      << "offset " << std::hex << "0x" << offset << " "         //
      << std::dec <<                                            //
      "size " << size << " "                                    //
      ;
  CHECK_EQ(sync, 0)
      << "Synchronize the buffer contents from device to host fail !";
}

void BufferObjectXrtEdgeImp::sync_for_write(uint64_t offset, size_t size) {
  if (data_ == nullptr) {
    LOG_IF(INFO, ENV_PARAM(DEBUG_BUFFER_OBJECT))
        << " meaningless for sync a device only memory";
    return;
  }
  auto sync =
      xclSyncBO(xrt_.handle, bo_, XCL_BO_SYNC_BO_TO_DEVICE, size, offset);
  LOG_IF(INFO, ENV_PARAM(DEBUG_BUFFER_OBJECT))
      << "phy " << std::hex << "0x" << phy_ << std::dec << " "  //
      << "offset " << std::hex << "0x" << offset << " "         //
      << std::dec <<                                            //
      "size " << size << " "                                    //
      ;
  CHECK_EQ(sync, 0)
      << "Synchronize the buffer contents from host to device fail !";
}

void BufferObjectXrtEdgeImp::copy_from_host(const void* buf, size_t size,
                                            size_t offset) {
  LOG_IF(INFO, ENV_PARAM(DEBUG_BUFFER_OBJECT))
      << "copy from host to device "
      << "phy " << std::hex << "0x" << phy_ << std::dec << " "  //
      << "offset " << std::hex << "0x" << offset << " "         //
      << std::dec <<                                            //
      "size " << size << " "                                    //
      ;
  CHECK_LE(offset + size, size_) << " out of range";
  auto ok = 0;
#if IS_EDGE
  memcpy(static_cast<char*>(data_w()) + offset, buf, size);
  sync_for_write(offset, size);
#else
  auto flags = 0;
  ok = xclUnmgdPwrite(xrt_.handle, flags, buf, size, phy(offset));
#endif
  CHECK_EQ(ok, 0) << "fail to write bo "
                  << "size " << size << " "      //
                  << "offset " << offset << " "  //
                  << "phy " << std::hex << "0x" << phy_ << " ";
}
void BufferObjectXrtEdgeImp::copy_to_host(void* buf, size_t size,
                                          size_t offset) {
  LOG_IF(INFO, ENV_PARAM(DEBUG_BUFFER_OBJECT))
      << "copy from host to device "
      << "phy " << std::hex << "0x" << phy_ << std::dec << " "  //
      << "offset " << std::hex << "0x" << offset << " "         //
      << std::dec <<                                            //
      "size " << size << " "                                    //
      ;
  CHECK_LE(offset + size, size_) << " out of range";
  auto ret = 0;
#if IS_EDGE
  sync_for_read(offset, size);
  memcpy(buf, static_cast<const char*>(data_r()) + offset, size);
#else
  auto flags = 0;
  ret = xclUnmgdPread(xrt_.handle, flags, buf, size, phy(offset));
#endif
  CHECK_EQ(ret, 0) << "fail to read bo "
                   << "size " << size << " "      //
                   << "offset " << offset << " "  //
                   << "phy " << std::hex << "0x" << phy_ << " ";
}

xir::XclBo BufferObjectXrtEdgeImp::get_xcl_bo() const {
  return xir::XclBo{xrt_.handle, bo_};
}
}  // namespace

REGISTER_INJECTION_BEGIN(xir::BufferObject, 1, BufferObjectXrtEdgeImp, size_t&,
                         size_t&, const std::string&) {
  auto ret = xclProbe() > 0;
  LOG_IF(INFO, ENV_PARAM(DEBUG_BUFFER_OBJECT))
      << " ret=" << ret
      << " register factory methord of BufferObjectXrtEdgeImp for "
         " xir::BufferObject with priority `1`";
  return ret;
}
REGISTER_INJECTION_END
