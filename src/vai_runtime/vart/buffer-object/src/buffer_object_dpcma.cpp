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

#include "./buffer_object_dpcma.hpp"

#include <fcntl.h>
#include <glog/logging.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <vitis/ai/env_config.hpp>

#include "./buffer_object_fd.hpp"
#include "./dpu.h"
#include "vitis/ai/xxd.hpp"
#define DEV "/dev/dpu"
DEF_ENV_PARAM(DEBUG_BUFFER_OBJECT, "0")

namespace {

static int read_env(const char* name, int default_value) {
  auto p = getenv(name);
  return p ? std::stoi(p) : default_value;
}

BufferObjectDpCma::BufferObjectDpCma(
    size_t size, size_t device_id /* not used */,
    const std::string& cu_name /* std::string("") not used*/)
    : BufferObject(),                                          //
      size_{size},                                             //
      cache_{read_env("BUFFER_OBJECT_CACHE", 1) != 0},         //
      fd_{vitis::xir::buffer_object_fd::create(DEV, O_RDWR)},  //
      mm_fd_{cache_ ? nullptr
                    : vitis::xir::buffer_object_fd::create("/dev/mem",
                                                           O_RDWR | O_SYNC)},
      capacity_{0},            //
      data_{nullptr},          //
      data_dev_mem_{nullptr},  //
      phy_{0} {
  dpcma_req_alloc req_alloc{size_, 0, 0};
  auto r = ioctl(fd_->fd(), DPUIOC_CREATE_BO, &req_alloc);
  CHECK_EQ(r, 0) << ", cannot allocate memory";
  phy_ = req_alloc.phy_addr;
  CHECK_EQ(static_cast<decltype(req_alloc.phy_addr)>(phy_), req_alloc.phy_addr)
      << ", size error";
  capacity_ = req_alloc.capacity;
  //
  data_ = mmap(NULL, capacity_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_->fd(),
               phy_);
  CHECK_NE(data_, MAP_FAILED) << "mmap failed";
  if (!cache_) {
    data_dev_mem_ = mmap(NULL, capacity_, PROT_READ | PROT_WRITE, MAP_SHARED,
                         mm_fd_->fd(), phy_);
    CHECK_NE(data_dev_mem_, MAP_FAILED) << "mmap failed";
  }
}

BufferObjectDpCma::~BufferObjectDpCma() {
  munmap(data_, capacity_);
  if (!cache_) munmap(data_dev_mem_, capacity_);
  auto req_free = dpcma_req_free{(u64)phy_};
  auto r = ioctl(fd_->fd(), DPUIOC_FREE_BO, &req_free);
  CHECK_EQ(r, 0) << ", cannot free memory";
}

const void* BufferObjectDpCma::data_r() const {  //
  return cache_ ? data_ : data_dev_mem_;
}

void* BufferObjectDpCma::data_w() {  //
  // it is wired that writing from user space memory to phy memory, /dev/mem
  // is fast enough, it is not true visa
  return cache_ ? data_ : data_dev_mem_;
}

size_t BufferObjectDpCma::size() { return size_; }

uint64_t BufferObjectDpCma::phy(size_t offset) { return phy_ + offset; }

void BufferObjectDpCma::sync_for_read(uint64_t offset, size_t size) {
  LOG_IF(INFO, ENV_PARAM(DEBUG_BUFFER_OBJECT)) << "sync_for_read "
                                               << "offset " << offset << " "  //
                                               << "size " << size << " "      //
                                               << std::endl;
  if (cache_) {
    dpcma_req_sync req_sync{(u64)(phy_ + offset), size,
                            DPCMA_FROM_DEVICE_TO_CPU};
    auto r = ioctl(fd_->fd(), DPUIOC_SYNC_BO, &req_sync);
    CHECK_EQ(r, 0) << ",ioctl error";
  }
  /*  LOG(INFO) << "CACHE:\n"
            << vitis::ai::xxd((unsigned char *)data_dev_mem_, 160, 8, 1);
  LOG(INFO) << "DEVMEM:\n"
            << vitis::ai::xxd((unsigned char *)data_, 160, 8, 1);
  bool same = true;
  auto p1 = (unsigned char *)data_dev_mem_;
  auto p2 = (unsigned char *)data_;
  auto i = 0u;
  for (i = 0u; i < size; ++i) {
    same = same && p1[i] == p2[i];
    if (!same) {
      break;
    }
  }
  if (!same) {
    LOG(INFO) << "CACHE:" << i << "\n" << vitis::ai::xxd(&p1[i], 160, 8, 1);
    LOG(INFO) << "DEVMEM:\n" << vitis::ai::xxd(&p2[i], 160, 8, 1);
  } else {
    LOG(INFO) << "do nothing: " << i;
    } */
}

void BufferObjectDpCma::sync_for_write(uint64_t offset, size_t size) {
  LOG_IF(INFO, ENV_PARAM(DEBUG_BUFFER_OBJECT)) << "sync_for_write "
                                               << "offset " << offset << " "  //
                                               << "size " << size << " "      //
                                               << std::endl;
  if (cache_) {
    dpcma_req_sync req_sync{(u64)(phy_ + offset), size,
                            DPCMA_FROM_CPU_TO_DEVICE};
    auto r = ioctl(fd_->fd(), DPUIOC_SYNC_BO, &req_sync);
    CHECK_EQ(r, 0) << ",ioctl error";
  }
}
void BufferObjectDpCma::copy_from_host(const void* buf, size_t size,
                                       size_t offset) {
  LOG_IF(INFO, ENV_PARAM(DEBUG_BUFFER_OBJECT))
      << "phy " << std::hex << "0x" << phy_ << std::dec << " "  //
      << "offset " << std::hex << "0x" << offset << " "         //
      << std::dec <<                                            //
      "size " << size << " "                                    //
      ;
  CHECK_LE(offset + size, size_) << " out of range";
  memcpy(static_cast<char*>(data_w()) + offset, buf, size);
  sync_for_write(offset, size);
}
void BufferObjectDpCma::copy_to_host(void* buf, size_t size, size_t offset) {
  LOG_IF(INFO, ENV_PARAM(DEBUG_BUFFER_OBJECT))
      << "phy " << std::hex << "0x" << phy_ << std::dec << " "  //
      << "offset " << std::hex << "0x" << offset << " "         //
      << std::dec <<                                            //
      "size " << size << " "                                    //
      ;
  CHECK_LE(offset + size, size_) << " out of range";
  sync_for_read(offset, size);
  memcpy(buf, static_cast<const char*>(data_r()) + offset, size);
}

}  // namespace
REGISTER_INJECTION_BEGIN(xir::BufferObject, 2, BufferObjectDpCma, size_t&,
                         size_t&, const std::string&) {
  auto fd = open(DEV, O_RDWR);
  auto ret = fd >= 0;
  close(fd);
  LOG_IF(INFO, ENV_PARAM(DEBUG_BUFFER_OBJECT))
      << " ret=" << ret
      << " register factory methord of BufferObjectDpCma for "
         " xir::BufferObject with priority `2`";

  return ret;
}
REGISTER_INJECTION_END
