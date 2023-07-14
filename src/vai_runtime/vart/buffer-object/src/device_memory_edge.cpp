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
#include "device_memory_edge.hpp"

#include <glog/logging.h>
//#include <vitis/ai/c++14.hpp>
#include <fcntl.h>
#include <sys/mman.h>

#include "vitis/ai/env_config.hpp"
DEF_ENV_PARAM(DEBUG_DEVICE_MEMORY, "0");
// using namespace std;
namespace {
DeviceMemoryEdge::DeviceMemoryEdge(size_t device_id) {}
DeviceMemoryEdge::~DeviceMemoryEdge() {}

// data ===> offset
bool DeviceMemoryEdge::upload(const void* data, uint64_t offset_addr,
                              size_t size) {
  LOG_IF(INFO, ENV_PARAM(DEBUG_DEVICE_MEMORY))
      << "data " << data << " "           //
      << "offset " << offset_addr << " "  //
      << "size " << size << " "           //
      ;
  long page_size = sysconf(_SC_PAGE_SIZE);
  unsigned long offset = offset_addr % page_size;
  unsigned long base =
      (offset_addr / page_size) * page_size;  // page size alignment;
  unsigned long extra_size = size + offset;
  unsigned long map_size =
      (extra_size / page_size) * page_size +
      (extra_size % page_size == 0 ? 0 : page_size);  // page size alignment;

  auto fd = open("/dev/mem", O_RDWR | O_SYNC);
  CHECK_GT(fd, 0);
  auto offset_data =
      mmap(NULL, extra_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, base);
  CHECK_NE(offset_data, MAP_FAILED);
  char* p = reinterpret_cast<char*>(offset_data);
  auto from = reinterpret_cast<const char*>(data);
  p = p + offset;
  for (size_t i = 0; i < size; i++) {
    *p++ = from[i];
  }
  munmap(offset_data, map_size);
  close(fd);
  return true;
}

// offset ===> data
bool DeviceMemoryEdge::download(void* data, uint64_t offsetaddr, size_t size) {
  LOG_IF(INFO, ENV_PARAM(DEBUG_DEVICE_MEMORY))
      << "data " << data << " "          //
      << "offset " << offsetaddr << " "  //
      << "size " << size << " "          //
      ;
  long page_size = sysconf(_SC_PAGE_SIZE);
  off_t offset = offsetaddr % page_size;
  off_t base = (offsetaddr / page_size) * page_size;  // page size alignment;
  size_t extra_size = size + offset;
  size_t map_size =
      (extra_size / page_size) * page_size +
      (extra_size % page_size == 0 ? 0 : page_size);  // page size alignment;

  auto fd = open("/dev/mem", O_RDWR | O_SYNC);
  CHECK_GT(fd, 0);
  auto fromdata =
      mmap(NULL, extra_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, base);
  CHECK_NE(fromdata, MAP_FAILED)
      << "fd = " << fd << " base=" << std::hex << "0x" << base << std::dec;
  // fwrite(reinterpret_cast<const char *>(fromdata) + offset, size, 1,data);
  memcpy(data, reinterpret_cast<const char*>(fromdata) + offset, size);
  munmap(fromdata, map_size);
  close(fd);
  return true;
}
}  // namespace

REGISTER_INJECTION_BEGIN(xir::DeviceMemory, 2, DeviceMemoryEdge, size_t&) {
  return true;
}
REGISTER_INJECTION_END
