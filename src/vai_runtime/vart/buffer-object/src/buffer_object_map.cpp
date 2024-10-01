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
#include "./buffer_object_map.hpp"
#include <iostream>
using namespace std;

#include <glog/logging.h>
#include <sys/mman.h>
#include <unistd.h>
namespace vitis {
namespace xir {

static size_t align(size_t a, size_t b) {
  if (a % b == 0) {
    return a;
  }
  return (a / b + 1) * b;
}

std::unique_ptr<buffer_object_map> buffer_object_map::create(
    std::shared_ptr<buffer_object_fd> fd, size_t phy_addr, size_t size) {
  return std::unique_ptr<buffer_object_map>(
      new buffer_object_map(fd, phy_addr, size));
}

buffer_object_map::buffer_object_map(std::shared_ptr<buffer_object_fd> fd,
                                     size_t phy_addr, size_t size)
    : fd_{fd},
      phy_addr_{phy_addr},  //
      size_{size},
      page_addr_{phy_addr / getpagesize() * getpagesize()},
      page_offset_{phy_addr % getpagesize()},
      page_size_{align(size + page_offset_, getpagesize())},
      virt_addr_{nullptr} {
  virt_addr_ = mmap(NULL, page_size_, PROT_READ | PROT_WRITE, MAP_SHARED,
                    fd_->fd(), page_addr_);
  CHECK_NE(virt_addr_, MAP_FAILED) << "map failed";
}
buffer_object_map::~buffer_object_map() { munmap(virt_addr_, page_size_); }
}  // namespace xir
}  // namespace vitis
