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

#include "./buffer_object_qnx.hpp"
#include "vitis/ai/xxd.hpp"
#include <algorithm>
#include <fcntl.h>
#include <glog/logging.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <vitis/ai/env_config.hpp>
#include <vitis/ai/weak.hpp>

DEF_ENV_PARAM(DEBUG_BUFFER_OBJECT, "0")

namespace {

uint64_t vtophys(void *addr) {
  off64_t buf;

  if (mem_offset64(addr, NOFD, 1, &buf, 0) < 0)
    return 0;
  return buf;
}

/**
 * dpu_alloc_mem -  alloc a memory block from the available memory list.
 * @memsize : size of memory
 *
 *  RETURN: address of alloced memory;  NULL returned if no enough space exists
 */

std::pair<void *, off64_t> dpu_alloc_mem(uint32_t memsize) {
  int fd=0;
  void* vaddr=NULL;

  fd = posix_typed_mem_open("/memory/below4G", O_RDWR, POSIX_TYPED_MEM_ALLOCATE_CONTIG);
    if (fd == -1) {
       vaddr  = mmap(NULL, memsize,
                               PROT_READ | PROT_WRITE | PROT_NOCACHE,
                               MAP_ANON | MAP_PHYS, NOFD, 0);
    } else {
        vaddr = mmap(NULL, memsize,
                               PROT_READ | PROT_WRITE | PROT_NOCACHE,
                               MAP_SHARED, fd, 0);
    }
    if (vaddr == MAP_FAILED) {
        perror("Unable to allocate DMA memory\n");
    }

  CHECK(MAP_FAILED != vaddr) << "alloc BO error."
                                << "memsize " << memsize << " " //
      ;
  // phy_addr 0x00000000XXXXX000 ?
  auto phy_addr = vtophys(vaddr);
  return std::make_pair(vaddr, phy_addr);
}

static size_t align_size(size_t size) {
  auto PAGE_SIZE = getpagesize();
  auto aligned = size % PAGE_SIZE == 0;
  auto new_size = aligned ? size : ((size / PAGE_SIZE + 1) * PAGE_SIZE);
  // dirty hack
  return std::max(new_size, (size_t)(2 * PAGE_SIZE));
}

BufferObjectQnx::BufferObjectQnx(size_t size)
    : BufferObject(),              //
      size_{size},                 //
      capacity_{align_size(size)}, //
      data_{nullptr},              //
      phy_{0}, cache_ctl_{} {
  LOG_IF(INFO, ENV_PARAM(DEBUG_BUFFER_OBJECT))
      << "size " << size << " "           //
      << "capacity_ " << capacity_ << " " //
      ;
  std::tie(data_, phy_) = dpu_alloc_mem(capacity_);
  cache_ctl_.fd = NOFD;
  CHECK_NE(cache_init(0, &cache_ctl_, NULL), -1);
  LOG_IF(INFO, ENV_PARAM(DEBUG_BUFFER_OBJECT))
      << "data_ 0x" << data_ << " "                             //
      << "phy_ " << std::hex << "0x" << phy_ << " " << std::dec //
      ;
}

BufferObjectQnx::~BufferObjectQnx() { //
  munmap(data_, capacity_);
  cache_fini(&cache_ctl_);
}

const void *BufferObjectQnx::data_r() const { //
  return data_;
}

void *BufferObjectQnx::data_w() { //
  return data_;
}

size_t BufferObjectQnx::size() { return size_; }

uint64_t BufferObjectQnx::phy(size_t offset) { return phy_ + offset; }

void BufferObjectQnx::sync_for_read(uint64_t offset, size_t size) {
  CACHE_INVAL(&cache_ctl_, get_w<char>(offset), phy_ + offset, size);
  LOG_IF(INFO, ENV_PARAM(DEBUG_BUFFER_OBJECT)) << "sync_for_read "           //
                                               << "data_ " << data_ << " "   //
                                               << "phy_ " << phy_ << " "     //
                                               << "offset " << offset << " " //
                                               << "size " << size << " "     //
                                               << std::endl;
}

void BufferObjectQnx::sync_for_write(uint64_t offset, size_t size) {
  CACHE_FLUSH(&cache_ctl_, get_w<char>(offset), phy_ + offset, size);
  LOG_IF(INFO, ENV_PARAM(DEBUG_BUFFER_OBJECT)) << "sync_for_write "          //
                                               << "data_ " << data_ << " "   //
                                               << "phy_ " << phy_ << " "     //
                                               << "offset " << offset << " " //
                                               << "size " << size << " "     //
                                               << std::endl;
}

namespace {
static struct BufferObjectQnxRegistar {
  BufferObjectQnxRegistar() {
    BufferObjectQnx::registar("00_qnx", [](size_t size) {
      return std::make_unique<BufferObjectQnx>(size);
    });
  }
} g_registar;
} // namespace
} // namespace
