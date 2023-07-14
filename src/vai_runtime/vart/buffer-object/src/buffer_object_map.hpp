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
#include <glog/logging.h>
#include <memory>
#include "./buffer_object_fd.hpp"
namespace vitis {
namespace xir {
class buffer_object_map {
 public:
  static std::unique_ptr<buffer_object_map> create(
      std::shared_ptr<buffer_object_fd> fd, size_t phy_addr, size_t size);

 public:
  explicit buffer_object_map(std::shared_ptr<buffer_object_fd> fd,
                             size_t phy_addr, size_t size);
  buffer_object_map(const buffer_object_map&) = delete;
  buffer_object_map& operator=(const buffer_object_map& other) = delete;

  virtual ~buffer_object_map();

  template <typename T>
  T* get(int offset = 0) {
    size_t xx = (page_offset_ + offset);
    auto ret = reinterpret_cast<T*>(static_cast<char*>(virt_addr_) + xx);
    if (0) {
      LOG(INFO) << " returing " << (void*)ret << " virt_addr_ "
                << (void*)virt_addr_ << std::hex << " page_offset_ 0x"
                << page_offset_ << " offset 0x" << offset << " phy 0x"
                << phy_addr_ << " xx 0x" << xx << std::dec;
    }
    return ret;
  }

 private:
  std::shared_ptr<buffer_object_fd> fd_;
  size_t phy_addr_;
  size_t size_;
  size_t page_addr_;
  size_t page_offset_;
  size_t page_size_;
  void* virt_addr_;
};

}  // namespace xir
}  // namespace vitis
