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
#include "./buffer_object_fd.hpp"
#include <fcntl.h>
#include <glog/logging.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <cassert>
#include <unordered_map>
namespace vitis {
namespace xir {
template <typename T>
struct WeakSingleton {
  template <typename... Args>
  static std::shared_ptr<T> create(Args&&... args) {
    static std::weak_ptr<T> the_instance_;
    std::shared_ptr<T> ret;
    if (the_instance_.expired()) {
      ret = std::make_shared<T>(std::forward<Args>(args)...);
      the_instance_ = ret;
    }
    ret = the_instance_.lock();
    assert(ret != nullptr);
    return ret;
  }
};

template <typename K, typename T>
struct WeakStore {
  template <typename... Args>
  static std::shared_ptr<T> create(const K& key, Args&&... args) {
    static std::unordered_map<K, std::weak_ptr<T>> the_store_;
    std::shared_ptr<T> ret;
    if (the_store_[key].expired()) {
      ret = std::make_shared<T>(std::forward<Args>(args)...);
      the_store_[key] = ret;
    }
    ret = the_store_[key].lock();
    assert(ret != nullptr);
    return ret;
  }
};

std::shared_ptr<buffer_object_fd> buffer_object_fd::create(
    const std::string& name, int flags) {
  // return std::make_shared<buffer_object_fd>(name, flags);
  return WeakStore<std::string, buffer_object_fd>::create(name, name, flags);
}

static int my_open(const std::string& name, int flags) {
  auto fd = open(name.c_str(), flags);
  CHECK_GT(fd, 0) << ", open(" << name << ") failed.";
  return fd;
}

buffer_object_fd::buffer_object_fd(const std::string& name, int flags)
    : fd_{my_open(name, flags)} {}
buffer_object_fd::~buffer_object_fd() { close(fd_); }
}  // namespace xir
}  // namespace vitis
