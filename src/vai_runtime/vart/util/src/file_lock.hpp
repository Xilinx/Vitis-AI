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
#include <string>
#include <vitis/ai/lock.hpp>

namespace vitis {
namespace ai {

class FileLock : public Lock {
 public:
  FileLock(const std::string filename, size_t offset = 0u);
  ~FileLock();
  FileLock(const FileLock& other) = delete;
  FileLock& operator=(const FileLock& rhs) = delete;

 public:
  void lock();
  bool try_lock();
  void unlock();

 public:
#if ! _WIN32
  const int fd_;
  const size_t offset_;
#endif
};

}  // namespace ai
}  // namespace vitis
