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
#include "./file_lock.hpp"

#include <fcntl.h>
#include <glog/logging.h>
#include <sys/stat.h>
#include <sys/types.h>

namespace vitis {
namespace ai {


FileLock::FileLock(const std::string filename, size_t offset) {
  CHECK(false) << "TODO";
}

FileLock::~FileLock() { //CHECK(false) << "TODO"; 
}

void FileLock::lock() {
  CHECK(false) << "TODO";
  return;
}

bool FileLock ::try_lock() {
  auto ret = true;
  CHECK(false) << "TODO";
  return ret;
}

void FileLock::unlock() {
  CHECK(false) << "TODO";
  return;
}

}  // namespace ai
}  // namespace vitis
