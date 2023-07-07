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
#include <memory>
#include <string>
namespace vitis {
namespace xir {
class buffer_object_fd {
 public:
  static std::shared_ptr<buffer_object_fd> create(const std::string& name,
                                                  int flags);

 public:
  explicit buffer_object_fd(const std::string& name, int flags);
  buffer_object_fd(const buffer_object_fd&) = delete;
  buffer_object_fd& operator=(const buffer_object_fd& other) = delete;
  virtual ~buffer_object_fd();

  int fd() { return fd_; }

 private:
  int fd_;
};
}  // namespace xir
}  // namespace vitis
