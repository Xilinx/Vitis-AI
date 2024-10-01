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
#include <memory>
#include <string>

namespace vitis {
namespace ai {

class Lock {
 protected:
  Lock();

 public:
  Lock(const Lock& other) = delete;
  Lock& operator=(const Lock& rhs) = delete;

  virtual ~Lock();

 public:
  static std::unique_ptr<Lock> create(const std::string& lock_name);

 public:
  virtual bool try_lock() = 0;
  virtual void lock() = 0;
  virtual void unlock() = 0;
  // TODO:
  // virtual string get_pid() = 0;
};

}  // namespace ai
}  // namespace vitis
