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

#include <stdint.h>

#include <array>
#include <chrono>
#include <cstring>
#include <limits>
#include <mutex>
#include <ostream>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vitis/ai/lock.hpp>

#include "hiredis/hiredis.h"

namespace vitis {
namespace ai {

class RedisLock : public Lock {
 public:
  RedisLock(const std::string& device_name);
  ~RedisLock();
  RedisLock(const RedisLock& other) = delete;
  RedisLock& operator=(const RedisLock& rhs) = delete;

 public:
  virtual bool try_lock() override;
  virtual void lock() override;
  virtual void unlock() override;

 private:
  void heart_beat();
  bool is_own_locked();
  std::unique_ptr<redisContext> ctx_;
  std::string device_name_;
  std::string token_;
  int timeout_;
  bool is_finished_;
  std::thread heart_beat_thread_;
  std::mutex mutex_;
};

}  // namespace ai
}  // namespace vitis
