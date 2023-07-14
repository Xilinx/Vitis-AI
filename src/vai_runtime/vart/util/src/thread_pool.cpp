
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

#include "vitis/ai/thread_pool.hpp"

#include <glog/logging.h>

#include "vitis/ai/env_config.hpp"
DEF_ENV_PARAM(DEBUG_THREAD_POOL, "0")
namespace vitis {
namespace ai {
std::unique_ptr<ThreadPool> ThreadPool::create(size_t num_of_threads) {
  return std::unique_ptr<ThreadPool>(new ThreadPool(num_of_threads));
}

ThreadPool::ThreadPool(size_t num_of_threads) : queue_(10u) {
  running_ = 1;
  pool_.reserve(num_of_threads);
  for (auto i = 0u; i < num_of_threads; ++i) {
    pool_.emplace_back(thread_main, this);
  }
}

ThreadPool::~ThreadPool() {
  running_ = 0;
  LOG_IF(INFO, ENV_PARAM(DEBUG_THREAD_POOL))
      << "@" << (void*)this << " waiting for all threads terminated";
  for (auto& t : pool_) {
    t.join();
  }
  LOG_IF(INFO, ENV_PARAM(DEBUG_THREAD_POOL)) << "@" << (void*)this << " byebye";
}

void ThreadPool::thread_main(ThreadPool* self) {
  LOG_IF(INFO, ENV_PARAM(DEBUG_THREAD_POOL))
      << "@" << (void*)self << " thread started";
  while (self->running_) {
    auto action = self->queue_.recv(std::chrono::milliseconds(500));
    if (action) {
      // LOG(INFO) << "start action ";
      (*action)();
    }
  }
  LOG_IF(INFO, ENV_PARAM(DEBUG_THREAD_POOL)) << "thread ended";
  return;
}
}  // namespace ai
}  // namespace vitis
