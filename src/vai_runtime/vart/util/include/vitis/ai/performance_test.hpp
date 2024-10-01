
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

#include <future>
#include <memory>
#include <mutex>
#include <vector>
#include <vitis/ai/env_config.hpp>
DEF_ENV_PARAM(SLEEP_MS, "60000");
namespace vitis {
namespace ai {
class PerformanceTestRunner {
 public:
  explicit PerformanceTestRunner() = default;
  virtual ~PerformanceTestRunner() = default;
  PerformanceTestRunner(const PerformanceTestRunner& other) = delete;
  PerformanceTestRunner& operator=(const PerformanceTestRunner& rhs) = delete;

 public:
  virtual void before() {}
  virtual void step(size_t step, int thread_id) = 0;
  virtual void after(){};
  virtual size_t get_result() = 0;
};

class PerformanceTest {
 public:
  using Clock = std::chrono::steady_clock;
  std::mutex mtx_;
  static std::unique_ptr<PerformanceTestRunner> thread_main(
      PerformanceTest* me, std::unique_ptr<PerformanceTestRunner>&& runner,
      int* stop, int thread_id) {
    runner->before();
    { std::lock_guard<std::mutex> lock(me->mtx_); }
    auto step = 0u;
    while (!*stop) {
      runner->step(step, thread_id);
      step = step + 1;
    }
    runner->after();
    return std::move(runner);
  }
  int main(int argc, char* argv[],
           std::vector<std::unique_ptr<PerformanceTestRunner>>&& runners) {
    std::vector<std::future<std::unique_ptr<PerformanceTestRunner>>> threads;
    int stop = 0;
    std::unique_lock<std::mutex> lock_main(mtx_);
    for (auto i = 0u; i < runners.size(); ++i) {
      threads.emplace_back(std::async(std::launch::async, thread_main, this,
                                      std::move(runners[i]), &stop, i));
    }
    lock_main.unlock();
    auto start = Clock::now();
    auto sleep_ms = ENV_PARAM(SLEEP_MS);
    LOG(INFO) << "0% ...";
    for (auto i = 0; i < 10; ++i) {
      std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms / 10));
      LOG(INFO) << (i + 1) * 10 << "% ...";
    }
    stop = 1;
    LOG(INFO) << "stop and waiting for all threads terminated....";
    auto t1 = Clock::now();
    size_t total = 0;
    auto i = 0;
    for (auto& t : threads) {
      runners[i] = std::move(t.get());
      LOG(INFO) << "thread-" << i << " processes " << runners[i]->get_result()
                << " frames";
      total = total + runners[i]->get_result();
      i = i + 1;
    }
    auto t2 = Clock::now();
    auto f = (float)total;
    auto time = (float)time_diff(start, t2).count();
    LOG(INFO) << "it takes " << time_diff(t1, t2).count() << " us for shutdown";
    LOG(INFO) << "FPS= " << f / time * 1.0e6 << " number_of_frames= " << f
              << " time= " << time / 1.0e6 << " seconds.";
    LOG(INFO) << "BYEBYE";
    return 0;
  }
  template <typename T>
  static std::chrono::microseconds time_diff(const T& t1, const T& t2) {
    return std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
  }
};

}  // namespace ai
}  // namespace vitis
