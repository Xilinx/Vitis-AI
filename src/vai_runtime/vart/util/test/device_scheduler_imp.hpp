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
#include <glog/logging.h>

#include <map>
#include <mutex>

#include "./device_scheduler.hpp"

namespace {
class DeviceSchedulerImp : public vart::dpu::DeviceScheduler {
 public:
  explicit DeviceSchedulerImp(int count);
  explicit DeviceSchedulerImp(const char* log, int count);
  virtual ~DeviceSchedulerImp() = default;
  DeviceSchedulerImp(const DeviceSchedulerImp&) = delete;
  DeviceSchedulerImp& operator=(const DeviceSchedulerImp& other) = delete;

 private:
  virtual void initialize() override;

 private:
  virtual int next() override;
  virtual void mark_busy_time(int device_id, int time) override;

 private:
  std::vector<int> busy_time_;
  std::mutex mtx_;
  int c = 0;
};
}  // namespace
