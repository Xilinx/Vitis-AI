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
#include <map>
#include <mutex>
#include <vector>

#include "vart/dpu/device_scheduler.hpp"

namespace {
class DeviceSchedulerImp : public vart::dpu::DeviceScheduler {
 public:
  explicit DeviceSchedulerImp(int count);
  virtual ~DeviceSchedulerImp() = default;
  DeviceSchedulerImp(const DeviceSchedulerImp&) = delete;
  DeviceSchedulerImp& operator=(const DeviceSchedulerImp& other) = delete;

 private:
  virtual int next() override;
  virtual void mark_busy_time(int device_id, int time) override;

 private:
  std::vector<int> queue_length_;
  std::mutex mtx_;
};
}  // namespace
