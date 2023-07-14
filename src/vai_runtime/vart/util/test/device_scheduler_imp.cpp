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
#include "./device_scheduler_imp.hpp"

#include <glog/logging.h>

#include <algorithm>
#include <iostream>
#include <numeric>
using namespace std;

#include <vector>
namespace {
DeviceSchedulerImp::DeviceSchedulerImp(int count)
    : DeviceScheduler(), busy_time_((size_t)count), mtx_{} {
  CHECK_EQ(busy_time_.size(), (size_t)count);
  LOG(INFO) << "name = null";
}

DeviceSchedulerImp::DeviceSchedulerImp(const char* name, int count)
    : DeviceScheduler(), busy_time_((size_t)count), mtx_{} {
  CHECK_EQ(busy_time_.size(), (size_t)count);
  LOG(INFO) << "name = " << name;
}

int DeviceSchedulerImp::next() {
  std::lock_guard<std::mutex> lock(mtx_);
  auto it = std::min_element(busy_time_.begin(), busy_time_.end());
  CHECK(it != busy_time_.end());
  auto v = *it;
  // try to avoid round trip
  for (auto& x : busy_time_) {
    x = x - v;
  }
  *it = (*it) + 1;
  int ret = (int) std::distance(busy_time_.begin(), it);
  return ret;
}

void DeviceSchedulerImp::mark_busy_time(int device_id, int time) {
  std::lock_guard<std::mutex> lock(mtx_);
  CHECK_GE(busy_time_[device_id], 0);
  busy_time_[device_id] = busy_time_[device_id] + time;
  CHECK_GE(busy_time_[device_id], 0);
}
void DeviceSchedulerImp::initialize() {
  c = c + 1;
  if (c != 1) {
    std::cerr << __FILE__ << ":" << __LINE__ << ": [" << __FUNCTION__ << "]"  //
              << "error! initialize twice";
    abort();
  }
}

}  // namespace

DECLARE_INJECTION(vart::dpu::DeviceScheduler, DeviceSchedulerImp,  //
                  int&&);
DECLARE_INJECTION(vart::dpu::DeviceScheduler, DeviceSchedulerImp,  //
                  const char*&&, int&&);

DECLARE_INJECTION_IN_SHARED_LIB(vart::dpu::DeviceScheduler,
                                DeviceSchedulerImp,  //
                                const char*&&, int&&);
