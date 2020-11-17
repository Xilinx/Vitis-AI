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
#include "./device_scheduler_imp.hpp"

#include <glog/logging.h>

#include <algorithm>
#include <numeric>
#include <vector>
#include <vitis/ai/env_config.hpp>
DEF_ENV_PARAM(DEBUG_DEVICE_SCHEDULER, "0");

namespace {
DeviceSchedulerImp::DeviceSchedulerImp(int count)
    : DeviceScheduler(), queue_length_((size_t)count), mtx_{} {
  CHECK_EQ(queue_length_.size(), (size_t)count);
}

static std::string to_string(const std::vector<int>& v) {
  std::ostringstream str;
  str << "[";
  int c = 0;
  for (auto& x : v) {
    if (c++ != 0) {
      str << ",";
    }
    str << x;
  }
  str << "]";
  return str.str();
}

int DeviceSchedulerImp::next() {
  std::lock_guard<std::mutex> lock(mtx_);
  auto it = std::min_element(queue_length_.begin(), queue_length_.end());
  CHECK(it != queue_length_.end());
  // auto v = *it;
  *it = (*it) + 1;
  LOG_IF(INFO, ENV_PARAM(DEBUG_DEVICE_SCHEDULER))
      << "currect id: " << *it << " total size: " << queue_length_.size();
  auto ret = std::distance(queue_length_.begin(), it);
  LOG_IF(INFO, ENV_PARAM(DEBUG_DEVICE_SCHEDULER))
      << "return " << ret << " v=" << to_string(queue_length_);
  return ret;
}

void DeviceSchedulerImp::mark_busy_time(int device_id, int time) {
  std::lock_guard<std::mutex> lock(mtx_);
  CHECK_GE(queue_length_[device_id], 0);
  queue_length_[device_id] = queue_length_[device_id] - 1;  // + time;
  CHECK_GE(queue_length_[device_id], 0);
  LOG_IF(INFO, ENV_PARAM(DEBUG_DEVICE_SCHEDULER))
      << "release " << device_id << " v=" << to_string(queue_length_);
}

//  static DeviceSchedulerImp::Injector<DeviceSchedulerImp> g_injector;
}  // namespace

DECLARE_INJECTION(vart::dpu::DeviceScheduler, DeviceSchedulerImp, int&&);
DECLARE_INJECTION(vart::dpu::DeviceScheduler, DeviceSchedulerImp, size_t&&);
