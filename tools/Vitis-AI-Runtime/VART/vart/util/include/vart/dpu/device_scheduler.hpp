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
#include <functional>
#include <memory>
#include <vitis/ai/with_injection.hpp>

namespace vart {
namespace dpu {
/**
 * @brief a device memory management
 */
class DeviceScheduler : public vitis::ai::WithInjection<DeviceScheduler> {
 protected:
  explicit DeviceScheduler() = default;

 public:
  DeviceScheduler(const DeviceScheduler&) = delete;
  DeviceScheduler& operator=(const DeviceScheduler& other) = delete;

  virtual ~DeviceScheduler() = default;

 public:
  /** get the next device id
   *
   *  this function has side effect, i.e. the busy time of the
   *  corresponding device is implicitly increased by one, so that
   *  even the used_time is not invoked, the device is scheduled in
   *  round robin manner.
   */
  virtual int next() = 0;

  /** mark the busy time.
   *
   * the unit of the time does not matter as long as it is always same.
   *
   * @param device_id the corresponding device
   * @param time the time used by the device.
   */
  virtual void mark_busy_time(int device_id, int time) = 0;
};
}  // namespace dpu
}  // namespace vart
