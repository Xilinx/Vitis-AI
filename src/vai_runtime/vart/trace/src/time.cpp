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

#include "time.hpp"

namespace vitis::ai::trace {

inline uint64_t get_tsc(void) {
#if defined(__i386__) or defined(__x86_64__)
  unsigned long long high, low;
  asm volatile("rdtsc" : "=a"(low), "=d"(high));
  return (low + (high << 32));
#else
  return 0;
#endif
};

double get_ts(void) {
  auto tp = std::chrono::steady_clock::now().time_since_epoch();
  auto ts = std::chrono::duration_cast<
                std::chrono::duration<double, std::ratio<1, 1>>>(tp)
                .count();

  return ts;
}

double get_xrt_ts(void) {
#ifdef ENABLE_XRT_TIMESTAMP
  auto xrt_time_ns = xrt_core::time_ns();
  double xrt_time_s = xrt_time_ns / 1000.0 / 1000.0 / 1000.0;

  return xrt_time_s;
#else
  return get_ts();
#endif
};

}  // namespace vitis::ai::trace
