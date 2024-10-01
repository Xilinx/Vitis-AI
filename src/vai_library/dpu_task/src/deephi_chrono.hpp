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
#include <chrono>
#include <ratio>
namespace vitis {
namespace chrono {
class cpu_clock {
 public:
  using duration = std::chrono::duration<uint64_t, std::ratio<1, 100000000>>;
  using rep = duration::rep;
  using period = duration::period;
  using time_point = std::chrono::time_point<cpu_clock, duration>;
  static constexpr bool is_steady = true;

 public:
  static time_point now() noexcept {
    return time_point(duration(now_internal()));
  }

 private:
  static inline uint64_t now_internal() {
#if defined(__aarch64__)
    uint64_t cntvct_el0;
    asm volatile("mrs %0, cntvct_el0" : "=r"(cntvct_el0));
    return cntvct_el0;
#elif defined(__ARM_ARCH)
#if (__ARM_ARCH >= 6)  // V6 is the earliest arch that has a standard cyclecount
    uint32_t pmccntr;
    uint32_t pmuseren;
    uint32_t pmcntenset;
    // Read the user mode perf monitor counter access permissions.
    asm volatile("mrc p15, 0, %0, c9, c14, 0" : "=r"(pmuseren));
    if (pmuseren & 1) {  // Allows reading perfmon counters for user mode code.
      asm volatile("mrc p15, 0, %0, c9, c12, 1" : "=r"(pmcntenset));
      if (pmcntenset & 0x80000000ul) {  // Is it counting?
        asm volatile("mrc p15, 0, %0, c9, c13, 0" : "=r"(pmccntr));
        // The counter is set up to count every 64th cycle
        return static_cast<int64_t>(pmccntr) * 64;  // Should optimize to << 6
      }
    }
#endif
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return static_cast<uint64_t>(tv.tv_sec) * 1000000 + tv.tv_usec;
#else
#error "NOT SUPPORTED"
#endif
    return 0;
  }
};
}  // namespace chrono
}  // namespace vitis
