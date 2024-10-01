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
#ifndef DEEPHI_PROFILING_HPP_
#define DEEPHI_PROFILING_HPP_

#include <glog/logging.h>
#include <chrono>
#include "./env_config.hpp"
DEF_ENV_PARAM(DEEPHI_PROFILING, "0");

namespace vitis {
namespace ai {
using Clock = std::chrono::steady_clock;

#define __TIC__(tag)                                                           \
  auto __##tag##_start_time =                                                  \
      ENV_PARAM(DEEPHI_PROFILING)                                              \
          ? vitis::ai::Clock::now()                                            \
          : std::chrono::time_point<vitis::ai::Clock>();

#define __TOC__(tag)                                                           \
  auto __##tag##_end_time = ENV_PARAM(DEEPHI_PROFILING)                        \
                                ? vitis::ai::Clock::now()                      \
                                : std::chrono::time_point<vitis::ai::Clock>(); \
  LOG_IF(INFO, ENV_PARAM(DEEPHI_PROFILING))                                    \
      << #tag << " : "                                                         \
      << std::chrono::duration_cast<std::chrono::microseconds>(                \
             __##tag##_end_time - __##tag##_start_time)                        \
             .count()                                                          \
      << "us";

#define __TOC_FLEX__(tag, level, timescale)                                    \
  auto __##tag##_end_time = ENV_PARAM(DEEPHI_PROFILING)                        \
                                ? vitis::ai::Clock::now()                      \
                                : std::chrono::time_point<vitis::ai::Clock>(); \
  LOG_IF(level, ENV_PARAM(DEEPHI_PROFILING))                                   \
      << #tag << " : "                                                         \
      << std::chrono::duration_cast<std::chrono::timescale>(                   \
             __##tag##_end_time - __##tag##_start_time)                        \
             .count()                                                          \
      << " " << #timescale;

#define __TIC_SUM__(tag)                                                       \
  static auto __##tag##_total_time = 0U;                                       \
  auto __##tag##_start_time =                                                  \
      ENV_PARAM(DEEPHI_PROFILING)                                              \
          ? vitis::ai::Clock::now()                                            \
          : std::chrono::time_point<vitis::ai::Clock>();

#define __TOC_SUM__(tag)                                                       \
  auto __##tag##_end_time = ENV_PARAM(DEEPHI_PROFILING)                        \
                                ? vitis::ai::Clock::now()                      \
                                : std::chrono::time_point<vitis::ai::Clock>(); \
  LOG_IF(INFO, ENV_PARAM(DEEPHI_PROFILING))                                    \
      << #tag << " : "                                                         \
      << (__##tag##_total_time +                                               \
          std::chrono::duration_cast<std::chrono::microseconds>(               \
              __##tag##_end_time - __##tag##_start_time)                       \
              .count())                                                        \
      << "us";

}  // namespace ai
}  // namespace vitis

#endif
