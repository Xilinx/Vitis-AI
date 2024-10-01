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
#ifndef PROFILING_TEST_HPP_
#define PROFILING_TEST_HPP_

#include <chrono>
#include <iostream>

#define __TIC__(tag) \
  auto __##tag##_start_time = std::chrono::steady_clock::now();

#define __TOC__(tag)                                                  \
  auto __##tag##_end_time = std::chrono::steady_clock::now();         \
  std::cout << #tag << " : "                                          \
            << std::chrono::duration_cast<std::chrono::microseconds>( \
                   __##tag##_end_time - __##tag##_start_time)         \
                   .count()                                           \
            << "us" << std::endl;

#define __TIC_SUM__(tag)                 \
  static auto __##tag##_total_time = 0U; \
  auto __##tag##_start_time = std::chrono::steady_clock::now();

#define __TOC_SUM__(tag)                                      \
  auto __##tag##_end_time = std::chrono::steady_clock::now(); \
  __##tag##_total_time +=                                     \
      std::chrono::duration_cast<std::chrono::microseconds>(  \
          __##tag##_end_time - __##tag##_start_time)          \
          .count();                                           \
  std::cout << #tag << " : " << __##tag##_total_time << "us" << std::endl;

#endif
