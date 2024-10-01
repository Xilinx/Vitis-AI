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
#ifndef TIME_HPP_
#define TIME_HPP_

#include <chrono>
#include <iostream>

using Clock = std::chrono::high_resolution_clock;

#define __TIC__(tag) auto __##tag##_start_time = Clock::now();

#define __TOC__(tag)                                                  \
  auto __##tag##_end_time = Clock::now();                             \
  std::cout << #tag << " : "                                          \
            << std::chrono::duration_cast<std::chrono::microseconds>( \
                   __##tag##_end_time - __##tag##_start_time)         \
                   .count()                                           \
            << std::endl;

#endif
