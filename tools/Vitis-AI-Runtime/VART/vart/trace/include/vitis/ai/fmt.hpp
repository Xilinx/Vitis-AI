/*
 * Copyright 2021 Xilinx Inc.
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
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

namespace vitis::ai::trace {

inline std::string to_string(const char* s_) { return std::string(s_); };

inline std::string to_string(std::string s_) { return std::string(s_); };

inline std::string to_string(float s_) {
  std::ostringstream oss;
  oss << std::setiosflags(std::ios::fixed) << std::setprecision(9) << s_;
  return oss.str();
};
inline std::string to_string(double s_) {
  std::ostringstream oss;
  oss << std::setiosflags(std::ios::fixed) << std::setprecision(9) << s_;
  return oss.str();
};

template <typename T>
std::string to_string(T&& s_) {
  return std::to_string(s_);
};
}  // namespace vitis::ai::trace
