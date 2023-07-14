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
#include <string>
#include <vector>
namespace vart {
namespace dpu {
enum class RegType {
  CODE = 0,
  // MSVC NOTE:: CONST seems a predefined macro on windows.
  XCONST = 1,
  DATA = 2,
};
struct DpuReg {
  explicit DpuReg(const std::string& name, RegType type,
                  const std::vector<char>& value)
      : name_{name}, type_{type}, size_{value.size()}, value_(value) {}
  explicit DpuReg(const std::string& name, size_t size)
      : name_{name}, type_{RegType::DATA}, size_{size}, value_{} {}

 public:
  std::string name_;
  RegType type_;
  const size_t size_;
  const std::vector<char> value_;
};
}  // namespace dpu
}  // namespace vart
