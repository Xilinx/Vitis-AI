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

#include "xir/cxir.h"

inline std::string conv_to_std_string(xir_string_t k) {
  std::string v(k.data, k.data + k.size);
  return v;
}
inline xir_string_t conv_to_xir_string(const std::string& s) {
  return xir_string_t{s.data(), s.size()};
}
