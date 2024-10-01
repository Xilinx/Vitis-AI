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
#include <sstream>
namespace vitis {
namespace ai {
inline std::string xxd(const unsigned char* p, int size, int column,
                       int group) {
  std::ostringstream str;
  char buf[128];
  for (int i = 0; i < size; ++i) {
    if (i % column == 0) {
      snprintf(buf, sizeof(buf), "\n%p %08x:", p + i, i);
      str << buf;
    }
    if (i % group == 0) {
      snprintf(buf, sizeof(buf), " ");
      str << buf;
    }
    snprintf(buf, sizeof(buf), "%02x", p[i]);
    str << buf;
  }
  snprintf(buf, sizeof(buf), "\n");
  str << buf;
  return str.str();
}
}  // namespace ai
}  // namespace vitis
