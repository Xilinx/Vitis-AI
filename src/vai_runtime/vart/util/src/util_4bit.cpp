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

#include "vart/util_4bit.hpp"

namespace vart {

void bump_idx(std::vector<int32_t>& idx, const std::vector<int32_t>& shape) {
  int dim = idx.size() - 1;
  do {
    if ((++idx[dim]) == shape[dim]) {
      idx[dim] = 0;
      dim--;
    } else {
      dim = -1;
    }
  } while (dim >= 0);
}

void bump_idx(std::vector<int32_t>& idx, const std::vector<uint32_t>& shape) {
  int dim = idx.size() - 1;
  do {
    if ((++idx[dim]) == static_cast<int32_t>(shape[dim])) {
      idx[dim] = 0;
      dim--;
    } else {
      dim = -1;
    }
  } while (dim >= 0);
}

}  // namespace vart
