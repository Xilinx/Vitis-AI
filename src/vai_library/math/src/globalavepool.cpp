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
#include "vitis/ai/globalavepool.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
namespace vitis {
namespace ai {

void globalAvePool(int8_t *src, int channel, int width, int height, int8_t *dst,
                   int num) {
  float sum;
  for (int i = 0; i < channel; i++) {
    sum = 0.0f;
    for (int j = 0; j < width * height; j++) {
      sum += src[i + channel * j];
    }
    int temp = round(((sum / (width * height)) * num));

    dst[i] = (int8_t)std::min(temp, 127);
    // dst[i] = (int8_t)(((sum / (width * height)) * num));
  }
}
}  // namespace ai
}  // namespace vitis
