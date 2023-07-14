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
#include <cstdint>
#include <vector>

#include "vitis/ai/preprocess.hpp"

using std::vector;

namespace vitis {
namespace ai {

void any_scale_mean_c(const uint8_t *input, unsigned int width,
                      unsigned int height, unsigned int ch,
                      vector<float> scales, vector<float> means,
                      int8_t *output) {
  for (unsigned int i = 0; i < width * height; ++i)
    for (unsigned int c = 0; c < ch; ++c) {
      unsigned int index = i * ch + c;
      output[index] = (int8_t)(((float)input[index] - means[c]) * scales[c]);
    }
}

void no_scale_mean_128_c(const uint8_t *input, unsigned int width,
                         unsigned int height, unsigned int ch, int8_t *output) {
  unsigned int size = ch * height * width;
  unsigned int aligned = size & (-8);

  const uint64_t *input8 = (const uint64_t *)input;
  int64_t *output8 = (int64_t *)output;

  for (unsigned int i = 0; i < aligned / 8; ++i) {
    *(output8++) = (*(input8++)) ^ 0x8080808080808080;
  }

  input += aligned;
  output += aligned;
  for (unsigned int i = 0; i < size - aligned; ++i) {
    *(output++) = (*(input++)) ^ 0x80;
  }
}
}  // namespace ai
}  // namespace vitis
