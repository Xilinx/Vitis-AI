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
#ifdef USE_NEON
#include <arm_neon.h>

#include <cstdint>
#include <vector>

using std::vector;

namespace vitis {
namespace ai {

void power_scale_mean_neon(const uint8_t *input, unsigned int width,
                           unsigned int height, unsigned int ch,
                           int right_shift, vector<int> means, int8_t *output) {
  // int8x16_t  vshrq_n_s8(int8x16_t a, __constrange(1,8) int b);
}

void power_scale_no_mean_neon(const uint8_t *input, unsigned int width,
                              unsigned int height, unsigned int ch,
                              int right_shift, int8_t *output) {
  // int16x8_t  vmovl_s8(int8x8_t a)
  // int8x8_t   vqshrn_n_s16(int16x8_t a, __constrange(1,8) int b);
}
}  // namespace ai
}  // namespace vitis

#endif
