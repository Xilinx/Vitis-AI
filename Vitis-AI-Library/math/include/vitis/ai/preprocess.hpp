/*
 * Copyright 2019 Xilinx Inc.
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
#ifndef DPMAP_PREPROCESS_HPP__
#define DPMAP_PREPROCESS_HPP__

#include <cstdint>
#include <vector>

namespace vitis {
namespace ai {

void any_scale_mean_c(const uint8_t *input, unsigned int width,
                      unsigned int height, unsigned int ch,
                      std::vector<float> scales, std::vector<float> means,
                      int8_t *output);

void no_scale_mean_128_c(const uint8_t *input, unsigned int width,
                         unsigned int height, unsigned int ch, int8_t *output);
} // namespace ai
} // namespace vitis

#endif
