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
#include <stdint.h>

#include <cstdlib>
#include <vector>
namespace vitis {
namespace ai {

/// \brief calculate the max index of a WxHxC feature map.
///
/// @param feature_map a feature map, size of buffer is width * height * channel
/// @param width the width of the feature map.
/// @param height the height of the feature map.
/// @param channel the channel of the feature map
/// @return a new feature map whose size is a width * height * 1
///
/// @note: when channel is 2, 4, 8, and 16, neon optimization is applied if any.

void max_index_void(int8_t *feature_map, int width, int height, int channel,
                    uint8_t *results);
std::vector<uint8_t> max_index(int8_t *feature_map, int width, int height,
                               int channel);
// max_index(int8_t * feature_map, int width, int height, int channel);
void max_index_c(int8_t *d, int c, int g, uint8_t *results);
}  // namespace ai
}  // namespace vitis
