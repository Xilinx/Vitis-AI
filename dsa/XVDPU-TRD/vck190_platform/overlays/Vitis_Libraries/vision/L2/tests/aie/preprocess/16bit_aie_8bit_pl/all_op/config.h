/*
 * Copyright 2021 Xilinx, Inc.
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

#ifndef __CONFIG_H_
#define __CONFIG_H_

#include <common/xf_aie_const.hpp>
#define PARALLEL_FACTOR_16b 16

#define OP_MODE 2 // 0 -> Mean Subtraction             (x - alpha)
                  // 1 -> Mean Subtraction and scale   (x - alpha) * beta
                  // 2 -> all op                       ((x - alpha) * beta) + gamma

// tile dimensions are normally computed by tiler but we need to
// hardcode these values to set the graph window sizes

using DATA_TYPE = int16_t;
static constexpr int TILE_WIDTH = 480;
static constexpr int TILE_HEIGHT = 8;
static constexpr int TILE_ELEMENTS = (TILE_WIDTH * TILE_HEIGHT);
static constexpr int TILE_WINDOW_SIZE = ((TILE_ELEMENTS * sizeof(DATA_TYPE)) + xf::cv::aie::METADATA_SIZE);

#define MAX_TILE_WIDTH 480
#define MAX_TILE_HEIGHT 8
#define VECTORIZATION_FACTOR PARALLEL_FACTOR_16b

#endif //__CONFIG_H_
