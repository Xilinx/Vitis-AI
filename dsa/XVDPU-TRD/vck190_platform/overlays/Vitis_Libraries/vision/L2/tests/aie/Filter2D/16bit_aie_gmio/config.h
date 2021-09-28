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

// tile dimensions are normally computed by tiler but we need to
// hardcode these values to set the graph window sizes
#define TILE_WIDTH 256
#define TILE_HEIGHT 16
#define TILE_BUFFER_SIZE (SMARTTILE_ELEMENTS + (TILE_WIDTH * TILE_HEIGHT))
#define TILE_WINDOW_SIZE (sizeof(int16_t) * TILE_BUFFER_SIZE)

/* Graph specific configuration */
#define IMG_WIDTH 1920
#define IMG_HEIGHT 1080

#define MAX_TILE_WIDTH 256
#define MAX_TILE_HEIGHT 16
#define VECTORIZATION_FACTOR PARALLEL_FACTOR_16b

#endif //__CONFIG_H_
