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
namespace vitis {
namespace ai {
/** @brief the softmax function.
 * the input is fix-point, and the output is floating point
 */
void softmax(const int8_t *input, float scale, unsigned int cls,
             unsigned int group, float *output);
/** @Method for float input data
 * Used for DPUV1
 */
void softmax(const float *input, float scale, unsigned int cls,
             unsigned int group, float *output);
/** @brief `yuv2bgr` converts a raw image from YUV422 to BGR, include
 * cropping.
 */
void yuv2bgr(int left, int top,                            //
             int width, int height,                        //
             unsigned char *__restrict y, int stride_y,    //
             unsigned char *__restrict uv, int stride_uv,  //
             unsigned char *bgr);

/** @brief reshape a feature map
 *
 *  input(iy, ix, c*tile_size + ty*tile_dim + tx) -> output(iy, ix, ty, tx, c)
 *
 * where  tile_size = tile_dim * tile_dim;
 */
void tiling(const int8_t *input, unsigned int width, unsigned int height,
            unsigned int tile_dim, unsigned int ch, int8_t *output);
}  // namespace ai
}  // namespace vitis
