/*
 * Copyright 2019 Xilinx Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef __NEONOPT__H
#define __NEONOPT__H

#include <arm_neon.h>
#include <cmath>
#include <cstdint>

/**
 * @brief transform the format of input image(3 channels) in DPU order using arm neon
 *
 * @param dst - pointer to target data
 * @param src - pointer to source image data
 * @param h - height of input input iamge
 * @param w - width of input input iamge
 * @param shift - point to the shift value array, size equals to channel
 * @param scale - scale value of each pixel
 * @param stride - the gap in bytes between two neighbour rows of image data
 *
 * @return void
 */
//void dpuProcessNormalizion(int8_t* dst, uint8_t* src, int h, int w, float* shift, float scale,
//                        int stride);

/**
 * @brief 4-class softmax
 *
 * @param output - pointer to target data
 * @param input - pointer to source data
 * @param scale - scale value in softmax
 * @param group - how many array to be calculated
 *
 * @return void
 */
void neon_softmax4(float* output, const int8_t* input, float scale, unsigned int group);
/**
 * @brief 2-class softmax
 *
 * @param output - pointer to target data
 * @param input - pointer to source data
 * @param scale - scale value in softmax
 * @param group - how many array to be calculated
 *
 * @return void
 */
void neon_softmax2(float* output, const int8_t* input, float scale, unsigned int group);

/**
 * @brief batch softmax using CPU
 *
 *
 * @param output - pointer to target data
 * @param input - pointer to source data
 * @param size - the length of each array
 * @param scale - scale value in softmax
 * @param batch - how many array to be calculated
 *
 * @return void
 */
void softmax_batch(float* output, const int8_t* input, unsigned int size, float scale,
                    unsigned int batch);
#endif
