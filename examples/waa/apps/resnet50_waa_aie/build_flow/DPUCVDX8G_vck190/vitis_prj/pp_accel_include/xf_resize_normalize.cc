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

#include "kernels.h"
#include "imgproc/xf_resize_normalize.hpp"

template <int WIDTH_IN, int HEIGHT_IN, int WIDTH_OUT, int HEIGHT_OUT, int IMG_HEIGHT_OUT>
void ResizeNormRunner<WIDTH_IN, HEIGHT_IN, WIDTH_OUT, HEIGHT_OUT, IMG_HEIGHT_OUT>::run(input_window<uint8_t>* input,
                                                                                       output_window<int8_t>* output,
                                                                                       int a0,
                                                                                       int a1,
                                                                                       int a2,
                                                                                       int a3,
                                                                                       int b0,
                                                                                       int b1,
                                                                                       int b2,
                                                                                       int b3) {
    uint8* img_in_ptr = (uint8*)input->ptr;
    int8* img_out_ptr = (int8*)output->ptr;

    xf::cv::aie::ResizeNorm<TILE_WIDTH_IN, TILE_HEIGHT_IN, TILE_WIDTH_OUT, TILE_HEIGHT_OUT, IMG_HEIGHT_OUT> resize_norm(
        mPos, mwtsX, mwtsY);
    //resize_norm.runImpl(img_in_ptr, img_out_ptr, row, a0, a1, a2, a3, b0, b1, b2, b3);
    resize_norm.runImpl(input, output, a0, a1, a2, a3, b0, b1, b2, b3);
}
