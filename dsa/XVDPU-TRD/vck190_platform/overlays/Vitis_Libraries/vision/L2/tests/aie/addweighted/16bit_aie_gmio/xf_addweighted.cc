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

#include "imgproc/xf_addweighted_aie.hpp"

void addweighted(input_window_int16* input1,
                 input_window_int16* input2,
                 output_window_int16* output,
                 const float& alpha,
                 const float& beta,
                 const float& gamma) {
    xf::cv::aie::addweighted_api(input1, input2, output, alpha, beta, gamma);
};
