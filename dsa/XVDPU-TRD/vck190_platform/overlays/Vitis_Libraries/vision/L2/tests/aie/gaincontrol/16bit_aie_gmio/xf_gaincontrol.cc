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
#include <imgproc/xf_gaincontrol_aie.hpp>

template <int code>
void gaincontrol(input_window_int16* input, output_window_int16* output, const int16_t& rgain, const int16_t& bgain) {
    xf::cv::aie::gaincontrol_api<code>(input, output, rgain, bgain);
};
