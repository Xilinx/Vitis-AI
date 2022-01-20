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

#include <adf/window/types.h>
#include <adf/stream/types.h>

// Thresholding types
enum _threshold_type {
    XF_THRESHOLD_TYPE_BINARY = 0,
    XF_THRESHOLD_TYPE_BINARY_INV = 1,
    XF_THRESHOLD_TYPE_TRUNC = 2,
    XF_THRESHOLD_TYPE_TOZERO = 3,
    XF_THRESHOLD_TYPE_TOZERO_INV = 4,
};
// typedef _threshold_type XF_threshold_type_e;

#define THRESH_TYPE XF_THRESHOLD_TYPE_BINARY

void threshold(input_window_int16* input, output_window_int16* output, const int16& thresh_val, const int16& max_val);
