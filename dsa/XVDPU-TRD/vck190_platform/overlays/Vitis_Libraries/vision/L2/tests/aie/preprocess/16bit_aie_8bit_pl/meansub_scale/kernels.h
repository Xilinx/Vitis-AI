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

#include "config.h"

#if OP_MODE == 0
void pp_top(input_window_int16* input, output_window_int16* output, const float& alpha);
#elif OP_MODE == 1
void pp_top(input_window_int16* input, output_window_int16* output, const float& alpha, const float& beta);
#else
void pp_top(
    input_window_int16* input, output_window_int16* output, const float& alpha, const float& beta, const float& gamma);
#endif
