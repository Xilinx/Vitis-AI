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

// 67d7842dbbe25473c3c32b93c0da8047785f30d78e8a024de1b57352245f9689

#include <adf/window/types.h>
#include <adf/stream/types.h>

#define PARALLEL_FACTOR_32b 16 // Parallelization factor for 32b operations (8x mults)
#define PARALLEL_FACTOR_16b 32 // Parallelization factor for 32b operations (8x mults)

#define OPMODE 0

// void blobFromImage( input_window_float * img_in, input_window_float * restrict img_out,float alpha, float beta, float
// gama,int threshold1,int threshold2);
// void blobFromImage( input_window_float * img_in, input_window_float * restrict img_out,int threshold2);

// void blobFromImage( input_window_int16 * img_in, output_window_int16 * img_out);
void blobFromImage(input_window_float* input, output_window_float* output);
