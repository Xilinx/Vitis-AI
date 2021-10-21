/*
 * Copyright 2019 Xilinx, Inc.
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

// Configure this based on the number of rows needed for the remap purpose
// e.g., If its a right to left flip two rows are enough
#define XF_WIN_ROWS 8

#define GRAY 1
#define RGB 0

// The type of interpolation, define "XF_REMAP_INTERPOLATION" as either "XF_INTERPOLATION_NN" or
// "XF_INTERPOLATION_BILINEAR"
#define XF_REMAP_INTERPOLATION XF_INTERPOLATION_NN
#define XF_USE_URAM true
