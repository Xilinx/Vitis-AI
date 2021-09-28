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

#define FILTER_SIZE_3 0
#define FILTER_SIZE_5 1
#define FILTER_SIZE_7 0

#define RO 0
#define NO 1

#define T_8U 0  // Input type of 8U
#define T_16U 0 // Input type of 16U
#define T_16S 1 // Input type of 16S

#if FILTER_SIZE_3
#define FILTER_WIDTH 3
#elif FILTER_SIZE_5
#define FILTER_WIDTH 5
#elif FILTER_SIZE_7
#define FILTER_WIDTH 7
#endif

#define XF_USE_URAM false

#define INPUT_PTR_WIDTH 256
#define OUTPUT_PTR_WIDTH 256
