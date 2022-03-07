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

#define TYPE_FLOW_WIDTH 16
#define TYPE_FLOW_INT 10
#define TYPE_FLOW_TYPE ap_fixed<TYPE_FLOW_WIDTH, TYPE_FLOW_INT>

#define INPUT_PTR_WIDTH 32
#define OUTPUT_PTR_WIDTH 32

#define WINSIZE_OFLOW 11

#define NUM_LEVELS 5
#define NUM_ITERATIONS 5

#define HEIGHT 1080
#define WIDTH 1920

#define NUM_LINES_FINDIT 50

// harris parameters
#define FILTER_WIDTH 3
#define BLOCK_WIDTH 3
#define NMS_RADIUS 1
#define MAXCORNERS 10000