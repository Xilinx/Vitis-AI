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

#define T_8U 1  // Input type of 8U
#define T_16S 0 // Input type of 16S

/*  set the optimisation type  */
#define NO 0 // Normal Operation
#define RO 1 // Resource Optimized

#define GRAY 1

#define ARRAY 0
#define SCALAR 1
// macros for accel
#define FUNCT_NUM 1
//#define EXTRA_ARG 0.05
#define EXTRA_PARM XF_CONVERT_POLICY_SATURATE

// OpenCV reference macros
#define CV_FUNCT_NAME add
//#define CV_EXTRA_ARG 0.05
