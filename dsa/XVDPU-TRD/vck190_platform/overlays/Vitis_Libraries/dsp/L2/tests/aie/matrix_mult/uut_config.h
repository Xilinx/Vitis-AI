/*
 * Copyright 2021 Xilinx, Inc.
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
// This file holds constants defining default values for the configuration
// of the parameterized graph class.
// This file further was required to capture an extern declaration
// of the specific class defined by these particular values of the generic
// class, as this was required before aiecompiler supported generic
// class definitions.
//------------------------------------------------------------------------------
// UUT CONFIGURATION
#ifndef T_DATA_A
#define T_DATA_A cint16
#endif
#ifndef T_DATA_B
#define T_DATA_B cint16
#endif
#ifndef P_DIM_A
#define P_DIM_A 16
#endif
#ifndef P_DIM_AB
#define P_DIM_AB 16
#endif
#ifndef P_DIM_B
#define P_DIM_B 16
#endif
#ifndef P_SHIFT
#define P_SHIFT 16
#endif
#ifndef P_ROUND_MODE
#define P_ROUND_MODE 0
#endif
#ifndef P_DIM_A_LEADING
#define P_DIM_A_LEADING 0
#endif
#ifndef P_DIM_B_LEADING
#define P_DIM_B_LEADING 1
#endif
#ifndef P_DIM_OUT_LEADING
#define P_DIM_OUT_LEADING 0
#endif
#ifndef P_ADD_TILING_A
#define P_ADD_TILING_A 1
#endif
#ifndef P_ADD_TILING_B
#define P_ADD_TILING_B 1
#endif
#ifndef P_ADD_DETILING_OUT
#define P_ADD_DETILING_OUT 1
#endif
#ifndef P_INPUT_WINDOW_VSIZE_A
#define P_INPUT_WINDOW_VSIZE_A P_DIM_A* P_DIM_AB
#endif
#ifndef P_INPUT_WINDOW_VSIZE_B
#define P_INPUT_WINDOW_VSIZE_B P_DIM_B* P_DIM_AB
#endif
#ifndef INPUT_FILE_A
#define INPUT_FILE_A "data/inputA.txt"
#endif
#ifndef INPUT_FILE_B
#define INPUT_FILE_B "data/inputB.txt"
#endif
#ifndef OUTPUT_FILE
#define OUTPUT_FILE "data/output.txt"
#endif
// quick fix for host.cpp
#ifndef REF_OUTPUT_FILE
#define REF_OUTPUT_FILE "data/ref_output.txt"
#endif

#ifndef NITER
#define NITER 1
#endif

#ifndef P_CASC_LEN
#define P_CASC_LEN 1
#endif

#define P_INPUT_SAMPLES_A P_INPUT_WINDOW_VSIZE_A* NITER
#define P_INPUT_SAMPLES_B P_INPUT_WINDOW_VSIZE_B* NITER

#define P_OUTPUT_SAMPLES P_INPUT_WINDOW_VSIZE_A / P_DIM_AB* P_INPUT_WINDOW_VSIZE_B / P_DIM_AB* NITER

// END OF UUT CONFIGURATION
//------------------------------------------------------------------------------
