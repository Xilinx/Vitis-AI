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
//------------------------------------------------------------------------------
// UUT DEFAULT CONFIGURATION
#ifndef DATA_TYPE
#define DATA_TYPE cfloat
#endif
#ifndef TWIDDLE_TYPE
#define TWIDDLE_TYPE cfloat
#endif
#ifndef POINT_SIZE
#define POINT_SIZE 64
#endif
#ifndef FFT_NIFFT
#define FFT_NIFFT 1
#endif
#ifndef SHIFT
#define SHIFT 0
#endif
#ifndef CASC_LEN
#define CASC_LEN 1
#endif
#ifndef DYN_PT_SIZE
#define DYN_PT_SIZE 1
#endif
#ifndef WINDOW_VSIZE
#define WINDOW_VSIZE 1024
#endif
#ifndef INPUT_FILE
#define INPUT_FILE "data/input.txt"
#endif
#ifndef OUTPUT_FILE
#define OUTPUT_FILE "data/output.txt"
#endif

#ifndef NITER
#define NITER 1
#endif

#ifndef INPUT_WINDOW_VSIZE
#if DATA_TYPE == cint16
#define INPUT_WINDOW_VSIZE ((8 * DYN_PT_SIZE) + POINT_SIZE)
#else
#define INPUT_WINDOW_VSIZE ((4 * DYN_PT_SIZE) + POINT_SIZE)
#endif
#endif

#define INPUT_SAMPLES INPUT_WINDOW_VSIZE* NITER
#define OUTPUT_SAMPLES INPUT_WINDOW_VSIZE* NITER

// END OF UUT CONFIGURATION
//------------------------------------------------------------------------------
