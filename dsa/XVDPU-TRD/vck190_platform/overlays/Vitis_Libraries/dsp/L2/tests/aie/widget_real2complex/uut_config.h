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
// of the parameterized Single Rate Asymmetrical FIR filter graph class.
// This file further was required to capture an extern declaration
// of the specific class defined by these particular values of the generic
// class, as this was required before aiecompiler supported generic
// class definitions.
//------------------------------------------------------------------------------
// UUT DEFAULT CONFIGURATION
#ifndef DATA_TYPE
#define DATA_TYPE int16
#endif
#ifndef DATA_OUT_TYPE
#define DATA_OUT_TYPE cint16
#endif
#ifndef WINDOW_VSIZE
#define WINDOW_VSIZE 256
#endif
#ifndef INPUT_FILE
#define INPUT_FILE "data/input.txt"
#endif
#ifndef OUTPUT_FILE
#define OUTPUT_FILE "data/output.txt"
#endif

#ifndef NUM_ITER
#define NUM_ITER 1
#endif

#define INPUT_SAMPLES WINDOW_VSIZE* NUM_ITER
#define OUTPUT_SAMPLES WINDOW_VSIZE* NUM_ITER

// END OF UUT CONFIGURATION
//------------------------------------------------------------------------------
