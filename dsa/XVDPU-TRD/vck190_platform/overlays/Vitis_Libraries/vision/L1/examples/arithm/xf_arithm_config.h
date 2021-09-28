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

#ifndef _XF_ARITHM_CONFIG_H_
#define _XF_ARITHM_CONFIG_H_

#include "common/xf_common.hpp"
#include "common/xf_utility.hpp"
#include "hls_stream.h"
#include "xf_config_params.h"
#include <ap_int.h>

#include "core/xf_arithm.hpp"

#define HEIGHT 128
#define WIDTH 128

// Resolve function name:
/*#if FUNCT_NUM == 0
#define FUNCT_NAME add
#elif FUNCT_NUM == 1
#define FUNCT_NAME addS
#elif FUNCT_NUM == 2
#define FUNCT_NAME subtract
#elif FUNCT_NUM == 3
#define FUNCT_NAME SubS
#elif FUNCT_NUM == 4
#define FUNCT_NAME SubRS
#define FUNCT_SUBRS
#elif FUNCT_NUM == 5
#define FUNCT_NAME multiply
#define FUNCT_MULTIPLY
#elif FUNCT_NUM == 6
#define FUNCT_NAME absdiff
#elif FUNCT_NUM == 7
#define FUNCT_NAME bitwise_and
#elif FUNCT_NUM == 8
#define FUNCT_NAME bitwise_xor
#elif FUNCT_NUM == 9
#define FUNCT_NAME bitwise_not
#define FUNCT_BITWISENOT
#elif FUNCT_NUM == 10
#define FUNCT_NAME bitwise_or
#elif FUNCT_NUM == 11
#define FUNCT_NAME min
#elif FUNCT_NUM == 12
#define FUNCT_NAME max
#elif FUNCT_NUM == 13
#define FUNCT_NAME set
#elif FUNCT_NUM == 14
#define FUNCT_NAME zero
#define FUNCT_ZERO
#elif FUNCT_NUM == 15
#define FUNCT_NAME compare
#else
#define FUNCT_NAME add
#endif*/

// Resolve pixel precision:

#if NO
#define NPC1 XF_NPPC1
#endif
#if RO
#define NPC1 XF_NPPC8
#endif

#if T_16S
#if GRAY
#define TYPE XF_16SC1
#if NO
#define PTR_WIDTH 16
#else
#define PTR_WIDTH 128
#endif
#else
#define TYPE XF_16SC3
#if NO
#define PTR_WIDTH 64
#else
#define PTR_WIDTH 512
#endif
#endif
#endif

#if T_8U
#if GRAY
#define TYPE XF_8UC1
#if NO
#define PTR_WIDTH 8
#else
#define PTR_WIDTH 64
#endif
#else
#define TYPE XF_8UC3
#if NO
#define PTR_WIDTH 32
#else
#define PTR_WIDTH 256
#endif
#endif
#endif

#if ARRAY
#if defined(FUNCT_BITWISENOT) || defined(FUNCT_ZERO)
void arithm_accel(
    ap_uint<PTR_WIDTH>* img_in1, ap_uint<PTR_WIDTH>* img_in2, ap_uint<PTR_WIDTH>* img_out, int height, int width);
#else
void arithm_accel(ap_uint<PTR_WIDTH>* img_in1,
                  ap_uint<PTR_WIDTH>* img_in2,
#ifdef FUNCT_MULTIPLY
                  float scale,
#endif
                  ap_uint<PTR_WIDTH>* img_out,
                  int height,
                  int width);
#endif
#endif
#if SCALAR
void arithm_accel(
    ap_uint<PTR_WIDTH>* img_in1, unsigned char* scl_in, ap_uint<PTR_WIDTH>* img_out, int height, int width);
#endif

#endif // end of _XF_ARITHM_CONFIG_H_
