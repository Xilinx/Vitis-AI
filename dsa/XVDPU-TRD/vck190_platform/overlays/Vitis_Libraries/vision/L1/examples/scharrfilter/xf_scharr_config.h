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

#ifndef _XF_SCHARR_CONFIG_H_
#define _XF_SCHARR_CONFIG_H_

#include "hls_stream.h"
#include "ap_int.h"
#include "common/xf_common.hpp"
#include "common/xf_utility.hpp"
#include "imgproc/xf_scharr.hpp"
#include "xf_config_params.h"

/* config width and height */
#define WIDTH 128
#define HEIGHT 128

typedef unsigned uint32_t;

/*  define the input and output types  */
#if NO
#define NPC1 XF_NPPC1
#endif

#if RO
#define NPC1 XF_NPPC8
#endif

#if T_8U

#if GRAY
#define IN_TYPE XF_8UC1
#define OUT_TYPE XF_8UC1
#else
#define IN_TYPE XF_8UC3
#define OUT_TYPE XF_8UC3
#endif

#else

#if GRAY
#define IN_TYPE XF_8UC1
#define OUT_TYPE XF_16SC1
#else
#define IN_TYPE XF_8UC3
#define OUT_TYPE XF_16SC3
#endif

#endif

#define INPUT_PTR_WIDTH 128
#define OUTPUT_PTR_WIDTH 128

void scharr_accel(ap_uint<INPUT_PTR_WIDTH>* img_inp,
                  ap_uint<OUTPUT_PTR_WIDTH>* img_out1,
                  ap_uint<OUTPUT_PTR_WIDTH>* img_out2,
                  int rows,
                  int cols);

#endif // _XF_SCHARR_CONFIG_H_
