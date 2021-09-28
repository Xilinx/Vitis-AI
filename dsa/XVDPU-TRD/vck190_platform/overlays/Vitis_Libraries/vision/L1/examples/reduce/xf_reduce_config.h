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

#ifndef _XF_REDUCE_CONFIG_H_
#define _XF_REDUCE_CONFIG_H_

#include "hls_stream.h"
#include "ap_int.h"

#include "common/xf_common.hpp"
#include "common/xf_utility.hpp"
#include "imgproc/xf_reduce.hpp"
#include "xf_config_params.h"

#define HEIGHT 128
#define WIDTH 128

// Set the output image size:
#if DIM
#define ONE_D_HEIGHT 128
#define ONE_D_WIDTH 1
#else
#define ONE_D_HEIGHT 1
#define ONE_D_WIDTH 128
#endif

// Set the optimization type:
#if NO
#define NPC1 XF_NPPC1
#endif

// Set the input pixel depth
#define IN_TYPE XF_8UC1
#if NO
#define INPUT_PTR_WIDTH 8
#else
#define INPUT_PTR_WIDTH 64
#endif

#if (REDUCTION_OP == 1)
#define DST_T XF_32SC1
#define OUTPUT_PTR_WIDTH 32
#elif (REDUCTION_OP == 0)
#define DST_T XF_32SC1
#define OUTPUT_PTR_WIDTH 32
#else
#define DST_T XF_8UC1
#define OUTPUT_PTR_WIDTH 8
#endif
void reduce_accel(ap_uint<INPUT_PTR_WIDTH>* img_in,
                  unsigned char dimension,
                  ap_uint<OUTPUT_PTR_WIDTH>* img_out,
                  int height,
                  int width);
#endif // end of _XF_REDUCE_CONFIG_H_
