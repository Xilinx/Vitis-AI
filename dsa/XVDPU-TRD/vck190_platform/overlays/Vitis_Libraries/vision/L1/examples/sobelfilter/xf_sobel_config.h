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

#ifndef _XF_SOBEL_CONFIG_H_
#define _XF_SOBEL_CONFIG_H_

#include "hls_stream.h"
#include "ap_int.h"
#include "common/xf_common.hpp"
#include "common/xf_utility.hpp"
#include "imgproc/xf_sobel.hpp"
#include "xf_config_params.h"

typedef unsigned int uint32_t;

//////////////  To set the parameters in Top and Test bench //////////////////

/* config width and height */
#define WIDTH 128
#define HEIGHT 128

//#define DDEPTH XF_8UC1

#if FILTER_SIZE_3
#define FILTER_WIDTH 3
#elif FILTER_SIZE_5
#define FILTER_WIDTH 5
#elif FILTER_SIZE_7
#define FILTER_WIDTH 7
#endif
#if GRAY
#if (FILTER_WIDTH == 3 | FILTER_WIDTH == 5)
#if RO
#define IN_TYPE XF_8UC1
#if T_8U
#define OUT_TYPE XF_8UC1
#else
#define OUT_TYPE XF_16SC1
#endif
#define NPC1 XF_NPPC8
#endif
#if NO
#define IN_TYPE XF_8UC1
#if T_8U
#define OUT_TYPE XF_8UC1
#else
#define OUT_TYPE XF_16SC1
#endif
#define NPC1 XF_NPPC1
#endif
#endif

#if (FILTER_WIDTH == 7)
#if NO
#define IN_TYPE XF_8UC1
#if T_8U
#define OUT_TYPE XF_8UC1
#else
#define OUT_TYPE XF_16SC1
#endif
#define NPC1 XF_NPPC1
#endif
#endif

#else

#if (FILTER_WIDTH == 3 | FILTER_WIDTH == 5)
#if RO
#define IN_TYPE XF_8UC3
#if T_8U
#define OUT_TYPE XF_8UC3
#else
#define OUT_TYPE XF_16SC3
#endif
#define NPC1 XF_NPPC8
#endif
#if NO
#define IN_TYPE XF_8UC3
#if T_8U
#define OUT_TYPE XF_8UC3
#else
#define OUT_TYPE XF_16SC3
#endif
#define NPC1 XF_NPPC1
#endif
#endif

#if (FILTER_WIDTH == 7)
#if NO
#define IN_TYPE XF_8UC3
#if T_8U
#define OUT_TYPE XF_8UC3
#else
#define OUT_TYPE XF_16SC3
#endif
#define NPC1 XF_NPPC1
#endif
#endif
#endif

#define INPUT_PTR_WIDTH 128
#define OUTPUT_PTR_WIDTH 128

void sobel_accel(ap_uint<INPUT_PTR_WIDTH>* img_inp,
                 ap_uint<OUTPUT_PTR_WIDTH>* img_out1,
                 ap_uint<OUTPUT_PTR_WIDTH>* img_out2,
                 int rows,
                 int cols);
#endif //  _XF_SOBEL_CONFIG_H_
