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

#ifndef _XF_BILATERAL_FILTER_CONFIG_H_
#define _XF_BILATERAL_FILTER_CONFIG_H_

#include "hls_stream.h"
#include "ap_int.h"

#include "common/xf_common.hpp"
#include "common/xf_utility.hpp"
#include "xf_config_params.h"
#include "imgproc/xf_bilateral_filter.hpp"

typedef unsigned short int uint16_t;

#define ERROR_THRESHOLD 0 // acceptable error threshold range 0 to 255

// Resolve optimization type:
#if RO

#if GRAY
#define NPC1 XF_NPPC8
#else
#define NPC1 XF_NPPC4
#endif

#define PTR_WIDTH 128
#else
#define NPC1 XF_NPPC1
#define PTR_WIDTH 128
#endif

#if FILTER_SIZE_3
#define FILTER_WIDTH 3
#elif FILTER_SIZE_5
#define FILTER_WIDTH 5
#elif FILTER_SIZE_7
#define FILTER_WIDTH 7
#endif

#if GRAY
#define TYPE XF_8UC1
#else
#define TYPE XF_8UC3
#endif

void bilateral_filter_accel(
    ap_uint<PTR_WIDTH>* img_in, float sigma_color, float sigma_space, int rows, int cols, ap_uint<PTR_WIDTH>* img_out);

#endif //_XF_BILATERAL_FILTER_CONFIG_H_
