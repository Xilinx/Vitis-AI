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

#ifndef _XF_THRESHOLD_CONFIG_H_
#define _XF_THRESHOLD_CONFIG_H_

#include "hls_stream.h"
#include "ap_int.h"
#include "common/xf_common.hpp"
#include "common/xf_utility.hpp"
#include "imgproc/xf_inrange.hpp"
#include "xf_config_params.h"

#define HEIGHT 128
#define WIDTH 128

// Resolve optimization type:
#if RO
#define NPC1 XF_NPPC8
#endif
#if NO
#define NPC1 XF_NPPC1
#endif

// Set the pixel type:
#if RGB
#define IN_TYPE XF_8UC3
#define OUT_TYPE XF_8UC1
#if NO
#define INPUT_PTR_WIDTH 32
#define OUTPUT_PTR_WIDTH 8
#else
#define INPUT_PTR_WIDTH 256
#define OUTPUT_PTR_WIDTH 64
#endif
#else
#define IN_TYPE XF_8UC1
#define OUT_TYPE XF_8UC1
#if NO
#define INPUT_PTR_WIDTH 8
#define OUTPUT_PTR_WIDTH 8
#else
#define INPUT_PTR_WIDTH 64
#define OUTPUT_PTR_WIDTH 64
#endif
#endif
void inrange_accel(ap_uint<INPUT_PTR_WIDTH>* img_in,
                   unsigned char lower_thresh,
                   unsigned char upper_thresh,
                   ap_uint<OUTPUT_PTR_WIDTH>* img_out,
                   int height,
                   int width);

#endif // end of _XF_THRESHOLD_CONFIG_H_
