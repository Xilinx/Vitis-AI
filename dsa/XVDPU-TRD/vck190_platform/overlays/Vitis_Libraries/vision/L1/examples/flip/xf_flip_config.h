/*
 * Copyright 2021 Xilinx, Inc.
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

#ifndef _XF_FLIP_CONFIG_H_
#define _XF_FLIP_CONFIG_H_

#include "hls_stream.h"
#include <ap_int.h>
#include "xf_config_params.h"
#include "common/xf_common.hpp"
#include "common/xf_structs.hpp"

#include "imgproc/xf_flip.hpp"

/* Set the image height and width */
#define HEIGHT 2160
#define WIDTH 3840

// Resolve the optimization type:
#if NO
#define NPC1 XF_NPPC1
#if GRAY
#define PTR_WIDTH 8
#else
#define PTR_WIDTH 32
#endif
#endif
#if RO
#define NPC1 XF_NPPC4
#if GRAY
#define PTR_WIDTH 32
#else
#define PTR_WIDTH 128
#endif
#endif

// Set the input and output pixel depth:
#if GRAY
#define TYPE XF_8UC1
#else
#define TYPE XF_8UC3
#endif

void flip_accel(ap_uint<PTR_WIDTH>* img_in, ap_uint<PTR_WIDTH>* img_out, int height, int width, int direction);

#endif //_XF_FLIP_CONFIG_H_
