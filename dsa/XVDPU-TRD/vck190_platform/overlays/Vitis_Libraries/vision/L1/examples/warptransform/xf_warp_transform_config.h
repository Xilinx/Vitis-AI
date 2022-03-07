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

#ifndef __XF_TRANSFORM_CONFIG__
#define __XF_TRANSFORM_CONFIG__
#include <ap_int.h>
#include <cmath>
#include <iostream>
#include <math.h>
#include <iostream>
#include "hls_stream.h"
#include "common/xf_common.hpp"
#include "common/xf_utility.hpp"
#include "xf_config_params.h"
#include "imgproc/xf_warp_transform.hpp"

// Set the pixel depth:
#if RGBA
#define TYPE XF_8UC3
#else
#define TYPE XF_8UC1
#endif

#define PTR_WIDTH 32

// Set the optimization type:
#define NPC1 XF_NPPC1

void warp_transform_accel(
    ap_uint<PTR_WIDTH>* img_in, float* transform, ap_uint<PTR_WIDTH>* img_out, int rows, int cols);

#endif
