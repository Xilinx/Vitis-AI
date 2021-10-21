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

#ifndef _XF_REMAP_CONFIG_H_
#define _XF_REMAP_CONFIG_H_

#include "hls_stream.h"
#include "common/xf_common.hpp"
#include "common/xf_utility.hpp"
#include "imgproc/xf_remap.hpp"
#include "xf_config_params.h"

// Resolve interpolation type:
#if INTERPOLATION == 0
#define XF_INTERPOLATION_TYPE XF_INTERPOLATION_NN
#else
#define XF_INTERPOLATION_TYPE XF_INTERPOLATION_BILINEAR
#endif

// Set the image type and maps pixel depth:
#if GRAY
#define PTR_IMG_WIDTH 8
#else
#define PTR_IMG_WIDTH 32
#endif
#define TYPE_XY XF_32FC1
#define PTR_MAP_WIDTH 32

#if GRAY
#define TYPE XF_8UC1
#define CHANNELS 1
#else // RGB
#define TYPE XF_8UC3
#define CHANNELS 3
#endif

// Set the optimization type:
// Only XF_NPPC1 is available for this algorithm currently
#define NPC XF_NPPC1

#define HEIGHT 1080
#define WIDTH 1920

#endif
