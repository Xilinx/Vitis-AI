/*
 * Copyright 2020 Xilinx, Inc.
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

#ifndef _XF_MODE_FILTER_CONFIG_H_
#define _XF_MODE_FILTER_CONFIG_H_

#include "ap_int.h"
#include "common/xf_common.hpp"
#include "common/xf_utility.hpp"
#include "hls_stream.h"
#include "xf_config_params.h"

#include "imgproc/xf_modefilter.hpp"

#define HEIGHT 2160
#define WIDTH 3840

// Set the optimization type:
#if NO == 1
#define NPC1 XF_NPPC1
#define PTR_WIDTH 128
#else

#if GRAY
#define NPC1 XF_NPPC8
#else
#define NPC1 XF_NPPC1
#endif

#define PTR_WIDTH 128
#endif

// Set the pixel depth:
#if GRAY
#define TYPE XF_8UC1
#else
#define TYPE XF_8UC3
#endif

#endif
