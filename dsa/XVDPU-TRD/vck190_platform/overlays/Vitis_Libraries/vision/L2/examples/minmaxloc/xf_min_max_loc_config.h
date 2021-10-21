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

#ifndef _XF_MIN_MAX_LOC_CONFIG_H_
#define _XF_MIN_MAX_LOC_CONFIG_H_

#include "hls_stream.h"
#include "ap_int.h"
#include "xf_config_params.h"
#include "common/xf_utility.hpp"
#include "common/xf_common.hpp"
#include "core/xf_min_max_loc.hpp"

#define HEIGHT 128
#define WIDTH 128

// Resolve pixel depth:
#if T_8U
#define TYPE XF_8UC1
#if NO
#define PTR_WIDTH 8
#else
#define PTR_WIDTH 64
#endif
#define INTYPE unsigned char
#endif
#if T_16U
#define TYPE XF_16UC1
#if NO
#define PTR_WIDTH 16
#else
#define PTR_WIDTH 128
#endif
#define INTYPE unsigned short
#endif
#if T_16S
#define TYPE XF_16SC1
#if NO
#define PTR_WIDTH 16
#else
#define PTR_WIDTH 128
#endif
#define INTYPE signed short
#endif
#if T_32S
#define TYPE XF_32SC1
#if NO
#define PTR_WIDTH 32
#else
#define PTR_WIDTH 256
#endif
#define INTYPE signed int
#endif

// Resolve optimization type:
#if NO
#define NPC1 XF_NPPC1
#endif

#if RO
#define NPC1 XF_NPPC8
#endif

#endif
