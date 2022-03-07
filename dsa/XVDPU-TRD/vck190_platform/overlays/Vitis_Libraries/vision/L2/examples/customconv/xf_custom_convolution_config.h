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

#ifndef _XF_CUSTOM_CONV_CONFIG_H_
#define _XF_CUSTOM_CONV_CONFIG_H_

#include "hls_stream.h"
#include "ap_int.h"
#include "common/xf_common.hpp"
#include "common/xf_utility.hpp"
#include "imgproc/xf_custom_convolution.hpp"
#include "xf_config_params.h"

#define HEIGHT 2160
#define WIDTH 3840

/*  specify the shift parameter */
#define SHIFT 15

// Resolve optimization type:
#if RO
#define NPC1 XF_NPPC8
#endif
#if NO
#define NPC1 XF_NPPC1
#endif

#if GRAY
// Resolve pixel depth:
#if OUT_8U
#define INTYPE XF_8UC1
#define OUTTYPE XF_8UC1
#if NO
#define PTR_IN_WIDTH 8
#define PTR_OUT_WIDTH 8
#else
#define PTR_IN_WIDTH 64
#define PTR_OUT_WIDTH 64
#endif
#endif

#if OUT_16S
#define INTYPE XF_8UC1
#define OUTTYPE XF_16SC1
#if NO
#define PTR_IN_WIDTH 8
#define PTR_OUT_WIDTH 16
#else
#define PTR_IN_WIDTH 64
#define PTR_OUT_WIDTH 128
#endif
#endif

#else
// Resolve pixel depth:
#if OUT_8U
#define INTYPE XF_8UC3
#define OUTTYPE XF_8UC3
#if NO
#define PTR_IN_WIDTH 32
#define PTR_OUT_WIDTH 32
#else
#define PTR_IN_WIDTH 256
#define PTR_OUT_WIDTH 246
#endif
#endif

#if OUT_16S
#define INTYPE XF_8UC3
#define OUTTYPE XF_16SC3
#if NO
#define PTR_IN_WIDTH 64
#define PTR_OUT_WIDTH 64
#else
#define PTR_IN_WIDTH 512
#define PTR_OUT_WIDTH 512
#endif
#endif

#endif
#endif // end of _XF_CUSTOM_CONV_CONFIG_H_
