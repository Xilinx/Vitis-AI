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

#define HEIGHT 1080
#define WIDTH 1920

// Set the output image size:
#if DIM
#define ONE_D_HEIGHT 1080
#define ONE_D_WIDTH 1
#else
#define ONE_D_HEIGHT 1
#define ONE_D_WIDTH 1920
#endif

// Set the optimization type:
#if NO
#define NPC1 XF_NPPC1
#endif

// Set the input pixel depth
#define IN_TYPE XF_8UC1
#if NO
#define PTR_IN_WIDTH 8
#endif

// Resolve reduction type
#if REDUCTION_OP == 1
#define XF_REDUCE XF_REDUCE_AVG
#define CV_REDUCE XF_REDUCE_AVG
#define OUT_TYPE XF_32SC1
#if NO
#define PTR_OUT_WIDTH 32
#endif
#elif REDUCTION_OP == 0
#define XF_REDUCE XF_REDUCE_SUM
#define CV_REDUCE XF_REDUCE_SUM
#define OUT_TYPE XF_32SC1
#if NO
#define PTR_OUT_WIDTH 32
#endif
#elif REDUCTION_OP == 2
#define XF_REDUCE XF_REDUCE_MAX
#define CV_REDUCE XF_REDUCE_MAX
#define OUT_TYPE XF_8UC1
#if NO
#define PTR_OUT_WIDTH 8
#endif
#else
#define XF_REDUCE XF_REDUCE_MIN
#define CV_REDUCE XF_REDUCE_MIN
#define OUT_TYPE XF_8UC1
#if NO
#define PTR_OUT_WIDTH 8
#endif
#endif

#endif // end of _XF_REDUCE_CONFIG_H_
