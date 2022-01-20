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

#ifndef _XF_GTM_CONFIG_H_
#define _XF_GTM_CONFIG_H_

#include "hls_stream.h"
#include <ap_int.h>
#include "xf_config_params.h"
#include "common/xf_common.hpp"
#include "common/xf_structs.hpp"

#include "imgproc/xf_gtm.hpp"

/* Set the image height and width */
#define HEIGHT 676
#define WIDTH 1024

// Resolve the optimization type:
#if NO
#define NPC1 XF_NPPC1
#define INPUT_PTR_WIDTH 64
#define OUTPUT_PTR_WIDTH 32
#endif
#if RO
#define NPC1 XF_NPPC2
#define INPUT_PTR_WIDTH 128
#define OUTPUT_PTR_WIDTH 64
#endif

// Set the input and output pixel depth:
#if T_16U
#define IN_TYPE XF_16UC3
#define SIN_CHANNEL_IN_TYPE XF_16UC1
#endif

#define OUT_TYPE XF_8UC3
#define SIN_CHANNEL_OUT_TYPE XF_8UC1
#endif //_XF_GTM_CONFIG_H_
