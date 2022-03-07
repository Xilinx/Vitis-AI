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

#ifndef _XF_DILATION_CONFIG_H_
#define _XF_DILATION_CONFIG_H_

#include "hls_stream.h"
#include "ap_int.h"
#include "common/xf_common.hpp"
#include "common/xf_utility.hpp"
#include "imgproc/xf_dilation.hpp"
#include "xf_config_params.h"

/* config width and height */
#define WIDTH 3840
#define HEIGHT 2160

/*  define the input and output types  */
#if NO
#define NPC_T XF_NPPC1
#else // RO
#define NPC_T XF_NPPC8
#endif

#if GRAY
#define TYPE XF_8UC1
#define CH_TYPE XF_GRAY
#else // RGB
#define TYPE XF_8UC3
#define CH_TYPE XF_RGB
#endif

#endif // _XF_DILATION_CONFIG_H_
