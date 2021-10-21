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

#ifndef _XF_PHASE_CONFIG_H_
#define _XF_PHASE_CONFIG_H_

#include "hls_stream.h"
#include "ap_int.h"
#include "common/xf_common.hpp"
#include "common/xf_utility.hpp"
#include "core/xf_phase.hpp"
#include "xf_config_params.h"

typedef unsigned short int uint16_t;

/*  set the height and weight  */
#define HEIGHT 2160
#define WIDTH 3840

/*  define the input and output types  */
#if NO
#define NPC1 XF_NPPC1
#endif

#if RO
#define NPC1 XF_NPPC8
#endif

#if RADIANS
#define DEG_TYPE XF_RADIANS
#endif
#if DEGREES
#define DEG_TYPE XF_DEGREES
#endif

void phase_accel(xf::cv::Mat<XF_16SC1, HEIGHT, WIDTH, NPC1>& _src1,
                 xf::cv::Mat<XF_16SC1, HEIGHT, WIDTH, NPC1>& _src2,
                 xf::cv::Mat<XF_16SC1, HEIGHT, WIDTH, NPC1>& _dst);

#endif // end of _XF_PHASE_CONFIG_H_
