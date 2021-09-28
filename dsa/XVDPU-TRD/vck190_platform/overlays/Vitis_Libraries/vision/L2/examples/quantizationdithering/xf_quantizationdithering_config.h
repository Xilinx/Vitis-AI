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

#ifndef _XF_QUANTIZATIONDITHERING_CONFIG_
#define _XF_QUANTIZATIONDITHERING_CONFIG_

#include "ap_int.h"
#include "common/xf_common.hpp"
#include "hls_stream.h"
#include "imgproc/xf_quantizationdithering.hpp"
#include "xf_config_params.h"

/* Interface types*/
#if RO
#define NPC_T XF_NPPC2
#else
#define NPC_T XF_NPPC1
#endif

#if RGB

#if INPUTPIXELDEPTH == 8
#define TYPEIN XF_8UC3
#elif INPUTPIXELDEPTH == 16
#define TYPEIN XF_16UC3
#endif

#if OUTPUTPIXELDEPTH == 8
#define TYPEOUT XF_8UC3
#elif OUTPUTPIXELDEPTH == 16
#define TYPEOUT XF_16UC3
#endif

#define CH_TYPE XF_RGB
#else
#if INPUTPIXELDEPTH == 8
#define TYPEIN XF_8UC1
#elif INPUTPIXELDEPTH == 16
#define TYPEIN XF_16UC1
#endif

#if OUTPUTPIXELDEPTH == 8
#define TYPEOUT XF_8UC1
#elif OUTPUTPIXELDEPTH == 16
#define TYPEOUT XF_16UC1
#endif

#define CH_TYPE XF_GRAY
#endif

#endif
