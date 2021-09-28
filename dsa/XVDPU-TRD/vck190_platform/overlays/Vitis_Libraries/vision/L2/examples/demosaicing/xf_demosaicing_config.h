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

#ifndef _XF_DEMOSIACING_CONFIG_H_
#define _XF_DEMOSIACING_CONFIG_H_

#include "hls_stream.h"
#include "ap_int.h"
#include "common/xf_common.hpp"
#include "common/xf_utility.hpp"
#include "imgproc/xf_demosaicing.hpp"
#include "xf_config_params.h"

#define PTR_IN_WIDTH 128
#define PTR_OUT_WIDTH 128

// Resolve input and output pixel type:
#if T_8U
#define IN_TYPE XF_8UC1
#define OUT_TYPE XF_8UC3
#endif
#if T_16U
#define IN_TYPE XF_16UC1
#define OUT_TYPE XF_16UC3
#endif

// Resolve optimization type:
#define NPC1 NPPC

#if (T_16U)
#define CV_INTYPE CV_16UC1
#define CV_OUTTYPE CV_16UC3
#else
#define CV_INTYPE CV_8UC1
#define CV_OUTTYPE CV_8UC3
#endif

#define ERROR_THRESHOLD 1

#endif // _XF_DEMOSAICING_CONFIG_H_
