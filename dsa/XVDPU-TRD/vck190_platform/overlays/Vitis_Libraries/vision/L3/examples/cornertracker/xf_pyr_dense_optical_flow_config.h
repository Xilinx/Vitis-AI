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

#ifndef __XF_PYR_DENSE_OPTICAL_FLOW_CONFIG__
#define __XF_PYR_DENSE_OPTICAL_FLOW_CONFIG__

#include "ap_int.h"
#include "hls_stream.h"
#include "assert.h"
#include "common/xf_common.hpp"
#include "common/xf_utility.hpp"

#include "xf_config_params.h"
#include "video/xf_pyr_dense_optical_flow_wrapper.hpp"
#include "imgproc/xf_pyr_down.hpp"

#define IN_TYPE unsigned char
#define OUT_TYPE unsigned char

#define INPUT_PTR_WIDTH 32
#define OUTPUT_PTR_WIDTH 32
#define CH_TYPE XF_GRAY
#define TYPE XF_8UC1
#define NPC_T XF_NPPC1
#define XF_USE_URAM false

#endif
