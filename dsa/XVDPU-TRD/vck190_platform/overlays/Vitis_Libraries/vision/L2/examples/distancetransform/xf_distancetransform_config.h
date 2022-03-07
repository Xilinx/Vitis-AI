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

#ifndef __XF_VITIS_DISTANCETRANSFORM_CONFIG_H__
#define __XF_VITIS_DISTANCETRANSFORM_CONFIG_H__

#include "ap_int.h"
#include "common/xf_common.hpp"
#include "common/xf_utility.hpp"
#include "hls_stream.h"
#include "imgproc/xf_distancetransform.hpp"
#include "xf_config_params.h"

/* config width and height */
constexpr int WIDTH = 3840;
constexpr int HEIGHT = 2160;

constexpr int TYPE_IN = XF_8UC1;
constexpr int TYPE_OUT = XF_32FC1;
constexpr int NPC_T = XF_NPPC1;

#endif // __XF_VITIS_DISTANCETRANSFORM_CONFIG_H__
