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

#ifndef _XF_INTEGRAL_IMAGE_CONFIG_H_
#define _XF_INTEGRAL_IMAGE_CONFIG_H_

#include "hls_stream.h"
#include "ap_int.h"
#include "common/xf_common.hpp"
#include "common/xf_utility.hpp"
#include "imgproc/xf_integral_image.hpp"
#include "xf_config_params.h"

typedef unsigned short uint16_t;

#define NPC1 XF_NPPC1

#define INPUT_PTR_WIDTH 8
#define OUTPUT_PTR_WIDTH 32

void integral_accel(ap_uint<INPUT_PTR_WIDTH>* img_inp, ap_uint<OUTPUT_PTR_WIDTH>* img_out, int rows, int cols);
#endif

// _XF_INTEGRAL_IMAGE_CONFIG_H_
