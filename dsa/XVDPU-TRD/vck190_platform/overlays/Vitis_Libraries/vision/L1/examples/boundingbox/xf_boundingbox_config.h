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

#ifndef _XF_BOUNDINGBOX_CONFIG_H_
#define _XF_BOUNDINGBOX_CONFIG_H_

#include "hls_stream.h"
#include "ap_int.h"

#include "common/xf_common.hpp"
#include "common/xf_utility.hpp"
#include "imgproc/xf_boundingbox.hpp"
#include "xf_config_params.h"

typedef ap_uint<8> ap_uint8_t;
typedef ap_uint<64> ap_uint64_t;

#if NO
#define NPIX XF_NPPC1
#endif

#if GRAY
#define TYPE XF_8UC1
#define INPUT_CH_TYPE XF_GRAY
#define OUTPUT_CH_TYPE XF_GRAY
#else
#define TYPE XF_8UC4
#define INPUT_CH_TYPE 4
#define OUTPUT_CH_TYPE 4
#endif

#define INPUT_PTR_WIDTH 32
#define OUTPUT_PTR_WIDTH 32

void boundingbox_accel(
    ap_uint<INPUT_PTR_WIDTH>* in_img, int* roi, int color_info[MAX_BOXES][4], int height, int width, int num_box);

#endif // end of _XF_BOUNDINGBOX_CONFIG_H_
