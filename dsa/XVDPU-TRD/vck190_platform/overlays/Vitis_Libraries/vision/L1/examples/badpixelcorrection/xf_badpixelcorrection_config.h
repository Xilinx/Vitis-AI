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

#ifndef _XF_BPC_CONFIG_H_
#define _XF_BPC_CONFIG_H_
#include "common/xf_common.hpp"
#include "hls_stream.h"
#include "imgproc/xf_bpc.hpp"
#include "xf_config_params.h"
#include <ap_int.h>

// Set the image height and width
#define HEIGHT 128 // 2160
#define WIDTH 128  // 3840

#define NPC1 NPPC

// Resolve input and output pixel type:
#if T_8U
#define IN_TYPE XF_8UC1
#define OUT_TYPE XF_8UC1
#endif
#if T_16U
#define IN_TYPE XF_16UC1
#define OUT_TYPE XF_16UC1
#endif

void badpixelcorrection_accel(ap_uint<INPUT_PTR_WIDTH>* img_inp,
                              ap_uint<OUTPUT_PTR_WIDTH>* img_out,
                              int rows,
                              int cols);
#endif //_XF_BPC_CONFIG_H_
