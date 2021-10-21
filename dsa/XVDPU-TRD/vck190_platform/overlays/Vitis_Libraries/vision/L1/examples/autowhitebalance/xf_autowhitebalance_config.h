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

#ifndef _XF_AWB_CONFIG_H_
#define _XF_AWB_CONFIG_H_

#include "common/xf_common.hpp"
#include "hls_stream.h"
#include "imgproc/xf_autowhitebalance.hpp"
#include "imgproc/xf_duplicateimage.hpp"
#include "xf_config_params.h"
#include <ap_int.h>

/* Input image Dimensions */
#define WIDTH 1024 // Maximum Input image width
#define HEIGHT 676 // Maximum Input image height

#define NPC1 NPPC

// Resolve input and output pixel type:
#if T_8U
#define IN_TYPE XF_8UC3
#define OUT_TYPE XF_8UC3
#endif
#if T_16U
#define IN_TYPE XF_16UC3
#define OUT_TYPE XF_16UC3
#endif

#if T_8U
#define HIST_SIZE 256
#endif
#if T_10U
#define HIST_SIZE 1024
#endif
#if T_16U || T_12U
#define HIST_SIZE 4096
#endif

#define INPUT_PTR_WIDTH 64
#define OUTPUT_PTR_WIDTH 64

void autowhitebalance_accel(ap_uint<INPUT_PTR_WIDTH>* img_inp,
                            ap_uint<OUTPUT_PTR_WIDTH>* img_out,
                            float thresh,
                            int rows,
                            int cols,
                            float inputMin,
                            float inputMax,
                            float outputMin,
                            float outputMax);
#endif //_XF_AWB_CONFIG_H_
