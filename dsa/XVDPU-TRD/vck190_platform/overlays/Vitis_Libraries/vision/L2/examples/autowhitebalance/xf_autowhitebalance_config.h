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

#define NPC1 NPPC

// Set the image height and width
#define HEIGHT 1080 // 2160
#define WIDTH 1920  // 3840

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

// void autowhitebalance_accel(xf::cv::Mat<IN_TYPE, HEIGHT, WIDTH, NPC1>
// &imgInput1,  xf::cv::Mat<OUT_TYPE, HEIGHT,
// WIDTH, NPC1> &imgOutput, float thresh);
#endif //_XF_AWB_CONFIG_H_
