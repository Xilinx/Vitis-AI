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

#ifndef _XF_HOUGHLINES_CONFIG_H_
#define _XF_HOUGHLINES_CONFIG_H_

#include "hls_stream.h"
#include "ap_int.h"
#include "common/xf_common.hpp"
#include "common/xf_utility.hpp"
#include "imgproc/xf_houghlines.hpp"
//#include "xf_config_params.h"

#define HEIGHT 128
#define WIDTH 128

// Set the optimization type:
#define NPC1 XF_NPPC1

#define RHOSTEP 1

#define THETASTEP 2 // 6.1 format

#define LINESMAX 512

#define DIAGVAL 2203 // 275 //cvRound((sqrt(WIDTH*WIDTH + HEIGHT*HEIGHT)) / RHOSTEP);

#define MINTHETA 0

#define MAXTHETA 180

// Set the pixel depth:
#define TYPE XF_8UC1
#define PTR_WIDTH 8

void houghlines_accel(
    ap_uint<PTR_WIDTH>* img_in, short threshold, short maxlines, float* arrayy, float* arrayx, int rows, int cols);

#endif // _XF_HOUGHLINES_CONFIG_H_
