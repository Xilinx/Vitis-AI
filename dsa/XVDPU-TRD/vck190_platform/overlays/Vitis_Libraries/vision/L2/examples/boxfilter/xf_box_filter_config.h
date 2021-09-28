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

#ifndef _XF_BOX_FILTER_CONFIG_H_
#define _XF_BOX_FILTER_CONFIG_H_

#include "hls_stream.h"
#include "ap_int.h"
#include "common/xf_common.hpp"
#include "common/xf_utility.hpp"
#include "imgproc/xf_box_filter.hpp"
#include "xf_config_params.h"

/* set the height and width */
#define HEIGHT 2160
#define WIDTH 3840

#if RO
#define NPIX XF_NPPC8
#endif
#if NO
#define NPIX XF_NPPC1
#endif

#if T_8U
#define IN_T XF_8UC1
#define IN_TYPE unsigned char
#endif
#if T_16U
#define IN_T XF_16UC1
#define IN_TYPE unsigned short int
#endif
#if T_16S
#define IN_T XF_16SC1
#define IN_TYPE short int
#endif

void box_filter_accel(xf::cv::Mat<IN_T, HEIGHT, WIDTH, NPIX>& _src, xf::cv::Mat<IN_T, HEIGHT, WIDTH, NPIX>& _dst);

#endif // end of _XF_BOX_FILTER_CONFIG_H_
