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

#ifndef _XF_CANNY_CONFIG_H__
#define _XF_CANNY_CONFIG_H__

#include "ap_int.h"
#include "common/xf_common.hpp"
#include "common/xf_utility.hpp"
#include "hls_stream.h"
#include "imgproc/xf_canny.hpp"
#include "imgproc/xf_edge_tracing.hpp"
#include "xf_config_params.h"

#define WIDTH 1920
#define HEIGHT 1080

#if NO
#define INTYPE XF_NPPC1
#define OUTTYPE XF_NPPC32
#elif RO
#define INTYPE XF_NPPC8
#define OUTTYPE XF_NPPC32
#endif

#if L1NORM
#define NORM_TYPE XF_L1NORM
#elif L2NORM
#define NORM_TYPE XF_L2NORM
#endif

void canny_accel(xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, INTYPE>& _src,
                 xf::cv::Mat<XF_2UC1, HEIGHT, WIDTH, XF_NPPC32>& _dst1,
                 xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, XF_NPPC8>& _dst2,
                 unsigned char low_threshold,
                 unsigned char high_threshold);

#endif
