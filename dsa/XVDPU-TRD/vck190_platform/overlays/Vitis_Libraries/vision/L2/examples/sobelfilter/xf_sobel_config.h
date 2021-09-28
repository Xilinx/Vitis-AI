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

#ifndef _XF_SOBEL_CONFIG_H_
#define _XF_SOBEL_CONFIG_H_

#include "hls_stream.h"
#include "ap_int.h"
#include "common/xf_common.hpp"
#include "common/xf_utility.hpp"
#include "imgproc/xf_sobel.hpp"
#include "xf_config_params.h"

typedef unsigned int uint32_t;

//////////////  To set the parameters in Top and Test bench //////////////////

/* config width and height */
#define WIDTH 3840
#define HEIGHT 2160

//#define DDEPTH XF_8UC1

#if FILTER_SIZE_3
#define FILTER_WIDTH 3
#elif FILTER_SIZE_5
#define FILTER_WIDTH 5
#elif FILTER_SIZE_7
#define FILTER_WIDTH 7
#endif

#if GRAY
#if (FILTER_WIDTH == 3 | FILTER_WIDTH == 5)
#if RO
#define IN_TYPE XF_8UC1
#define OUT_TYPE XF_8UC1
#define NPC1 XF_NPPC8
#endif
#if NO
#define IN_TYPE XF_8UC1
#define OUT_TYPE XF_8UC1
#define NPC1 XF_NPPC1
#endif
#endif

#if (FILTER_WIDTH == 7)
#if NO
#define IN_TYPE XF_8UC1
#define OUT_TYPE XF_8UC1
#define NPC1 XF_NPPC1
#endif
#endif

#define CH_TYPE XF_GRAY

#else

#if (FILTER_WIDTH == 3 | FILTER_WIDTH == 5)
#if RO
#define IN_TYPE XF_8UC3
#define OUT_TYPE XF_8UC3
#define NPC1 XF_NPPC8
#endif
#if NO
#define IN_TYPE XF_8UC3
#define OUT_TYPE XF_8UC3
#define NPC1 XF_NPPC1
#endif
#endif

#if (FILTER_WIDTH == 7)
#if NO
#define IN_TYPE XF_8UC3
#define OUT_TYPE XF_8UC3
#define NPC1 XF_NPPC1
#endif
#endif
#define CH_TYPE XF_RGB
#endif

void sobel_accel(xf::cv::Mat<IN_TYPE, HEIGHT, WIDTH, NPC1>& _src,
                 xf::cv::Mat<OUT_TYPE, HEIGHT, WIDTH, NPC1>& _dstgx,
                 xf::cv::Mat<OUT_TYPE, HEIGHT, WIDTH, NPC1>& _dstgy);

#endif //  _XF_SOBEL_CONFIG_H_
