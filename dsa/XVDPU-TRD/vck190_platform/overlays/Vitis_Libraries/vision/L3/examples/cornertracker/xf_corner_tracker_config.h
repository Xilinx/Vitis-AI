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

#ifndef __XF_CORNER_TRACKER_CONFIG__
#define __XF_CORNER_TRACKER_CONFIG__

#include "ap_int.h"
#include "hls_stream.h"
#include "assert.h"
#include "common/xf_common.hpp"
#include "xf_config_params.h"
#include "video/xf_pyr_dense_optical_flow_wrapper.hpp"
#include "imgproc/xf_pyr_down.hpp"
#include "features/xf_harris.hpp"
#include "imgproc/xf_corner_update.hpp"
#include "imgproc/xf_corner_img_to_list.hpp"

#define CH_TYPE XF_GRAY

/*void cornerTracker(xf::cv::Mat<XF_32UC1,HEIGHT,WIDTH,XF_NPPC1> & flow, xf::cv::Mat<XF_32UC1,HEIGHT,WIDTH,XF_NPPC1> &
 * flow_iter, xf::cv::Mat<XF_8UC1,HEIGHT,WIDTH,XF_NPPC1> mat_imagepyr1[NUM_LEVELS] ,
 * xf::cv::Mat<XF_8UC1,HEIGHT,WIDTH,XF_NPPC1> mat_imagepyr2[NUM_LEVELS] , xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, XF_NPPC1>
 * &inHarris, xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, XF_NPPC1> &outHarris, unsigned int *list, unsigned long *listfixed,
 * int pyr_h[NUM_LEVELS], int pyr_w[NUM_LEVELS], unsigned int *params);*/
#endif
