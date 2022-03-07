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

#ifndef _XF_CVT_COLOR_CONFIG_GEN_VITIS_H_
#define _XF_CVT_COLOR_CONFIG_GEN_VITIS_H_

#include "hls_stream.h"
#include "ap_int.h"
#include "xf_config_params.h"
#include "common/xf_common.hpp"
#include "common/xf_utility.hpp"
#include "imgproc/xf_cvt_color.hpp"
#include "imgproc/xf_cvt_color_1.hpp"
//#include "imgproc/xf_rgb2hsv.hpp"
//#include "imgproc/xf_bgr2hsv.hpp"
// Has to be set when synthesizing
#define _XF_SYNTHESIS_ 1

// Image Dimensions
static constexpr int WIDTH = 1920;
static constexpr int HEIGHT = 1080;

#if (IYUV2NV12 || NV122IYUV || NV212IYUV || NV122YUV4 || NV212YUV4 || UYVY2NV12 || UYVY2NV21 || YUYV2NV12 ||           \
     YUYV2NV21 || RGBA2NV12 || RGBA2NV21 || RGB2NV12 || RGB2NV21 || NV122RGBA || NV212RGB || NV212RGBA || NV122RGB ||  \
     NV122BGR || NV212BGR || NV122YUYV || NV212YUYV || NV122UYVY || NV212UYVY || NV122NV21 || NV212NV12 || BGR2NV12 || \
     BGR2NV21)
#if NO
static constexpr int NPC1 = XF_NPPC1;
static constexpr int NPC2 = XF_NPPC1;
#endif
#if RO
static constexpr int NPC1 = XF_NPPC8;
static constexpr int NPC2 = XF_NPPC4;
#endif

#else
#if NO
static constexpr int NPC1 = XF_NPPC1;
static constexpr int NPC2 = XF_NPPC1;
#else
static constexpr int NPC1 = XF_NPPC8;
static constexpr int NPC2 = XF_NPPC8;
#endif
#endif

void cvtcolor_rgba2iyuv(ap_uint<32 * NPC1>* imgInput,
                        ap_uint<8 * NPC1>* imgOutput0,
                        ap_uint<8 * NPC1>* imgOutput1,
                        ap_uint<8 * NPC1>* imgOutput2);
void cvtcolor_rgba2nv12(ap_uint<32 * NPC1>* imgInput, ap_uint<8 * NPC1>* imgOutput0, ap_uint<16 * NPC2>* imgOutput1);
void cvtcolor_rgba2nv21(ap_uint<32 * NPC1>* imgInput, ap_uint<8 * NPC1>* imgOutput0, ap_uint<16 * NPC2>* imgOutput1);
void cvtcolor_rgba2yuv4(ap_uint<32 * NPC1>* imgInput,
                        ap_uint<8 * NPC1>* imgOutput0,
                        ap_uint<8 * NPC1>* imgOutput1,
                        ap_uint<8 * NPC1>* imgOutput2);
void cvtcolor_rgb2iyuv(ap_uint<32 * NPC1>* imgInput,
                       ap_uint<8 * NPC1>* imgOutput0,
                       ap_uint<8 * NPC1>* imgOutput1,
                       ap_uint<8 * NPC1>* imgOutput2);
void cvtcolor_rgb2nv12(ap_uint<32 * NPC1>* imgInput, ap_uint<8 * NPC1>* imgOutput0, ap_uint<16 * NPC2>* imgOutput1);
void cvtcolor_rgb2nv21(ap_uint<32 * NPC1>* imgInput, ap_uint<8 * NPC1>* imgOutput0, ap_uint<16 * NPC2>* imgOutput1);
void cvtcolor_rgb2yuv4(ap_uint<32 * NPC1>* imgInput,
                       ap_uint<8 * NPC1>* imgOutput0,
                       ap_uint<8 * NPC1>* imgOutput1,
                       ap_uint<8 * NPC1>* imgOutput2);
void cvtcolor_rgb2uyvy(ap_uint<32 * NPC1>* imgInput, ap_uint<16 * NPC1>* imgOutput);
void cvtcolor_rgb2yuyv(ap_uint<32 * NPC1>* imgInput, ap_uint<16 * NPC1>* imgOutput);
void cvtcolor_rgb2bgr(ap_uint<32 * NPC1>* imgInput, ap_uint<32 * NPC1>* imgOutput);
void cvtcolor_bgr2uyvy(ap_uint<32 * NPC1>* imgInput, ap_uint<16 * NPC1>* imgOutput);
void cvtcolor_bgr2yuyv(ap_uint<32 * NPC1>* imgInput, ap_uint<16 * NPC1>* imgOutput);
void cvtcolor_bgr2rgb(ap_uint<32 * NPC1>* imgInput, ap_uint<32 * NPC1>* imgOutput);
void cvtcolor_bgr2nv12(ap_uint<32 * NPC1>* imgInput, ap_uint<8 * NPC1>* imgOutput0, ap_uint<16 * NPC2>* imgOutput1);
void cvtcolor_bgr2nv21(ap_uint<32 * NPC1>* imgInput, ap_uint<8 * NPC1>* imgOutput0, ap_uint<16 * NPC2>* imgOutput1);
void cvtcolor_iyuv2nv12(ap_uint<8 * NPC1>* imgInput0,
                        ap_uint<8 * NPC1>* imgInput1,
                        ap_uint<8 * NPC1>* imgInput2,
                        ap_uint<8 * NPC1>* imgOutput0,
                        ap_uint<16 * NPC2>* imgOutput1);
void cvtcolor_iyuv2rgba(ap_uint<8 * NPC1>* imgInput0,
                        ap_uint<8 * NPC1>* imgInput1,
                        ap_uint<8 * NPC1>* imgInput2,
                        ap_uint<32 * NPC1>* imgOutput);
void cvtcolor_iyuv2rgb(ap_uint<8 * NPC1>* imgInput0,
                       ap_uint<8 * NPC1>* imgInput1,
                       ap_uint<8 * NPC1>* imgInput2,
                       ap_uint<32 * NPC1>* imgOutput);
void cvtcolor_iyuv2yuv4(ap_uint<8 * NPC1>* imgInput0,
                        ap_uint<8 * NPC1>* imgInput1,
                        ap_uint<8 * NPC1>* imgInput2,
                        ap_uint<8 * NPC1>* imgOutput0,
                        ap_uint<8 * NPC1>* imgOutput1,
                        ap_uint<8 * NPC1>* imgOutput2);
void cvtcolor_nv122iyuv(ap_uint<8 * NPC1>* imgInput0,
                        ap_uint<16 * NPC2>* imgInput1,
                        ap_uint<8 * NPC1>* imgOutput0,
                        ap_uint<8 * NPC1>* imgOutput1,
                        ap_uint<8 * NPC1>* imgOutput2);
void cvtcolor_nv122rgba(ap_uint<8 * NPC1>* imgInput0, ap_uint<16 * NPC2>* imgInput1, ap_uint<32 * NPC1>* imgOutput);
void cvtcolor_nv122yuv4(ap_uint<8 * NPC1>* imgInput0,
                        ap_uint<16 * NPC2>* imgInput1,
                        ap_uint<8 * NPC1>* imgOutput0,
                        ap_uint<8 * NPC1>* imgOutput1,
                        ap_uint<8 * NPC1>* imgOutput2);
void cvtcolor_nv122rgb(ap_uint<8 * NPC1>* imgInput0, ap_uint<16 * NPC2>* imgInput1, ap_uint<32 * NPC1>* imgOutput);
void cvtcolor_nv122bgr(ap_uint<8 * NPC1>* imgInput0, ap_uint<16 * NPC2>* imgInput1, ap_uint<32 * NPC1>* imgOutput);
void cvtcolor_nv122uyvy(ap_uint<8 * NPC1>* imgInput0, ap_uint<16 * NPC2>* imgInput1, ap_uint<16 * NPC1>* imgOutput);
void cvtcolor_nv122yuyv(ap_uint<8 * NPC1>* imgInput0, ap_uint<16 * NPC2>* imgInput1, ap_uint<16 * NPC1>* imgOutput);
void cvtcolor_nv122nv21(ap_uint<8 * NPC1>* imgInput0,
                        ap_uint<16 * NPC2>* imgInput1,
                        ap_uint<8 * NPC1>* imgOutput0,
                        ap_uint<16 * NPC2>* imgOutput1);
void cvtcolor_nv212iyuv(ap_uint<8 * NPC1>* imgInput0,
                        ap_uint<16 * NPC2>* imgInput1,
                        ap_uint<8 * NPC1>* imgOutput0,
                        ap_uint<8 * NPC1>* imgOutput1,
                        ap_uint<8 * NPC1>* imgOutput2);
void cvtcolor_nv212rgba(ap_uint<8 * NPC1>* imgInput0, ap_uint<16 * NPC2>* imgInput1, ap_uint<32 * NPC1>* imgOutput);
void cvtcolor_nv212rgb(ap_uint<8 * NPC1>* imgInput0, ap_uint<16 * NPC2>* imgInput1, ap_uint<32 * NPC1>* imgOutput);
void cvtcolor_nv212bgr(ap_uint<8 * NPC1>* imgInput0, ap_uint<16 * NPC2>* imgInput1, ap_uint<32 * NPC1>* imgOutput);
void cvtcolor_nv212yuv4(ap_uint<8 * NPC1>* imgInput0,
                        ap_uint<16 * NPC2>* imgInput1,
                        ap_uint<8 * NPC1>* imgOutput0,
                        ap_uint<8 * NPC1>* imgOutput1,
                        ap_uint<8 * NPC1>* imgOutput2);
void cvtcolor_nv212uyvy(ap_uint<8 * NPC1>* imgInput0, ap_uint<16 * NPC2>* imgInput1, ap_uint<16 * NPC1>* imgOutput);
void cvtcolor_nv212yuyv(ap_uint<8 * NPC1>* imgInput0, ap_uint<16 * NPC2>* imgInput1, ap_uint<16 * NPC1>* imgOutput);
void cvtcolor_nv212nv12(ap_uint<8 * NPC1>* imgInput0,
                        ap_uint<16 * NPC2>* imgInput1,
                        ap_uint<8 * NPC1>* imgOutput0,
                        ap_uint<16 * NPC2>* imgOutput1);
void cvtcolor_uyvy2iyuv(ap_uint<16 * NPC1>* imgInput,
                        ap_uint<8 * NPC1>* imgOutput0,
                        ap_uint<8 * NPC1>* imgOutput1,
                        ap_uint<8 * NPC1>* imgOutput2);
void cvtcolor_uyvy2nv12(ap_uint<16 * NPC1>* imgInput, ap_uint<8 * NPC1>* imgOutput0, ap_uint<16 * NPC2>* imgOutput1);
void cvtcolor_uyvy2nv21(ap_uint<16 * NPC1>* imgInput, ap_uint<8 * NPC1>* imgOutput0, ap_uint<16 * NPC2>* imgOutput1);
void cvtcolor_uyvy2rgba(ap_uint<16 * NPC1>* imgInput, ap_uint<32 * NPC1>* imgOutput);
void cvtcolor_uyvy2rgb(ap_uint<16 * NPC1>* imgInput, ap_uint<32 * NPC1>* imgOutput);
void cvtcolor_uyvy2bgr(ap_uint<16 * NPC1>* imgInput, ap_uint<32 * NPC1>* imgOutput);
void cvtcolor_uyvy2yuyv(ap_uint<16 * NPC1>* imgInput, ap_uint<16 * NPC1>* imgOutput);
void cvtcolor_yuyv2iyuv(ap_uint<16 * NPC1>* imgInput,
                        ap_uint<8 * NPC1>* imgOutput0,
                        ap_uint<8 * NPC1>* imgOutput1,
                        ap_uint<8 * NPC1>* imgOutput2);
void cvtcolor_yuyv2nv12(ap_uint<16 * NPC1>* imgInput, ap_uint<8 * NPC1>* imgOutput0, ap_uint<16 * NPC2>* imgOutput1);
void cvtcolor_yuyv2nv21(ap_uint<16 * NPC1>* imgInput, ap_uint<8 * NPC1>* imgOutput0, ap_uint<16 * NPC2>* imgOutput1);
void cvtcolor_yuyv2rgba(ap_uint<16 * NPC1>* imgInput, ap_uint<32 * NPC1>* imgOutput);
void cvtcolor_yuyv2rgb(ap_uint<16 * NPC1>* imgInput, ap_uint<32 * NPC1>* imgOutput);
void cvtcolor_yuyv2bgr(ap_uint<16 * NPC1>* imgInput, ap_uint<32 * NPC1>* imgOutput);
void cvtcolor_yuyv2uyvy(ap_uint<16 * NPC1>* imgInput, ap_uint<16 * NPC1>* imgOutput);
void cvtcolor_rgb2gray(ap_uint<32 * NPC1>* imgInput, ap_uint<8 * NPC1>* imgOutput);
void cvtcolor_bgr2gray(ap_uint<32 * NPC1>* imgInput, ap_uint<8 * NPC1>* imgOutput);
void cvtcolor_gray2rgb(ap_uint<8 * NPC1>* imgInput, ap_uint<32 * NPC1>* imgOutput);
void cvtcolor_gray2bgr(ap_uint<8 * NPC1>* imgInput, ap_uint<32 * NPC1>* imgOutput);
void cvtcolor_rgb2xyz(ap_uint<32 * NPC1>* imgInput, ap_uint<32 * NPC1>* imgOutput);
void cvtcolor_bgr2xyz(ap_uint<32 * NPC1>* imgInput, ap_uint<32 * NPC1>* imgOutput);
void cvtcolor_xyz2rgb(ap_uint<32 * NPC1>* imgInput, ap_uint<32 * NPC1>* imgOutput);
void cvtcolor_xyz2bgr(ap_uint<32 * NPC1>* imgInput, ap_uint<32 * NPC1>* imgOutput);
void cvtcolor_rgb2ycrcb(ap_uint<32 * NPC1>* imgInput, ap_uint<32 * NPC1>* imgOutput);
void cvtcolor_bgr2ycrcb(ap_uint<32 * NPC1>* imgInput, ap_uint<32 * NPC1>* imgOutput);
void cvtcolor_ycrcb2rgb(ap_uint<32 * NPC1>* imgInput, ap_uint<32 * NPC1>* imgOutput);
void cvtcolor_ycrcb2bgr(ap_uint<32 * NPC1>* imgInput, ap_uint<32 * NPC1>* imgOutput);
void cvtcolor_rgb2hls(ap_uint<32 * NPC1>* imgInput, ap_uint<32 * NPC1>* imgOutput);
void cvtcolor_bgr2hls(ap_uint<32 * NPC1>* imgInput, ap_uint<32 * NPC1>* imgOutput);
void cvtcolor_hls2rgb(ap_uint<32 * NPC1>* imgInput, ap_uint<32 * NPC1>* imgOutput);
void cvtcolor_hls2bgr(ap_uint<32 * NPC1>* imgInput, ap_uint<32 * NPC1>* imgOutput);
void cvtcolor_rgb2hsv(ap_uint<32 * NPC1>* imgInput, ap_uint<32 * NPC1>* imgOutput);
void cvtcolor_bgr2hsv(ap_uint<32 * NPC1>* imgInput, ap_uint<32 * NPC1>* imgOutput);
void cvtcolor_hsv2rgb(ap_uint<32 * NPC1>* imgInput, ap_uint<32 * NPC1>* imgOutput);
void cvtcolor_hsv2bgr(ap_uint<32 * NPC1>* imgInput, ap_uint<32 * NPC1>* imgOutput);
#endif
