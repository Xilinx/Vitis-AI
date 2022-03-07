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

/*  set the optimisation type  */
#define NO 0 // Normal Operation
#define RO 1 // Resource Optimized

// Check if define already passed in command line
#if !(defined(BGR2NV12) || defined(BGR2NV21) || defined(NV122BGR) || defined(NV122IYUV) || defined(NV122NV21) ||   \
      defined(NV122UYVY) || defined(NV122YUV4) || defined(NV122YUYV) || defined(NV212BGR) || defined(NV212IYUV) || \
      defined(NV212NV12) || defined(NV212UYVY) || defined(NV212YUYV) || defined(RGB2IYUV) || defined(RGB2NV12) ||  \
      defined(RGB2NV21) || defined(RGB2UYVY) || defined(RGB2YUV4) || defined(RGB2YUYV) || defined(RGBA2IYUV) ||    \
      defined(UYVY2IYUV) || defined(UYVY2NV12) || defined(UYVY2NV21) || defined(UYVY2RGB) || defined(UYVY2YUYV) || \
      defined(YUYV2IYUV) || defined(YUYV2NV12) || defined(YUYV2RGBA) || defined(YUYV2UYVY))

#ifndef RGBA2IYUV
#define RGBA2IYUV 0
#endif

#ifndef RGBA2NV12
#define RGBA2NV12 0
#endif

#ifndef RGBA2NV21
#define RGBA2NV21 0
#endif

#ifndef RGBA2YUV4
#define RGBA2YUV4 0
#endif

#ifndef RGB2IYUV
#define RGB2IYUV 0
#endif

#ifndef RGB2NV12
#define RGB2NV12 0
#endif

#ifndef RGB2NV21
#define RGB2NV21 0
#endif

#ifndef RGB2YUV4
#define RGB2YUV4 0
#endif

#ifndef RGB2UYVY
#define RGB2UYVY 0
#endif

#ifndef RGB2YUYV
#define RGB2YUYV 0
#endif

#ifndef RGB2BGR
#define RGB2BGR 0
#endif

#ifndef BGR2UYVY
#define BGR2UYVY 0
#endif

#ifndef BGR2YUYV
#define BGR2YUYV 0
#endif

#ifndef BGR2RGB
#define BGR2RGB 0
#endif

#ifndef BGR2NV12
#define BGR2NV12 0
#endif

#ifndef BGR2NV21
#define BGR2NV21 0
#endif

#ifndef IYUV2NV12
#define IYUV2NV12 0
#endif

#ifndef IYUV2RGBA
#define IYUV2RGBA 0
#endif

#ifndef IYUV2RGB
#define IYUV2RGB 1
#endif

#ifndef IYUV2YUV4
#define IYUV2YUV4 0
#endif

#ifndef NV122IYUV
#define NV122IYUV 0
#endif

#ifndef NV122RGBA
#define NV122RGBA 0
#endif

#ifndef NV122YUV4
#define NV122YUV4 0
#endif

#ifndef NV122RGB
#define NV122RGB 0
#endif

#ifndef NV122BGR
#define NV122BGR 0
#endif

#ifndef NV122UYVY
#define NV122UYVY 0
#endif

#ifndef NV122YUYV
#define NV122YUYV 0
#endif

#ifndef NV122NV21
#define NV122NV21 0
#endif

#ifndef NV212IYUV
#define NV212IYUV 0
#endif

#ifndef NV212RGBA
#define NV212RGBA 0
#endif

#ifndef NV212RGB
#define NV212RGB 0
#endif

#ifndef NV212BGR
#define NV212BGR 0
#endif

#ifndef NV212YUV4
#define NV212YUV4 0
#endif

#ifndef NV212UYVY
#define NV212UYVY 0
#endif

#ifndef NV212YUYV
#define NV212YUYV 0
#endif

#ifndef NV212NV12
#define NV212NV12 0
#endif

#ifndef UYVY2IYUV
#define UYVY2IYUV 0
#endif

#ifndef UYVY2NV12
#define UYVY2NV12 0
#endif

#ifndef UYVY2NV21
#define UYVY2NV21 0
#endif

#ifndef UYVY2RGBA
#define UYVY2RGBA 0
#endif

#ifndef UYVY2RGB
#define UYVY2RGB 0
#endif

#ifndef UYVY2BGR
#define UYVY2BGR 0
#endif

#ifndef UYVY2YUYV
#define UYVY2YUYV 0
#endif

#ifndef YUYV2IYUV
#define YUYV2IYUV 0
#endif

#ifndef YUYV2NV12
#define YUYV2NV12 0
#endif

#ifndef YUYV2NV21
#define YUYV2NV21 0
#endif

#ifndef YUYV2RGBA
#define YUYV2RGBA 0
#endif

#ifndef YUYV2RGB
#define YUYV2RGB 0
#endif

#ifndef YUYV2BGR
#define YUYV2BGR 0
#endif

#ifndef YUYV2UYVY
#define YUYV2UYVY 0
#endif

#ifndef RGB2GRAY
#define RGB2GRAY 0
#endif

#ifndef BGR2GRAY
#define BGR2GRAY 0
#endif

#ifndef GRAY2RGB
#define GRAY2RGB 0
#endif

#ifndef GRAY2BGR
#define GRAY2BGR 0
#endif

#ifndef RGB2XYZ
#define RGB2XYZ 0
#endif

#ifndef BGR2XYZ
#define BGR2XYZ 0
#endif

#ifndef XYZ2RGB
#define XYZ2RGB 0
#endif

#ifndef XYZ2BGR
#define XYZ2BGR 0
#endif

#ifndef RGB2YCrCb
#define RGB2YCrCb 0
#endif

#ifndef BGR2YCrCb
#define BGR2YCrCb 0
#endif

#ifndef YCrCb2RGB
#define YCrCb2RGB 0
#endif

#ifndef YCrCb2BGR
#define YCrCb2BGR 0
#endif

#ifndef RGB2HLS
#define RGB2HLS 0
#endif

#ifndef BGR2HLS
#define BGR2HLS 0
#endif

#ifndef HLS2RGB
#define HLS2RGB 0
#endif

#ifndef HLS2BGR
#define HLS2BGR 0
#endif

#ifndef RGB2HSV
#define RGB2HSV 0
#endif

#ifndef BGR2HSV
#define BGR2HSV 0
#endif

#ifndef HSV2RGB
#define HSV2RGB 0
#endif

#ifndef HSV2BGR
#define HSV2BGR 0
#endif

#endif
