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

// Set Conversion type
#define IYUV2NV12 0
#define IYUV2RGBA 0
#define IYUV2RGB 0
#define IYUV2YUV4 0

#define NV122IYUV 0
#define NV122RGBA 0
#define NV122RGB 0
#define NV122BGR 0
#define NV122YUV4 0

#define NV212IYUV 0
#define NV212RGBA 0
#define NV212RGB 0
#define NV212BGR 0
#define NV212YUV4 0

#define RGBA2IYUV 0
#define RGBA2NV12 0
#define RGBA2NV21 0
#define RGBA2YUV4 0

#define RGB2IYUV 0
#define RGB2NV12 0
#define RGB2NV21 0
#define RGB2YUV4 0

#define UYVY2IYUV 0
#define UYVY2NV12 0
#define UYVY2RGBA 0
#define UYVY2RGB 0

#define YUYV2IYUV 0
#define YUYV2NV12 0
#define YUYV2RGBA 0
#define YUYV2RGB 0

#define RGB2GRAY 0
#define BGR2GRAY 1
#define GRAY2RGB 0
#define GRAY2BGR 0

#define RGB2XYZ 0
#define BGR2XYZ 0
#define XYZ2RGB 0
#define XYZ2BGR 0

#define RGB2YCrCb 0
#define BGR2YCrCb 0
#define YCrCb2RGB 0
#define YCrCb2BGR 0

#define RGB2HSV 0
#define BGR2HSV 0
#define HSV2RGB 0
#define HSV2BGR 0

#define RGB2HLS 0
#define BGR2HLS 0
#define HLS2RGB 0
#define HLS2BGR 0

/*  set the optimisation type  */
#define NO 1 // Normal Operation
#define RO 0 // Resource Optimized
