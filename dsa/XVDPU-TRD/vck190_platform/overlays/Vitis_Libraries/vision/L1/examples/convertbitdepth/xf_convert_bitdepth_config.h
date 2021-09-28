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

#ifndef _XF_CONVERT_BITDEPTH_CONFIG_H_
#define _XF_CONVERT_BITDEPTH_CONFIG_H_

#include "ap_int.h"
#include "hls_stream.h"
#include "common/xf_common.hpp"
#include "common/xf_utility.hpp"
#include "xf_config_params.h"
#include "core/xf_convert_bitdepth.hpp"

#define HEIGHT 128
#define WIDTH 128

// Resolve optimization type:
#if RO
#define NPC1 XF_NPPC8
#elif NO
#define NPC1 XF_NPPC1
#endif

// Resolve bit depth conversion type:
#if XF_CONVERT8UTO16U
#define IN_TYPE XF_8UC1
#define OUT_TYPE XF_16UC1
#define IN_BYTE 1
#define OUT_BYTE 2
#define CONVERT_TYPE XF_CONVERT_8U_TO_16U
#define OCV_INTYPE CV_8UC1
#define OCV_OUTTYPE CV_16UC1
#if NO
#define INPUT_PTR_WIDTH 8
#define OUTPUT_PTR_WIDTH 16
#else
#define INPUT_PTR_WIDTH 64
#define OUTPUT_PTR_WIDTH 128
#endif
#endif

#if XF_CONVERT8UTO16S
#define IN_TYPE XF_8UC1
#define OUT_TYPE XF_16SC1
#define CONVERT_TYPE XF_CONVERT_8U_TO_16S
#define IN_BYTE 1
#define OUT_BYTE 2
#define OCV_INTYPE CV_8UC1
#define OCV_OUTTYPE CV_16SC1
#if NO
#define INPUT_PTR_WIDTH 8
#define OUTPUT_PTR_WIDTH 16
#else
#define INPUT_PTR_WIDTH 64
#define OUTPUT_PTR_WIDTH 128
#endif
#endif

#if XF_CONVERT8UTO32S
#define IN_TYPE XF_8UC1
#define OUT_TYPE XF_32SC1
#define CONVERT_TYPE XF_CONVERT_8U_TO_32S
#define IN_BYTE 1
#define OUT_BYTE 4
#define OCV_INTYPE CV_8UC1
#define OCV_OUTTYPE CV_32SC1
#if NO
#define INPUT_PTR_WIDTH 8
#define OUTPUT_PTR_WIDTH 32
#else
#define INPUT_PTR_WIDTH 64
#define OUTPUT_PTR_WIDTH 256
#endif
#endif

#if XF_CONVERT16UTO32S
#define IN_TYPE XF_16UC1
#define OUT_TYPE XF_32SC1
#define CONVERT_TYPE XF_CONVERT_16U_TO_32S
#define IN_BYTE 2
#define OUT_BYTE 4
#define OCV_INTYPE CV_16UC1
#define OCV_OUTTYPE CV_32SC1
#if NO
#define INPUT_PTR_WIDTH 16
#define OUTPUT_PTR_WIDTH 32
#else
#define INPUT_PTR_WIDTH 128
#define OUTPUT_PTR_WIDTH 256
#endif
#endif

#if XF_CONVERT16STO32S
#define IN_TYPE XF_16SC1
#define OUT_TYPE XF_32SC1
#define CONVERT_TYPE XF_CONVERT_16S_TO_32S
#define IN_BYTE 2
#define OUT_BYTE 4
#define OCV_INTYPE CV_16SC1
#define OCV_OUTTYPE CV_32SC1
#if NO
#define INPUT_PTR_WIDTH 16
#define OUTPUT_PTR_WIDTH 32
#else
#define INPUT_PTR_WIDTH 128
#define OUTPUT_PTR_WIDTH 256
#endif
#endif

#if XF_CONVERT16UTO8U
#define IN_TYPE XF_16UC1
#define OUT_TYPE XF_8UC1
#define IN_BYTE 2
#define OUT_BYTE 1
#define CONVERT_TYPE XF_CONVERT_16U_TO_8U
#define OCV_INTYPE CV_16UC1
#define OCV_OUTTYPE CV_8UC1
#if NO
#define INPUT_PTR_WIDTH 16
#define OUTPUT_PTR_WIDTH 8
#else
#define INPUT_PTR_WIDTH 128
#define OUTPUT_PTR_WIDTH 64
#endif
#endif

#if XF_CONVERT16STO8U
#define IN_TYPE XF_16SC1
#define OUT_TYPE XF_8UC1
#define IN_BYTE 2
#define OUT_BYTE 1
#define CONVERT_TYPE XF_CONVERT_16S_TO_8U
#define OCV_INTYPE CV_16SC1
#define OCV_OUTTYPE CV_8UC1
#if NO
#define INPUT_PTR_WIDTH 16
#define OUTPUT_PTR_WIDTH 8
#else
#define INPUT_PTR_WIDTH 128
#define OUTPUT_PTR_WIDTH 64
#endif
#endif

#if XF_CONVERT32STO8U
#define IN_TYPE XF_32SC1
#define OUT_TYPE XF_8UC1
#define CONVERT_TYPE XF_CONVERT_32S_TO_8U
#define IN_BYTE 4
#define OUT_BYTE 1
#define OCV_INTYPE CV_32SC1
#define OCV_OUTTYPE CV_8UC1
#if NO
#define INPUT_PTR_WIDTH 32
#define OUTPUT_PTR_WIDTH 8
#else
#define INPUT_PTR_WIDTH 256
#define OUTPUT_PTR_WIDTH 64
#endif
#endif

#if XF_CONVERT32STO16U
#define IN_TYPE XF_32SC1
#define OUT_TYPE XF_16UC1
#define CONVERT_TYPE XF_CONVERT_32S_TO_16U
#define IN_BYTE 4
#define OUT_BYTE 2
#define OCV_INTYPE CV_32SC1
#define OCV_OUTTYPE CV_16UC1
#if NO
#define INPUT_PTR_WIDTH 32
#define OUTPUT_PTR_WIDTH 16
#else
#define INPUT_PTR_WIDTH 256
#define OUTPUT_PTR_WIDTH 128
#endif
#endif

#if XF_CONVERT32STO16S
#define IN_TYPE XF_32SC1
#define OUT_TYPE XF_16SC1
#define CONVERT_TYPE XF_CONVERT_32S_TO_16S
#define IN_BYTE 4
#define OUT_BYTE 2
#define OCV_INTYPE CV_32SC1
#define OCV_OUTTYPE CV_16SC1
#if NO
#define INPUT_PTR_WIDTH 32
#define OUTPUT_PTR_WIDTH 16
#else
#define INPUT_PTR_WIDTH 256
#define OUTPUT_PTR_WIDTH 128
#endif
#endif

void convert_bitdepth_accel(
    ap_uint<INPUT_PTR_WIDTH>* img_in, int shift, ap_uint<OUTPUT_PTR_WIDTH>* img_out, int height, int width);

#endif // _XF_CONVERT_BITDEPTH_CONFIG_H_
