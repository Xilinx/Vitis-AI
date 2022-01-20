/*
 * Copyright 2021 Xilinx, Inc.
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

#ifndef _XF_RGBIR_CONFIG_H_
#define _XF_RGBIR_CONFIG_H_

#include "hls_stream.h"
#include "ap_int.h"
#include "common/xf_common.hpp"
#include "ap_axi_sdata.h"
#include "common/xf_infra.hpp"
#include "common/xf_axi_io.hpp"

#include "imgproc/xf_rgbir.hpp"
#include "xf_config_params.h"

// Resolve input and output pixel type:
#if T_8U
#define IN_TYPE XF_8UC1
#define OUT_TYPE XF_8UC1
#define CVTYPE unsigned char
#endif
#if T_10U
#define IN_TYPE XF_10UC1
#define OUT_TYPE XF_10UC1
#define CVTYPE unsigned short
#endif
#if T_12U
#define IN_TYPE XF_12UC1
#define OUT_TYPE XF_12UC1
#define CVTYPE unsigned short
#endif
#if T_16U
#define IN_TYPE XF_16UC1
#define OUT_TYPE XF_16UC1
#define CVTYPE unsigned short
#endif

#if (T_16U || T_10U || T_12U)
#define CV_INTYPE CV_16UC1
#define CV_OUTTYPE CV_16UC1
#else
#define CV_INTYPE CV_8UC1
#define CV_OUTTYPE CV_8UC1
#endif

#define _DATA_WIDTH_(_T, _N) (XF_PIXELWIDTH(_T, _N) * XF_NPIXPERCYCLE(_N))
#define _BYTE_ALIGN_(_N) ((((_N) + 7) / 8) * 8)

#define IN_DATA_WIDTH _DATA_WIDTH_(IN_TYPE, NPC)
#define OUT_DATA_WIDTH _DATA_WIDTH_(OUT_TYPE, NPC)

#define AXI_WIDTH_IN _BYTE_ALIGN_(IN_DATA_WIDTH)
#define AXI_WIDTH_OUT _BYTE_ALIGN_(OUT_DATA_WIDTH)

// Input/Output AXI video buses
typedef ap_axiu<AXI_WIDTH_IN, 1, 1, 1> InStrmBus_t;
typedef ap_axiu<AXI_WIDTH_OUT, 1, 1, 1> OutStrmBus_t;

// Input/Output AXI video stream
typedef hls::stream<InStrmBus_t> InStream;
typedef hls::stream<OutStrmBus_t> OutStream;

static constexpr int FILTERSIZE1 = 5, FILTERSIZE2 = 3;
#define ERROR_THRESHOLD_RGB 11
#define ERROR_THRESHOLD_IR 3

void rgbir_accel(InStream& img_in,
                 OutStream& rggb_out,
                 OutStream& ir_out,
                 char* R_IR_C1_wgts,
                 char* R_IR_C2_wgts,
                 char* B_at_R_wgts,
                 char* IR_at_R_wgts,
                 char* IR_at_B_wgts,
                 char* sub_wgts,
                 int height,
                 int width);

#endif // _XF_RGBIR_CONFIG_H_
