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

#ifndef _XF_ISP_TYPES_H_
#define _XF_ISP_TYPES_H_

// --------------------------------------------------------------------
// Required files
// --------------------------------------------------------------------
#include "hls_stream.h"
#include "ap_int.h"
#include "common/xf_common.hpp"
//#include "common/xf_utility.h"
#include "ap_axi_sdata.h"
#include "common/xf_axi_io.hpp"
#include "xf_config_params.h"

// Requried Vision modules
#include "imgproc/xf_bpc.hpp"
#include "imgproc/xf_gaincontrol.hpp"
#include "imgproc/xf_autowhitebalance.hpp"
#include "imgproc/xf_demosaicing.hpp"
#include "imgproc/xf_ltm.hpp"
#include "imgproc/xf_quantizationdithering.hpp"
#include "imgproc/xf_lensshadingcorrection.hpp"
#include "imgproc/xf_colorcorrectionmatrix.hpp"
#include "imgproc/xf_black_level.hpp"
#include "imgproc/xf_aec.hpp"
#include "imgproc/xf_cvt_color.hpp"
#include "imgproc/xf_cvt_color_1.hpp"
#include "imgproc/xf_gammacorrection.hpp"
#include "imgproc/xf_hdrmerge.hpp"

#define S_DEPTH 4096

#define NO_EXPS 2

#if T_8U
#define W_B_SIZE 256
#endif
#if T_10U
#define W_B_SIZE 1024
#endif
#if T_12U
#define W_B_SIZE 4096
#endif
#if T_16U
#define W_B_SIZE 65536
#endif

// --------------------------------------------------------------------
// Macros definations
// --------------------------------------------------------------------

// Useful macro functions definations
#define _DATA_WIDTH_(_T, _N) (XF_PIXELWIDTH(_T, _N) * XF_NPIXPERCYCLE(_N))
#define _BYTE_ALIGN_(_N) ((((_N) + 7) / 8) * 8)

#define IN_DATA_WIDTH _DATA_WIDTH_(XF_SRC_T, XF_NPPC)
//#define OUT_DATA_WIDTH _DATA_WIDTH_(XF_DST_T, XF_NPPC)
//#define OUT_DATA_WIDTH _DATA_WIDTH_(XF_LTM_T, XF_NPPC)
#define OUT_DATA_WIDTH _DATA_WIDTH_(XF_16UC1, XF_NPPC)

#define AXI_WIDTH_IN _BYTE_ALIGN_(IN_DATA_WIDTH)
#define AXI_WIDTH_OUT _BYTE_ALIGN_(OUT_DATA_WIDTH)

#define NR_COMPONENTS 3
static constexpr int BLOCK_HEIGHT = 64;
static constexpr int BLOCK_WIDTH = 64;
// --------------------------------------------------------------------
// Internal types
// --------------------------------------------------------------------
// Input/Output AXI video buses
typedef ap_axiu<AXI_WIDTH_IN, 1, 1, 1> InVideoStrmBus_t;
typedef ap_axiu<AXI_WIDTH_OUT, 1, 1, 1> OutVideoStrmBus_t;

// Input/Output AXI video stream
typedef hls::stream<InVideoStrmBus_t> InVideoStrm_t;
typedef hls::stream<OutVideoStrmBus_t> OutVideoStrm_t;

#if T_8U
#define HIST_SIZE 256
#elif T_10U
#define HIST_SIZE 1024
#else
#define HIST_SIZE 4096
#endif

#define BLACK_LEVEL 32
#define MAX_PIX_VAL (1 << (XF_DTPIXELDEPTH(XF_SRC_T, XF_NPPC))) - 1

// HW Registers
typedef struct {
    uint16_t width;
    uint16_t height;
    //    uint16_t video_format;
    uint16_t bayer_phase;
} HW_STRUCT_REG;

#endif //_XF_ISP_TYPES_H_
