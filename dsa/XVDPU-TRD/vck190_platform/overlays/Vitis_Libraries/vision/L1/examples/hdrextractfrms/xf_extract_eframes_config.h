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

#ifndef __XF_EXTRACT_EFRAMES_CONFIG_H_
#define __XF_EXTRACT_EFRAMES_CONFIG_H_

#include "hls_stream.h"
#include "ap_int.h"
#include "common/xf_common.hpp"
//#include "common/xf_utility.h"
#include "ap_axi_sdata.h"
#include "common/xf_axi_io.hpp"
#include "common/xf_utility.hpp"
#include "imgproc/xf_extract_eframes.hpp"
#include "xf_config_params.h"
#include <ap_int.h>

// Useful macro functions definations
#define _DATA_WIDTH_(_T, _N) (XF_PIXELWIDTH(_T, _N) * XF_NPIXPERCYCLE(_N))
#define _BYTE_ALIGN_(_N) ((((_N) + 7) / 8) * 8)

#define IN_DATA_WIDTH _DATA_WIDTH_(XF_SRC_T, XF_NPPC)

#define AXI_WIDTH_IN _BYTE_ALIGN_(IN_DATA_WIDTH)
#define AXI_WIDTH_OUT _BYTE_ALIGN_(IN_DATA_WIDTH)

// --------------------------------------------------------------------
// Internal types
// --------------------------------------------------------------------
// Input/Output AXI video buses
typedef ap_axiu<AXI_WIDTH_IN, 1, 1, 1> InVideoStrmBus_t_e_s;
typedef ap_axiu<AXI_WIDTH_OUT, 1, 1, 1> OutVideoStrmBus_t_e_s;

// Input/Output AXI video stream
typedef hls::stream<InVideoStrmBus_t_e_s> InVideoStrm_t_e_s;
typedef hls::stream<OutVideoStrmBus_t_e_s> OutVideoStrm_t_e_s;

void extractEFrames_accel(InVideoStrm_t_e_s& in_ptr,
                          OutVideoStrm_t_e_s& lef_ptr,
                          OutVideoStrm_t_e_s& sef_ptr,
                          ap_uint<IMAGE_SIZE_WIDTH> height,
                          ap_uint<IMAGE_SIZE_WIDTH> width);

#endif //__XF_EXTRACT_EFRAMES_CONFIG_H_
