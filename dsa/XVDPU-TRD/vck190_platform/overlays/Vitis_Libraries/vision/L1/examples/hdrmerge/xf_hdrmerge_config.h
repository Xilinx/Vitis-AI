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
#ifndef _XF_HDRMERGE_CONFIG_H_
#define _XF_HDRMERGE_CONFIG_H_

#include "hls_stream.h"
#include "ap_int.h"

#include "common/xf_common.hpp"
#include "ap_axi_sdata.h"
#include "common/xf_axi_io.hpp"
#include "common/xf_utility.hpp"
#include "imgproc/xf_hdrmerge.hpp"

#include "xf_config_params.h"

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

// Useful macro functions definations
#define _DATA_WIDTH_(_T, _N) (XF_PIXELWIDTH(_T, _N) * XF_NPIXPERCYCLE(_N))
#define _BYTE_ALIGN_(_N) ((((_N) + 7) / 8) * 8)

#define IN_DATA_WIDTH _DATA_WIDTH_(IN_TYPE, NPIX)

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

void hdrmerge_accel(InVideoStrm_t_e_s& img_in1,
                    InVideoStrm_t_e_s& img_in2,
                    OutVideoStrm_t_e_s& img_out,
                    int rows,
                    int cols,
                    short wr_hls[NO_EXPS * NPIX * W_B_SIZE]);
#endif
