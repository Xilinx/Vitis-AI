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

/**
 * @file XAccPIKKernel2.hpp
 */

#ifndef _XF_CODEC_XACCPIKKERNEL2_HPP_
#define _XF_CODEC_XACCPIKKERNEL2_HPP_

#include <ap_int.h>
#include <hls_math.h>
#include <hls_stream.h>

#include "pik_common.hpp"

struct Config {
    uint32_t xsize;
    uint32_t ysize;
    uint32_t xblock8;
    uint32_t yblock8;
    uint32_t xblock32;
    uint32_t yblock32;
    uint32_t xgroup;
    uint32_t ygroup;

    int in_quant_field_num;
    int cmap_num0;
    int cmap_num1;
    int ac_num;
    int dc_num;
    int acs_num;
    int out_quant_field_num;

    bool kChooseAcStrategy;
    float discretization_factor;
    float kMulInhomogeneity16x16;
    float kMulInhomogeneity32x32;
    float butteraugli_target;
    float intensity_multiplier;
    float quant_dc;

    int src_num[3];
    int src_offset[3];
};

struct Quantizer {
    int quant_dc;
    int global_scale;
    float global_scale_float;
    float inv_global_scale;
    float inv_quant_dc;
};

extern "C" void kernel2Top(ap_uint<AXI_SZ> config[MAX_NUM_CONFIG],

                           ap_uint<2 * AXI_SZ> src[AXI_OUT / 2],
                           ap_uint<AXI_SZ> quant_field_in[AXI_QF],
                           ap_uint<AXI_SZ> cmap[AXI_CMAP],

                           ap_uint<AXI_SZ> ac[MAX_NUM_AC],
                           ap_uint<AXI_SZ> dc[MAX_NUM_DC],
                           ap_uint<AXI_SZ> quant_field_out[AXI_QF],
                           ap_uint<AXI_SZ> ac_strategy[MAX_NUM_BLOCK88],
                           ap_uint<AXI_SZ> block[MAX_NUM_BLOCK88],
                           ap_uint<AXI_SZ> order[MAX_NUM_ORDER]);

#endif
