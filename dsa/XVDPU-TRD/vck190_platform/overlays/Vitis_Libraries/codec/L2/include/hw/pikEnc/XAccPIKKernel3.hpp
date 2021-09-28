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
 * @file XAccPIKKernel3.hpp
 */

#ifndef _XF_CODEC_XACCPIKKERNEL3_HPP_
#define _XF_CODEC_XACCPIKKERNEL3_HPP_

#include "kernel3/ac_tokenize.hpp"
#include "kernel3/ans.hpp"
#include "kernel3/build_table_encode_histo.hpp"

#include "kernel3/build_cluster.hpp"
#include "kernel3/ctrl_tokenize.hpp"
#include "kernel3/dc_shrink.hpp"
#include "kernel3/dc_tokenize.hpp"

#include "pik_common.hpp"

struct ConfigKernel3 {
    uint32_t xsize;
    uint32_t ysize;
    uint32_t xblock8;
    uint32_t yblock8;
    uint32_t xblock32;
    uint32_t yblock32;
    uint32_t xblock64;
    uint32_t yblock64;
    uint32_t ac_xgroup;
    uint32_t ac_ygroup;
    uint32_t dc_xgroup;
    uint32_t dc_ygroup;
    uint32_t ac_group;
    uint32_t dc_group;
    uint32_t num_dc;
    uint32_t num_ac;
};

#ifndef __SYNTHESIS__
#define DEBUG (1)
#define DEBUGAXItoPikAcStream
#define DEBUGAXItoStream
#endif

extern "C" void kernel3Top(ap_uint<32>* config,

                           ap_uint<32>* ddr_ac,
                           ap_uint<32>* ddr_dc,
                           ap_uint<32>* ddr_quant_field,
                           ap_uint<32>* ddr_ac_strategy,
                           ap_uint<32>* ddr_block,
                           ap_uint<32>* hls_order,

                           ap_uint<32>* histo_cfg,
                           ap_uint<32>* dc_histo_code_out,
                           ap_uint<32>* dc_code_out,
                           ap_uint<32>* ac_histo_code_out,
                           ap_uint<32>* ac_code_out);

#endif
