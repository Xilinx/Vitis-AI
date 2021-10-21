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

#ifndef _XF_CODEC_HOST_DEV_HPP_
#define _XF_CODEC_HOST_DEV_HPP_

#include <ap_int.h>
#include <hls_math.h>
#include <hls_stream.h>

#include "pik_common.hpp"

#ifndef HLS_TEST
void host_func(std::string xclbinPath,
               float* dataDDR,
               ap_uint<AXI_SZ> k1_config[MAX_NUM_CONFIG],
               ap_uint<AXI_SZ> k2_config[MAX_NUM_CONFIG],
               ap_uint<AXI_SZ> k3_config[MAX_NUM_CONFIG],

               ap_uint<AXI_SZ> cmap[AXI_CMAP],
               ap_uint<AXI_SZ> order[MAX_NUM_ORDER],
               ap_uint<AXI_SZ> quant_field[AXI_QF],

               int len_dc_histo[2 * MAX_DC_GROUP],
               int len_dc[2 * MAX_DC_GROUP],
               ap_uint<AXI_SZ> dc_histo_code_out[2 * MAX_DC_GROUP * MAX_DC_HISTO_SIZE],
               ap_uint<AXI_SZ> dc_code_out[2 * MAX_DC_GROUP * MAX_DC_SIZE],

               int len_ac_histo[MAX_AC_GROUP],
               int len_ac[MAX_AC_GROUP],
               ap_uint<AXI_SZ> ac_histo_code_out[MAX_AC_GROUP * MAX_AC_HISTO_SIZE],
               ap_uint<AXI_SZ> ac_code_out[MAX_AC_GROUP * MAX_AC_SIZE]);
#endif

#endif
