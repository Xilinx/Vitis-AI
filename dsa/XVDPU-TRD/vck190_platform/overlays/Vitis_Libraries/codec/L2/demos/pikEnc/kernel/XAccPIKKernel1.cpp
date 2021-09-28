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

#include "pikEnc/XAccPIKKernel1.hpp"

extern "C" void kernel1Top(ap_uint<32> config[MAX_NUM_CONFIG],
                           ap_uint<AXI_WIDTH> rbuf[BUF_DEPTH / 2],
                           ap_uint<32> axi_out[AXI_OUT],
                           ap_uint<32> axi_cmap[AXI_CMAP],
                           ap_uint<32> axi_qf[AXI_QF]) {
#pragma HLS INTERFACE m_axi offset = slave latency = 8 num_write_outstanding = 4 num_read_outstanding = \
    4 max_write_burst_length = 8 max_read_burst_length = 8 bundle = gmem0_0 port = config
#pragma HLS INTERFACE s_axilite port = config bundle = control

#pragma HLS INTERFACE m_axi offset = slave latency = 64 num_write_outstanding = 4 num_read_outstanding = \
    8 max_write_burst_length = 8 max_read_burst_length = 256 bundle = gmem0_1 port = rbuf
#pragma HLS INTERFACE s_axilite port = rbuf bundle = control

#pragma HLS INTERFACE m_axi offset = slave latency = 64 num_write_outstanding = 16 num_read_outstanding = \
    4 max_write_burst_length = 128 max_read_burst_length = 8 bundle = gmem1_0 port = axi_out
#pragma HLS INTERFACE s_axilite port = axi_out bundle = control

#pragma HLS INTERFACE m_axi offset = slave latency = 64 num_write_outstanding = 16 num_read_outstanding = \
    4 max_write_burst_length = 32 max_read_burst_length = 8 bundle = gmem1_1 port = axi_cmap
#pragma HLS INTERFACE s_axilite port = axi_cmap bundle = control

#pragma HLS INTERFACE m_axi offset = slave latency = 64 num_write_outstanding = 16 num_read_outstanding = \
    4 max_write_burst_length = 32 max_read_burst_length = 8 bundle = gmem1_2 port = axi_qf
#pragma HLS INTERFACE s_axilite port = axi_qf bundle = control

#pragma HLS INTERFACE s_axilite port = return bundle = control

    int len[3];
    int offsets[3];
    int xsize;
    int ysize;
    float quant_ac;

    loadConfig(config, len, offsets, xsize, ysize, quant_ac);

    kernel1_core(rbuf, len, offsets, xsize, ysize, quant_ac, axi_out, axi_cmap, axi_qf);
}
